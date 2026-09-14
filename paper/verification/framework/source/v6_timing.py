"""v5 时机实验：冻结策略（三个 v4r 仅监督 S 权重），检测器/编码器/actor/记忆规则全部冻结，只改变
(1) 信号情境：需要停车 / 允许通行（外部信号相位偏移），(2) 执行条件：jerk 上限 2 / 4 m/s³（同时改 curve_layer.J/Q 与 env20.J_MAX），
(3) 视觉条件：正常可见 / 中等延迟 / 较晚可见（整个信号灯在指定位置前不出现在图像里——检测与编码两条路径同时不可见）。
从一组预先冻结的接近状态（灯尚不可见处）出发的短程分支，记录“机会如何消失”的四个时刻与停车余量 M_j(t)=d_line−D_stop,j(v,a)。
D_stop,j 用现有 curve_layer.backup_distance（该备份制动的停车距离；负余量只说明该备份失效，不等于物理不可能）。

子命令：
  freeze   --policy runs/v4r/v4_pilot/supervised/final_nets.pt --x-freeze 2740 → runs/v5_timing/frozen_state.pkl
  select   → 由冻结状态与备份制动距离计算候选延迟位置（写 runs/v5_timing/delays.json，含推导过程）
  run      --policies ... --situations stop,pass --jerks 2,4 --visuals normal,medium,late [--budget-s 150] → 每分支一个 JSON（配置哈希命名，已完成跳过）
  summarize → 汇总表
"""
import argparse, copy, hashlib, json, math, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
torch.set_num_threads(1)
import pipeline as P
from pipeline import S, Perception, VisionEnv, FrameStore, obs_from_frames, STACK
from renderer import SceneCamera, signal_color, LIGHT_AHEAD
from vision_state import VisionMemory
from reevaluate import learner_from_final_nets
import curve_layer as C
E = S.E
XS = float(S.R.SIGNALS[0]); X_END_BRANCH = XS + 300.; OUT = ROOT / 'runs' / 'v5_timing'
PHASE_IDX = {'red': 0, 'yellow': 1, 'green': 2, 'off': 3, 'unknown': 4}


def set_vfree(v):
    """v6：第二组冻结状态用较低巡航速度——只把记忆给执行层的速度上限 v_limit() 封顶到 v（策略输入归一化 V_FREE 不变，避免策略输入分布偏移）。None 表示恢复默认。"""
    import vision_state as VS
    if not hasattr(VS.VisionMemory, '_v_limit_orig'): VS.VisionMemory._v_limit_orig = VS.VisionMemory.v_limit
    if v is None: VS.VisionMemory.v_limit = VS.VisionMemory._v_limit_orig; return None
    cap = float(v); VS.VisionMemory.v_limit = lambda self: min(VS.VisionMemory._v_limit_orig(self), cap); return cap


def set_jerk_limit(j):
    """统一执行层（curve_layer 备份/区间）、基础环境（env20 投影裁剪与越界计数）的 jerk 上限。"""
    C.J = float(j); C.Q = C.J * C.DT; E.J_MAX = float(j)
    return dict(curve_layer_J=C.J, curve_layer_Q=C.Q, env20_J_MAX=E.J_MAX)


def stop_distance(v, a, j):
    """给定 jerk 上限 j 的备份制动停车距离（curve_layer.backup_distance，当前离散动力学）；不可行返回 inf。"""
    old = (C.J, C.Q); C.J = float(j); C.Q = C.J * C.DT
    try: return float(C.backup_distance(float(v), float(a), 0.))
    finally: C.J, C.Q = old


def margin_exec(d):
    return d - (0.15 * d + 8.)   # v4r 记忆停车余量：执行层目标点 = 线前 0.15d+8


def load_cfg(name='v4r_cpu.json'):
    return json.load(open(ROOT / 'configs' / name))


def make_perception(cfg, encoder=None):
    return Perception(ROOT / (encoder or cfg['pretrained_encoder']), 'roi_head', det_thr=cfg['det_thr']['traffic_light'])


def load_policy(pol):
    """'constant' → 常量命令 u=+1（执行族参照，无 actor）；否则 final_nets.pt。"""
    if pol == 'constant':
        from v4_probe_closed_loop import ConstantLearner
        L = ConstantLearner(1.0); L._path = 'constant'; return L, {}
    L, ck = learner_from_final_nets(ROOT / pol); L._path = pol; return L, ck


def cap(camera, env, store, hist, eid, per, memory, mask_color=False):
    f = camera.capture(episode_id=eid, sim_time=env.t, pose=dict(x=env.x, v=env.v, offsets=env.offsets))
    fid = f"{eid}_{f['meta']['frame_index']:05d}"; store.put(fid, f); hist.append(fid); del hist[:-STACK]
    dets, probs = per.observe(store, hist, eid, env.t)
    if mask_color: probs = None; per.last['probs'] = None   # v6 'color_mask'：灯色在感知输出处延迟（图像与检测照常；灯位置可知、灯色未知）
    memory.update(dets, probs, env.v, E.DT if len(hist) > 1 else 0.)
    env.set_perceived(memory.sig['phase'], memory.sig['age'] if memory.sig['seen'] else 0., 'vision_memory'); env.perceived['detail'] = dict(per.last, memory=memory.snapshot())
    return dets, probs


def act(learner, memory, env, store, hist, eid):
    rec = dict(frame_ids=list(hist), episode_id=eid, decision_time=float(env.t), legacy=memory.legacy(env.v, env.a, env.soc, env.T, env.t_end, env.t, env.t_budget),
               association_valid=memory.light_detected(), extra=memory.extra(extended=int(getattr(getattr(learner, 'cfg', None), 'extra_dim', 6)) == 10))
    obs = obs_from_frames(store, [rec])
    with torch.no_grad(): out = learner.encoder(obs); u = float(learner.actor(learner.adapter(obs, out['z']))[0, 0])
    return u


# ------------------------------------------------------------------ 冻结状态
def freeze(a):
    cfg = load_cfg(a.config); mk = {k: cfg[k] for k in ('det_thr', 'hold_s')}; per = make_perception(cfg)
    capped = False   # v6：速度封顶只从 --cap-from 位置起生效（在弯道区封顶会因帧不同触发解除牌误检为红灯的已知幻影死锁），到冻结点时车速已稳定在封顶值
    learner, ck = learner_from_final_nets(ROOT / a.policy); conds = S.conditions('development'); soc, temp, v0, off = conds[a.cond]
    memory = VisionMemory(**mk); env = VisionEnv(soc, temp, off, memory=memory); env.reset(v0); eid = 'frz'
    camera = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=a.camera_seed * 1000 + a.cond, noise_seed=a.camera_seed * 1000 + 500 + a.cond, world='v4'); camera.new_episode(eid)
    store = FrameStore(); hist = []; cap(camera, env, store, hist, eid, per, memory); done = False
    while not done and env.t < 600 and env.x < a.x_freeze:
        if getattr(a, 'vfree', None) and not capped and env.x >= a.cap_from: set_vfree(a.vfree); capped = True
        u = act(learner, memory, env, store, hist, eid); cmd = S.command(u)
        for _ in range(4):
            _, r, d, info = env.step(cmd); cap(camera, env, store, hist, eid, per, memory)
            if info['red_crossing'] or info['arrived'] or env.x >= a.x_freeze: done = True; break
    assert env.x >= a.x_freeze and env.x < XS - 200., f'冻结位置 {env.x:.1f} 不在灯可见区之前'
    assert not memory.sig['seen'], '冻结时记忆里不应有灯轨迹'
    bundle = dict(env=env, camera=camera, frames={fid: store.frames[fid] for fid in hist}, hist=list(hist), eid=eid, t=float(env.t), x=float(env.x), v=float(env.v), a=float(env.a), soc=float(env.soc), T=float(env.T),
                  policy=a.policy, cond=a.cond, camera_seed=a.camera_seed, x_freeze=a.x_freeze, identity=ck.get('identity'), memory_snapshot=memory.snapshot(), frozen_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    bundle['vfree'] = getattr(a, 'vfree', None); name = getattr(a, 'out', None) or 'frozen_state'
    OUT.mkdir(parents=True, exist_ok=True); pickle.dump(bundle, open(OUT / f'{name}.pkl', 'wb'))
    meta = {k: v for k, v in bundle.items() if k not in ('env', 'camera', 'frames', 'hist')}; json.dump(meta, open(OUT / f'{name}.json', 'w'), indent=1, ensure_ascii=False)
    print('冻结状态', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items() if k in ('t', 'x', 'v', 'a', 'soc', 'T', 'policy', 'cond')})


# ------------------------------------------------------------------ 延迟位置选择（物理余量）
def select(a):
    b = pickle.load(open(OUT / 'frozen_state.pkl', 'rb')); v, acc = b['v'], b['a']
    D = {j: stop_distance(v, acc, j) for j in (2., 4.)}
    tau = a.latency_s; lat = v * tau
    # 中等延迟：采纳时刻 M_2 ≈ 执行层余量（jerk 2 的备份恰能停在执行层目标点）→ 灯出现时 d_line = D_2 + (0.15 d + 8) + v·τ
    # 解 d_m：d_m − v·τ = D_2 + 0.15·(d_m − v·τ) + 8  →  d_adopt = (D_2 + 8)/0.85；d_m = d_adopt + v·τ
    d_adopt_m = (D[2.] + 8.) / 0.85; d_m = d_adopt_m + lat
    # 较晚延迟：采纳时刻 M_2 < 0 而 M_4 ≥ 0 之间取中点（jerk 2 备份已不能停在线上，jerk 4 备份仍能）；若 D_2 − D_4 过小则无可分辨窗口
    d_adopt_l = (D[2.] + D[4.]) / 2.; d_l = d_adopt_l + lat
    out = dict(v_freeze=v, a_freeze=acc, D_stop={str(k): x for k, x in D.items()}, latency_s_assumed=tau, latency_m=lat,
               medium=dict(d_reveal=d_m, x_reveal=XS - d_m, d_adopt_expected=d_adopt_m, M2_at_adopt=d_adopt_m - D[2.], M4_at_adopt=d_adopt_m - D[4.]),
               late=dict(d_reveal=d_l, x_reveal=XS - d_l, d_adopt_expected=d_adopt_l, M2_at_adopt=d_adopt_l - D[2.], M4_at_adopt=d_adopt_l - D[4.]),
               window_m=D[2.] - D[4.], note='D_stop 为 curve_layer.backup_distance（当前离散动力学下该备份制动的停车距离）；采纳延迟 τ 为烟测实测前的假定，烟测后可按实测更新一次并冻结')
    json.dump(out, open(OUT / 'delays.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: (round(v_, 2) if isinstance(v_, float) else v_) for k, v_ in out.items() if k not in ('medium', 'late')}, ensure_ascii=False)); print('中等', {k: round(v_, 1) for k, v_ in out['medium'].items()}); print('较晚', {k: round(v_, 1) for k, v_ in out['late'].items()})
    if out['window_m'] < 5.: print('警告：D_2 − D_4 窗口 < 5 m，两种 jerk 上限在当前几何下可能没有可分辨的准备窗口')


# ------------------------------------------------------------------ 分支
def branch_id(policy, situation, jerk, visual, x_reveal, bundle_hash):
    key = json.dumps(dict(policy=str(policy), situation=situation, jerk=jerk, visual=visual, x_reveal=None if x_reveal is None else round(x_reveal, 2), bundle=bundle_hash), sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]

V_HI = 10.   # 高速段阈值：v ≥ 10 m/s 的接近子步（制动决策段）；终末低速修正（v < 10、d 小）是执行层停车目标收敛的固有现象，单列


def approach_metrics(recs, t0, jerk, v_hi=V_HI):
    """接近段 = 从分支起点到首次停稳（线前 60 m 内）或过线为止。"""
    i_end = len(recs)
    for i_, r_ in enumerate(recs):
        if (r_['v'] < 0.05 and XS - 60. < r_['x'] < XS + 5.) or r_['x'] >= XS: i_end = i_ + 1; break
    appr = recs[:i_end]; ov = lambda r_: bool(r_['fallback'] or r_['jerk_override']); first_fb = next((r_ for r_ in appr if ov(r_)), None)
    hs = [r_ for r_ in appr if ov(r_) and r_['v'] >= v_hi]
    return dict(substeps=len(appr), Ij=float(sum(r_['jerk'] ** 2 for r_ in appr) * E.DT), jerk_max=float(max([abs(r_['jerk']) for r_ in appr] or [0.])), jerk_override_substeps=int(sum(r_['jerk_override'] for r_ in appr)),
                fallback_substeps=int(sum(r_['fallback'] for r_ in appr)), safe_override_substeps=int(sum(r_['safe_override'] for r_ in appr)), end_x=float(appr[-1]['x']) if appr else None, end_v=float(appr[-1]['v']) if appr else None,
                smooth=bool(appr and first_fb is None), first_fallback=None if first_fb is None else dict(t=first_fb['t'] - t0, x=first_fb['x'], d_line=XS - first_fb['x'], v=first_fb['v'], a=first_fb['a']),
                smooth_highspeed=bool(appr and not hs), highspeed_override_substeps=len(hs), terminal_override_substeps=int(sum(1 for r_ in appr if ov(r_) and r_['v'] < v_hi)), v_hi=v_hi,
                first_highspeed_fallback=None if not hs else dict(t=hs[0]['t'] - t0, d_line=XS - hs[0]['x'], v=hs[0]['v']))


def run_branch(bundle, learner, per, situation, offset, jerk, visual, x_reveal, budget_s, thr, hide_key='traffic_light', prep_mode='conservative'):
    jp = set_jerk_limit(jerk)
    if bundle.get('vfree'): set_vfree(bundle['vfree'])
    b = copy.deepcopy(bundle); env = b['env']; memory = env.memory; camera = b['camera']; store = FrameStore(); store.frames.update(b['frames']); hist = list(b['hist']); eid = b['eid']
    env.prep_mode = prep_mode   # v6：'conservative' / 'minimal'
    env.offsets[0] = float(offset)   # 外部信号时序：冻结状态在灯可见区之前，与相位无关
    t0, x0, e0, jsq0, n_tr0, n_ll0 = env.t, env.x, env.e_batt, env.jerk_sq, len(env.trace), len(env.layer_log)
    color_mask = hide_key == 'color_mask'; memory.colorless_track = color_mask
    revealed = (x_reveal is None) or env.x >= x_reveal; camera.hide = set() if (revealed or color_mask) else {hide_key}
    recs = []; decisions = []; t_cue = env.t if revealed else None; done = False; violation = 0
    while not done and env.t - t0 < budget_s and env.x < X_END_BRANCH:
        u = act(learner, memory, env, store, hist, eid); cmd = S.command(u); n_before = len(recs); st0 = dict(t=env.t, x=env.x, v=env.v, a=env.a, soc=env.soc, e_batt=env.e_batt)
        decisions.append(dict(u=float(u), cmd=float(cmd), state0=st0))
        for _ in range(4):
            _, r, d, info = env.step(cmd)
            if not revealed and env.x >= x_reveal: revealed = True; camera.hide = set(); t_cue = env.t   # 从此帧起两条图像路径都看到灯（color_mask：从此帧起记忆收到灯色）
            dets, probs = cap(camera, env, store, hist, eid, per, memory, mask_color=(color_mask and not revealed))
            L = env.layer_log[-1]; tr = env.trace[-1]; truth = signal_color(env.t, env.offsets[0]); lt = dets['traffic_light']
            det_color = None if probs is None else ['red', 'yellow', 'green', 'off', 'unknown'][int(np.argmax(probs))]
            light_targets = [t_ for t_ in memory.executor_targets(env.x) if t_[0] < XS + 30. and t_[1] == 0.0]
            recs.append(dict(t=float(env.t), x=float(env.x), v=float(env.v), a=float(env.a), cmd=float(cmd), applied=float(L['applied']), fallback=bool(L['fallback']), intervened=bool(L['intervened']),
                             truth=truth, hidden=not revealed, det_score=float(lt['score']), det_color=det_color, mem_seen=bool(memory.sig['seen']), mem_phase=memory.sig['phase'], mem_d_line=memory.sig['d_line'],
                             light_target=bool(light_targets), jerk=float(tr[S.TRACE_COLS.index('jerk')]), jerk_override=int(tr[S.TRACE_COLS.index('jerk_override')]), safe_override=int(tr[S.TRACE_COLS.index('safe_override')]),
                             red_crossing=int(info['red_crossing']), reward=float(r), decision=len(decisions)))
            if info['red_crossing']: violation += 1; done = True; break
            if env.x >= X_END_BRANCH: done = True; break
        dec = decisions[-1]; sub = recs[n_before:]
        dec.update(applied=[r_['applied'] for r_ in sub], jerk=[r_['jerk'] for r_ in sub], reward=[r_['reward'] for r_ in sub], reward_sum=float(sum(r_['reward'] for r_ in sub)),
                   state1=dict(t=env.t, x=env.x, v=env.v, a=env.a, soc=env.soc, e_batt=env.e_batt), dv=float(env.v - st0['v']), dx=float(env.x - st0['x']), dE_Wh=float((env.e_batt - st0['e_batt']) / 3600.),
                   fallback=[r_['fallback'] for r_ in sub], truth=[r_['truth'] for r_ in sub], mem_phase=[r_['mem_phase'] for r_ in sub])
    # 四个时刻与余量
    def first(pred):
        for r_ in recs:
            if pred(r_): return r_
        return None
    r_det = first(lambda r_: (not r_['hidden']) and r_['det_score'] >= thr and r_['det_color'] == r_['truth'])
    r_adopt = first(lambda r_: r_['mem_seen'] and r_['mem_phase'] == r_['truth'] and r_['mem_phase'] != 'unknown')
    r_target = first(lambda r_: r_['light_target'])
    r_act = first(lambda r_: r_adopt is not None and r_['t'] >= r_adopt['t'] and r_['intervened'] and r_['light_target'])
    def margins(r_):
        if r_ is None: return None
        d = XS - r_['x']; return dict(t=r_['t'], x=r_['x'], v=r_['v'], a=r_['a'], d_line=d, truth=r_['truth'], M2=d - stop_distance(r_['v'], r_['a'], 2.), M4=d - stop_distance(r_['v'], r_['a'], 4.), M_branch=d - stop_distance(r_['v'], r_['a'], jerk))
    jerks = [r_['jerk'] for r_ in recs]; dtb = env.t - t0
    stopped = any(r_['v'] < 0.05 and XS - 60. < r_['x'] < XS + 5. for r_ in recs)
    approach = approach_metrics(recs, t0, jerk)
    _unused = None
    i_end = len(recs)
    jerks = [r_['jerk'] for r_ in recs]; dtb = env.t - t0
    stopped = any(r_['v'] < 0.05 and XS - 60. < r_['x'] < XS + 5. for r_ in recs)
    approach = approach_metrics(recs, t0, jerk)
    _unused = None
    i_end = len(recs)
    for i_, r_ in enumerate(recs):
        if (r_['v'] < 0.05 and XS - 60. < r_['x'] < XS + 5.) or r_['x'] >= XS: i_end = i_ + 1; break
    appr = recs[:i_end]; first_fb = next((r_ for r_ in appr if r_['fallback'] or r_['jerk_override']), None)
    approach = dict(substeps=len(appr), Ij=float(sum(r_['jerk'] ** 2 for r_ in appr) * E.DT), jerk_max=float(max([abs(r_['jerk']) for r_ in appr] or [0.])), jerk_override_substeps=int(sum(r_['jerk_override'] for r_ in appr)),
                    fallback_substeps=int(sum(r_['fallback'] for r_ in appr)), safe_override_substeps=int(sum(r_['safe_override'] for r_ in appr)), end_x=float(appr[-1]['x']) if appr else None, end_v=float(appr[-1]['v']) if appr else None,
                    smooth=bool(appr and sum(r_['jerk_override'] for r_ in appr) == 0 and sum(r_['fallback'] for r_ in appr) == 0), first_fallback=None if first_fb is None else dict(t=first_fb['t'] - t0, x=first_fb['x'], d_line=XS - first_fb['x'], v=first_fb['v'], a=first_fb['a']),
                    # 高速段平顺：v ≥ 5 m/s 的接近子步内无应急回退/jerk 越界（终末低速修正 d<15 m、v<5 m/s 是执行层停车目标收敛的固有现象，单列）
                    smooth_highspeed=bool(appr and not any((r_['fallback'] or r_['jerk_override']) and r_['v'] >= 5. for r_ in appr)),
                    highspeed_override_substeps=int(sum(1 for r_ in appr if (r_['fallback'] or r_['jerk_override']) and r_['v'] >= 5.)), terminal_override_substeps=int(sum(1 for r_ in appr if (r_['fallback'] or r_['jerk_override']) and r_['v'] < 5.)))
    pre = [r_ for r_ in recs if r_['hidden']]; prep_max_decel = float(max([-r_['applied'] for r_ in pre] or [0.])); v_at_cue = float(pre[-1]['v']) if pre else None
    res = dict(situation=situation, offset=offset, jerk=jerk, jerk_params=jp, visual=visual, x_reveal=x_reveal, policy=str(learner_path_of(learner)), hide_key=hide_key, prep_mode=prep_mode, vfree=bundle.get('vfree'),
               preparation=dict(max_decel_before_cue=prep_max_decel, v_at_cue=v_at_cue, hidden_substeps=len(pre)),
               outcome=dict(reached_end=bool(env.x >= X_END_BRANCH), violation=violation, stopped_before_line=stopped, timeout=bool(env.t - t0 >= budget_s and env.x < X_END_BRANCH),
                            time_s=dtb, E_Wh=(env.e_batt - e0) / 3600., Ij=float((env.jerk_sq - jsq0) * E.DT), jerk_max=float(max(abs(j_) for j_ in jerks)) if jerks else 0.,
                            jerk_override_substeps=int(sum(r_['jerk_override'] for r_ in recs)), safe_override_substeps=int(sum(r_['safe_override'] for r_ in recs)), fallback_substeps=int(sum(r_['fallback'] for r_ in recs)),
                            min_v_near_line=float(min([r_['v'] for r_ in recs if XS - 60. < r_['x'] < XS + 5.] or [np.nan])), x_end=float(env.x), substeps=len(recs)),
               approach=approach,
               timeline=dict(t0=t0, x0=x0, cue=margins(first(lambda r_: (not r_['hidden']) and XS - r_['x'] < 200.)), detect=margins(r_det), adopt=margins(r_adopt), target=margins(r_target), act=margins(r_act)),   # 线索开放 = 灯首次可被渲染（未隐藏且进入 200 m 可见区）
               truth_at_cue=(signal_color(t_cue, offset) if t_cue is not None else None), decisions=decisions, recs=recs)
    return res


def learner_path_of(learner): return getattr(learner, '_path', '?')


def run(a):
    cfg = load_cfg(a.config); mk = {k: cfg[k] for k in ('det_thr', 'hold_s')}; per = make_perception(cfg); thr = cfg['det_thr']['traffic_light']
    bundle = pickle.load(open(OUT / 'frozen_state.pkl', 'rb')); bh = hashlib.sha256(open(OUT / 'frozen_state.pkl', 'rb').read()).hexdigest()[:12]
    delays = json.load(open(OUT / 'delays.json')); xr = dict(normal=None, medium=delays['medium']['x_reveal'], late=delays['late']['x_reveal'])
    t_f = bundle['t']; v_f = bundle['v']; t_arr = t_f + (XS - bundle['x']) / max(v_f, 1.)   # 名义到达时刻
    # 相位偏移：需停车 = 名义到达时处于红灯中段（phase 60：还剩 30 s 红）；允许通行 = 名义到达时绿灯早段（phase 5：还剩 25 s 绿）
    offsets = dict(stop=float((60. - t_arr) % 90.), pass_=float((5. - t_arr) % 90.))
    (OUT / a.tag).mkdir(parents=True, exist_ok=True); done_n = 0; t_start = time.time()
    for pol in a.policies.split(','):
        learner, ck = learner_from_final_nets(ROOT / pol); learner._path = pol
        for sit in a.situations.split(','):
            off = offsets['pass_' if sit == 'pass' else sit]
            for jk in [float(x) for x in a.jerks.split(',')]:
                for vis in a.visuals.split(','):
                    bid = branch_id(pol, sit, jk, vis, xr[vis], bh); fp = OUT / a.tag / f'{bid}.json'
                    if fp.exists(): print('跳过已完成', bid, pol, sit, jk, vis); continue
                    t1 = time.time(); res = run_branch(bundle, learner, per, sit, off, jk, vis, xr[vis], a.budget_s, thr)
                    res.update(branch_id=bid, bundle_hash=bh, policy_identity=ck.get('identity'), wall_s=time.time() - t1, tag=a.tag)
                    json.dump(res, open(fp, 'w'), indent=1, ensure_ascii=False); done_n += 1; o = res['outcome']; tl = res['timeline']
                    fmt = lambda m: '–' if m is None else f"t{m['t'] - t_f:.1f}s d{m['d_line']:.0f} v{m['v']:.1f} M2{m['M2']:+.0f} M4{m['M4']:+.0f}"
                    ap_ = res['approach']; fb_ = ap_['first_fallback']; fbs_ = '–' if fb_ is None else 't%.1f d%.0f v%.1f' % (fb_['t'], fb_['d_line'], fb_['v'])
                    print(f"[{pol.split('/')[1]}|{sit}|jerk{jk:.0f}|{vis}] 到终 {o['reached_end']} 违规 {o['violation']} 线前停 {o['stopped_before_line']} 时间 {o['time_s']:.1f} 能耗 {o['E_Wh']:.1f} I_j {o['Ij']:.1f} | 接近段 平顺 {ap_['smooth']} I_j {ap_['Ij']:.1f} jerk_max {ap_['jerk_max']:.1f} 越界 {ap_['jerk_override_substeps']} 回退 {ap_['fallback_substeps']} 首次回退 {fbs_} | 出现 {fmt(tl['cue'])} | 检出 {fmt(tl['detect'])} | 采纳 {fmt(tl['adopt'])} | 动作 {fmt(tl['act'])} ({res['wall_s']:.0f}s)", flush=True)
    print(f'完成 {done_n} 条分支，{time.time() - t_start:.0f}s')


def sweep(a):
    """事后探索（标注为 post-hoc，不替代冻结的正式延迟）：按灯出现距离扫描，描出“高速段是否保持平顺”的边界随 jerk 上限的移动。"""
    cfg = load_cfg(a.config); per = make_perception(cfg); thr = cfg['det_thr']['traffic_light']
    bundle = pickle.load(open(OUT / 'frozen_state.pkl', 'rb')); bh = hashlib.sha256(open(OUT / 'frozen_state.pkl', 'rb').read()).hexdigest()[:12]
    t_arr = bundle['t'] + (XS - bundle['x']) / max(bundle['v'], 1.); off = float((60. - t_arr) % 90.)
    (OUT / a.tag).mkdir(parents=True, exist_ok=True); ds = [float(x) for x in a.d_reveals.split(',')]
    for pol in a.policies.split(','):
        learner, ck = learner_from_final_nets(ROOT / pol); learner._path = pol
        for jk in [float(x) for x in a.jerks.split(',')]:
            for d_ in ds:
                vis = f'd{d_:.0f}'; bid = branch_id(pol, 'stop', jk, vis, XS - d_, bh); fp = OUT / a.tag / f'{bid}.json'
                if fp.exists(): continue
                res = run_branch(bundle, learner, per, 'stop', off, jk, vis, XS - d_, a.budget_s, thr); res.update(branch_id=bid, bundle_hash=bh, tag=a.tag, d_reveal=d_)
                json.dump(res, open(fp, 'w'), indent=1, ensure_ascii=False)
    rows = [json.load(open(f)) for f in sorted((OUT / a.tag).glob('*.json'))]
    lines = ['| 策略 | jerk | 灯出现 d m | 采纳 d m | M2 / M4 采纳时 | 高速段平顺 | 高速段回退子步 | 首次高速回退 d m, v | 接近段 I_j | 违规 |', '|' + '---|' * 10]
    for r in sorted(rows, key=lambda r_: (r_['policy'], r_['jerk'], -r_['d_reveal'])):
        ap_ = approach_metrics(r['recs'], r['timeline']['t0'], r['jerk']); ad = r['timeline']['adopt']; fb = ap_['first_highspeed_fallback']
        ad_s = '–' if ad is None else '%.0f' % ad['d_line']; m_s = '–' if ad is None else '%+.0f / %+.0f' % (ad['M2'], ad['M4']); fb_s = '–' if fb is None else '%.0f, %.1f' % (fb['d_line'], fb['v'])
        lines.append(f"| {r['policy'].split('/')[1]} | {r['jerk']:.0f} | {r['d_reveal']:.0f} | {ad_s} | {m_s} | {'是' if ap_['smooth_highspeed'] else '否'} | {ap_['highspeed_override_substeps']} | {fb_s} | {ap_['Ij']:.1f} | {r['outcome']['violation']} |")
    md = '\n'.join(lines); open(OUT / a.tag / 'sweep.md', 'w').write(md); print(md)


def run6(a):
    """v6 闭环：信息条件 {early, cons(延迟-保守), min(延迟-最小准备)} × 情境 {stop, pass} × jerk × 采纳距离列表 × 策略；灯色隐藏（灯箱可见）到指定位置。"""
    cfg = load_cfg(a.config); mk = {k: cfg[k] for k in ('det_thr', 'hold_s')}; per = make_perception(cfg, a.encoder); thr = cfg['det_thr']['traffic_light']
    bundle = pickle.load(open(OUT / f'{a.frozen}.pkl', 'rb')); bh = hashlib.sha256(open(OUT / f'{a.frozen}.pkl', 'rb').read()).hexdigest()[:12]
    if bundle.get('vfree'): set_vfree(bundle['vfree'])
    t_f, v_f = bundle['t'], bundle['v']; t_arr = t_f + (XS - bundle['x']) / max(v_f, 1.)
    # v6 情境相位（与定理的 R/G 情境对应）：需停车 = 冻结时刻相位 40（整个接近段为红，红灯持续到冻结后 50 s）；允许通行 = 冻结时刻相位 0（绿灯刚开始，
    # 30 s 绿覆盖整个接近段与过线）。v5 的相位按名义到达时刻设定，通行情境在接近段前半仍是红灯，早知观察者会先按红灯制动——那是 v5 通行分支早知更慢的原因之一。
    offsets = dict(stop=float((40. - t_f) % 90.), pass_=float((0. - t_f) % 90.))
    d_adopts = [float(x) for x in a.d_adopts.split(',')]; lat_m = v_f * a.latency_s
    (OUT / a.tag).mkdir(parents=True, exist_ok=True); n = 0; t_start = time.time()
    for pol in a.policies.split(','):
        learner, ck = load_policy(pol)
        for sit in a.situations.split(','):
            off = offsets['pass_' if sit == 'pass' else sit]
            for jk in [float(x) for x in a.jerks.split(',')]:
                for info in a.info.split(','):
                    conds = [('early', None, 'conservative')] if info == 'early' else [(f'{info}_d{d_:.0f}', XS - (d_ + lat_m), 'minimal' if info == 'min' else 'conservative') for d_ in d_adopts]
                    for vis, xr, pm in conds:
                        bid = branch_id(pol, sit, jk, vis, xr, bh + a.hide_key + (a.encoder or '')); fp = OUT / a.tag / f'{bid}.json'
                        if fp.exists(): continue
                        t1 = time.time(); res = run_branch(bundle, learner, per, sit, off, jk, vis, xr, a.budget_s, thr, hide_key=a.hide_key, prep_mode=pm)
                        res.update(branch_id=bid, bundle_hash=bh, frozen=a.frozen, info=info, d_adopt_target=(None if xr is None else XS - xr - lat_m), policy_identity=ck.get('identity'), wall_s=time.time() - t1, tag=a.tag, encoder=a.encoder)
                        json.dump(res, open(fp, 'w'), indent=1, ensure_ascii=False); n += 1; o = res['outcome']; tl = res['timeline']; ap_ = approach_metrics(res['recs'], tl['t0'], jk)
                        ad = tl['adopt']; ad_s = '–' if ad is None else 'd%.0f v%.1f M%+.0f' % (ad['d_line'], ad['v'], ad['M_branch']); pname = pol.split('/')[1] if '/' in pol else pol
                        print(f"[{pname}|{sit}|j{jk:.0f}|{vis}|{pm[:4]}] 到终 {o['reached_end']} 违规 {o['violation']} 线前停 {o['stopped_before_line']} 时间 {o['time_s']:.1f} E {o['E_Wh']:.0f} | 准备最大减速 {res['preparation']['max_decel_before_cue']:.2f} v@cue {res['preparation']['v_at_cue']} | 高速平顺 {ap_['smooth_highspeed']} 回退 {ap_['highspeed_override_substeps']}/{ap_['terminal_override_substeps']} | 采纳 {ad_s} ({res['wall_s']:.0f}s)", flush=True)
    print(f'完成 {n} 条，{time.time() - t_start:.0f}s')


def summarize6(a):
    """v6 闭环汇总：按策略×jerk×冻结状态，(i) 停车支：各信息条件/采纳距离的高速段平顺与首次回退；测得的平顺边界 vs LP 可行边界；
    (ii) 通行支：延迟条件相对早知的时间差（准备代价）vs LP 通行支惩罚；(iii) p=0.5 期望时间惩罚 vs LP。"""
    rows = [json.load(open(f)) for f in sorted((OUT / a.tag).glob('*.json'))]
    lp = {}
    for name, f in (('22', 'theory_lp.json'), ('16', 'theory_lp_v16.json')):
        if (OUT / f).exists(): lp[name] = json.load(open(OUT / f))
    def lp_pred(v0, j, d_res):
        """LP：分辨距离 d_res（Δ = (d0 − d_res)/v0）下的可行性与 G 支惩罚（线性插值到 0.5 s 网格）。"""
        L = lp.get('22' if v0 > 20 else '16');
        if L is None: return None
        D = (L['d0'] - d_res) / L['v0']; cands = [r for r in L['rows'] if r['jerk'] == j]
        best = min(cands, key=lambda r: abs(r['delta'] - D)) if cands else None
        return None if best is None else dict(delta=best['delta'], feasible=best.get('delayed') == 'ok', penalty_s=best.get('penalty_s'), G_pen_s=None if best.get('delayed') != 'ok' else (best['lossG'] - 0.) / L['v0'], prep=best.get('prep_max_decel'))
    out = []
    keys = sorted({(r['policy'], r['jerk'], r.get('frozen', 'frozen_state')) for r in rows}, key=str)
    for pol, jk, fz in keys:
        sel = [r for r in rows if r['policy'] == pol and r['jerk'] == jk and r.get('frozen', 'frozen_state') == fz]
        v0 = sel[0]['recs'][0]['v'] if sel else 22.2
        early = {r['situation']: r for r in sel if r['info'] == 'early'}
        lines = [f"### {pol.split('/')[1] if '/' in pol else pol} | jerk {jk:.0f} | 冻结 {fz}（v0≈{v0:.1f}）", '',
                 '| 信息条件 | 目标采纳 d | 实际采纳 d（停/通） | 停车支 高速平顺 / 首次高速回退 d,v | 停车支 准备最大减速 | 通行支 Δ时间 vs 早知 s | 通行支 ΔE Wh | 通行支 准备最大减速 | LP：可行 / G 支惩罚 s / 最优准备 |', '|---|---|---|---|---|---|---|---|---|']
        for info in ('cons', 'min'):
            ds = sorted({r['d_adopt_target'] for r in sel if r['info'] == info and r['d_adopt_target'] is not None}, reverse=True)
            for d_ in ds:
                rs_ = [r for r in sel if r['info'] == info and r['d_adopt_target'] == d_]; st = next((r for r in rs_ if r['situation'] == 'stop'), None); ps = next((r for r in rs_ if r['situation'] == 'pass'), None)
                ap_ = approach_metrics(st['recs'], st['timeline']['t0'], jk) if st else None; fb = ap_['first_highspeed_fallback'] if ap_ else None
                ad_s = ' / '.join('–' if (r is None or r['timeline']['adopt'] is None) else f"{r['timeline']['adopt']['d_line']:.0f}" for r in (st, ps))
                dt = None if (ps is None or 'pass' not in early) else ps['outcome']['time_s'] - early['pass']['outcome']['time_s']; dE = None if (ps is None or 'pass' not in early) else ps['outcome']['E_Wh'] - early['pass']['outcome']['E_Wh']
                pred = lp_pred(v0, jk, d_)
                fb_s = '–' if fb is None else '%.0f, %.1f' % (fb['d_line'], fb['v']); sm_s = '–' if ap_ is None else ('是' if ap_['smooth_highspeed'] else '否')
                prep_st = '–' if st is None else '%.2f' % st['preparation']['max_decel_before_cue']; prep_ps = '–' if ps is None else '%.2f' % ps['preparation']['max_decel_before_cue']
                dt_s = '–' if dt is None else '%+.1f' % dt; dE_s = '–' if dE is None else '%+.1f' % dE
                if pred is None: lp_s = '–'
                else: lp_s = ('可行' if pred['feasible'] else '不可行') + ' / ' + ('–' if pred['G_pen_s'] is None else '%+.2f' % pred['G_pen_s']) + ' / ' + ('–' if pred['prep'] is None else '%.2f' % pred['prep'])
                lines.append(f"| {info} | {d_:.0f} | {ad_s} | {sm_s} / {fb_s} | {prep_st} | {dt_s} | {dE_s} | {prep_ps} | {lp_s} |")
        if 'stop' in early:
            ap_ = approach_metrics(early['stop']['recs'], early['stop']['timeline']['t0'], jk); lines.append(f"| early | – | {early['stop']['timeline']['adopt']['d_line']:.0f} / {early['pass']['timeline']['adopt']['d_line']:.0f} | {'是' if ap_['smooth_highspeed'] else '否'} | – | +0.0 (基准 {early['pass']['outcome']['time_s']:.1f} s) | +0.0 | – | – |")
        out.append('\n'.join(lines))
    # 闭环边界汇总：按策略×冻结×jerk：测得的“高速段平顺”最小采纳距离（min / cons）、通行支首次出现时间惩罚的采纳距离，对照 LP 的共同准备可行边界与无额外代价边界
    bl = ['', '## 边界汇总（测量 vs LP）', '', '| 策略 | 冻结 | jerk | 停车支高速段平顺首次丢失的目标采纳 d：min / cons（None=全部平顺） | 通行支惩罚 ≥0.5 s 开始的目标采纳 d：min / cons | LP 共同准备可行的最小分辨 d | LP 无额外代价的最小分辨 d |', '|---|---|---|---|---|---|---|']
    for pol, jk, fz in keys:
        sel = [r for r in rows if r['policy'] == pol and r['jerk'] == jk and r.get('frozen', 'frozen_state') == fz]; v0 = sel[0]['recs'][0]['v'] if sel else 22.2
        early = {r['situation']: r for r in sel if r['info'] == 'early'}
        def bmin(info):
            """停车支高速段平顺首次丢失的目标采纳距离（从大到小扫描，第一个“否”）；全部平顺 → None（用 '全部' 表示）。"""
            ds = sorted({r['d_adopt_target'] for r in sel if r['info'] == info and r['d_adopt_target'] is not None}, reverse=True)
            for d_ in ds:
                st = next((r for r in sel if r['info'] == info and r['d_adopt_target'] == d_ and r['situation'] == 'stop'), None)
                if st is not None and not approach_metrics(st['recs'], st['timeline']['t0'], jk)['smooth_highspeed']: return d_
            return None
        def bpen(info):
            ds = sorted({r['d_adopt_target'] for r in sel if r['info'] == info and r['d_adopt_target'] is not None}, reverse=True)
            for d_ in ds:
                ps = next((r for r in sel if r['info'] == info and r['d_adopt_target'] == d_ and r['situation'] == 'pass'), None)
                if ps is not None and 'pass' in early and ps['outcome']['time_s'] - early['pass']['outcome']['time_s'] >= 0.5: return d_
            return None
        L = lp.get('22' if v0 > 20 else '16'); lp_feas = lp_free = None
        if L:
            rr = [r for r in L['rows'] if r['jerk'] == jk]; feas = [L['d0'] - L['v0'] * r['delta'] for r in rr if r.get('delayed') == 'ok']; free = [L['d0'] - L['v0'] * r['delta'] for r in rr if r.get('delayed') == 'ok' and r['penalty_s'] < 1e-6]
            lp_feas = min(feas) if feas else None; lp_free = min(free) if free else None
        f_ = lambda x: '全部' if x is None else '%.0f' % x
        bl.append(f"| {pol.split('/')[1] if '/' in pol else pol} | {fz} | {jk:.0f} | {f_(bmin('min'))} / {f_(bmin('cons'))} | {f_(bpen('min'))} / {f_(bpen('cons'))} | {f_(lp_feas)} | {f_(lp_free)} |")
    out.append('\n'.join(bl))
    md = '\n\n'.join(out); open(OUT / a.tag / 'closure.md', 'w').write(md); print(md)


def summarize(a):
    rows = [json.load(open(f)) for f in sorted((OUT / a.tag).glob('*.json'))]
    print(f'分支数 {len(rows)}')
    hdr = '| 策略 | 情境 | jerk | 视觉 | 到终点 | 违规 | 线前停 | 时间 s | 能耗 Wh | I_j 全程 | 高速段平顺 | 高速/终末回退子步 | 接近段 I_j | 接近段 jerk峰值 | 首次回退 (t s, d m, v) | 出现→检出→采纳→动作 (s) | M2/M4 采纳时 |'
    lines = [hdr, '|' + '---|' * 17]
    for r in sorted(rows, key=lambda r_: (r_['policy'], r_['situation'], r_['jerk'], ['normal', 'medium', 'late'].index(r_['visual']))):
        o = r['outcome']; tl = r['timeline']; t0 = tl['t0']; r['approach'] = approach_metrics(r['recs'], t0, r['jerk'])
        ts = ' → '.join('–' if tl[k] is None else f"{tl[k]['t'] - t0:.1f}" for k in ('cue', 'detect', 'adopt', 'act'))
        mm = '–' if tl['adopt'] is None else f"{tl['adopt']['M2']:+.0f} / {tl['adopt']['M4']:+.0f}"
        ap_ = r['approach']; fb = ap_['first_fallback']; fbs = '–' if fb is None else f"{fb['t']:.1f}, {fb['d_line']:.0f}, {fb['v']:.1f}"
        lines.append(f"| {r['policy'].split('/')[1]} | {r['situation']} | {r['jerk']:.0f} | {r['visual']} | {o['reached_end']} | {o['violation']} | {o['stopped_before_line']} | {o['time_s']:.1f} | {o['E_Wh']:.1f} | {o['Ij']:.1f} | {'是' if ap_['smooth_highspeed'] else '否'} | {ap_['highspeed_override_substeps']}/{ap_['terminal_override_substeps']} | {ap_['Ij']:.1f} | {ap_['jerk_max']:.1f} | {fbs} | {ts} | {mm} |")
    md = '\n'.join(lines); open(OUT / a.tag / 'summary.md', 'w').write(md); print(md)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--policy', default='runs/v4r/v4_pilot/supervised/final_nets.pt'); f.add_argument('--config', default='v4r_cpu.json'); f.add_argument('--cond', type=int, default=0); f.add_argument('--camera-seed', type=int, default=1000); f.add_argument('--x-freeze', type=float, default=2740.); f.add_argument('--vfree', type=float, default=None); f.add_argument('--cap-from', type=float, default=2400.); f.add_argument('--out', default=None)
    r6 = sub.add_parser('run6'); r6.add_argument('--policies', default='constant,runs/v4r/v4_pilot/supervised/final_nets.pt,runs/v4r_s1/v4_pilot/supervised/final_nets.pt,runs/v4r_s2/v4_pilot/supervised/final_nets.pt'); r6.add_argument('--situations', default='stop,pass'); r6.add_argument('--jerks', default='2,3,4'); r6.add_argument('--info', default='early,cons,min'); r6.add_argument('--d-adopts', default='150,120,100,89,78,67,56,44'); r6.add_argument('--latency-s', type=float, default=0.5); r6.add_argument('--hide-key', default='light_color'); r6.add_argument('--encoder', default='runs/pretrain_v4_off/encoder.pt'); r6.add_argument('--frozen', default='frozen_state'); r6.add_argument('--budget-s', type=float, default=150.); r6.add_argument('--config', default='v4r_cpu.json'); r6.add_argument('--tag', default='v6_smoke')
    s = sub.add_parser('select'); s.add_argument('--latency-s', type=float, default=1.5)
    r = sub.add_parser('run'); r.add_argument('--policies', default='runs/v4r/v4_pilot/supervised/final_nets.pt,runs/v4r_s1/v4_pilot/supervised/final_nets.pt,runs/v4r_s2/v4_pilot/supervised/final_nets.pt'); r.add_argument('--situations', default='stop,pass'); r.add_argument('--jerks', default='2,4'); r.add_argument('--visuals', default='normal,medium,late'); r.add_argument('--budget-s', type=float, default=150.); r.add_argument('--config', default='v4r_cpu.json'); r.add_argument('--tag', default='smoke')
    m = sub.add_parser('summarize'); m.add_argument('--tag', default='smoke')
    m6 = sub.add_parser('summarize6'); m6.add_argument('--tag', default='v6_smoke')
    w = sub.add_parser('sweep'); w.add_argument('--policies', default='runs/v4r/v4_pilot/supervised/final_nets.pt,runs/v4r_s1/v4_pilot/supervised/final_nets.pt,runs/v4r_s2/v4_pilot/supervised/final_nets.pt'); w.add_argument('--jerks', default='2,4'); w.add_argument('--d-reveals', default='140,130,125,120,115,110,105,100,95,90,85,80'); w.add_argument('--budget-s', type=float, default=150.); w.add_argument('--config', default='v4r_cpu.json'); w.add_argument('--tag', default='sweep_posthoc')
    a = ap.parse_args(); dict(freeze=freeze, select=select, run=run, summarize=summarize, sweep=sweep, run6=run6, summarize6=summarize6)[a.cmd](a)


if __name__ == '__main__':
    main()
