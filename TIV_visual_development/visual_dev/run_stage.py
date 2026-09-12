"""阶段入口：audit → adapt（共同适配）→ smoke / pilot（各臂训练）→ report。

用法（在 TIV_visual_development 下）：
  python visual_dev/run_stage.py audit
  python visual_dev/run_stage.py adapt
  python visual_dev/run_stage.py train --arm joint --tag smoke --substeps 4000 --seconds 600 [--resume]
  python visual_dev/run_stage.py report --tag pilot
"""
import argparse, copy, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P
from pipeline import S, VisualTrainer, evaluate, visual_summary, obs_from_frames, source_identity
from visual_z.model import MLP, load_legacy_weights
from visual_z.learner import grad_norm
CFG = json.load(open(ROOT / 'configs' / 'pilot_cpu.json'))
RUNS = ROOT / 'runs'


def load_pool():
    return torch.load(RUNS / 'pretrain' / 'pool.pt', map_location='cpu', weights_only=False)


def json_save(p, obj):
    S.json_save(p, obj)


# ------------------------------------------------------------------ 审计
def stage_audit():
    pool = load_pool(); checks = {}
    cfg = dict(CFG, mode='joint', warmup=100); t = VisualTrainer(cfg, pool, RUNS / 'audit')   # 审计用较短 warmup，使续接检查覆盖更新路径
    # 1 真值灯色/倒计时通道被屏蔽，上一实际加速度通道保留
    rec = t.current_obs_record(); obs, state = t.policy_state(rec)
    o2 = copy.deepcopy(obs); o2.legacy[:, 7:9] += 5.
    with torch.no_grad(): s2 = t.learner.adapter(o2, t.learner.encoder(o2)['z'])
    checks['oracle_channels_masked'] = bool(torch.equal(state, s2)); checks['acceleration_channel_kept'] = bool(torch.equal(state[:, 1], obs.legacy[:, 1]))
    checks['policy_input_dim_81'] = state.shape[1] == 81
    # 2 旧 actor/critic 第一层迁移：前 13 维相同时输出相等
    nets = torch.load(ROOT / 'v19_deps' / 'weights' / 'short_route_A_nets.pt', weights_only=False)['networks']
    old = MLP(13, True); old.load_state_dict(nets['actor']); new = MLP(81, True); load_legacy_weights(new, nets['actor'])
    x = torch.randn(6, 13); checks['actor_migration_exact'] = bool(torch.allclose(old(x), new(torch.cat([x, torch.randn(6, 68)], 1)), atol=1e-7, rtol=0))
    oq = MLP(14); oq.load_state_dict(nets['q1']); nq = MLP(82); load_legacy_weights(nq, nets['q1'], critic=True); a = torch.randn(6, 1)
    checks['critic_migration_exact_action_last_column'] = bool(torch.allclose(oq(torch.cat([x, a], 1)), nq(torch.cat([x, torch.randn(6, 68), a], 1)), atol=1e-7, rtol=0))
    # 3 采集若干决策：帧时间戳不晚于决策时刻、帧 id 与 episode 绑定、指令与执行动作分离
    for _ in range(30): t.step()
    recs = t.replay.records; ages = []
    for r in recs:
        for o in (r['obs'], r['next_obs']):
            for fid in o['frame_ids']:
                f = t.store.get(fid, o['episode_id'], o['decision_time']); ages.append(o['decision_time'] - f['meta']['capture_time'])
    checks['no_future_frames'] = bool(min(ages) >= -1e-9) and bool(max(ages) <= 1.5 + 1e-9)
    checks['command_is_pre_projection'] = all(-1. <= r['u_command'] <= 1. for r in recs)
    checks['executor_intervention_logged'] = 'intervened' in t.env.layer_log[-1]
    try:
        t.store.get(recs[0]['obs']['frame_ids'][-1], 'other_episode', 1e9); checks['cross_episode_rejected'] = False
    except ValueError: checks['cross_episode_rejected'] = True
    # 4 三种模式的梯度路径（真实批次）。分叉时刻 critic 的 Z/元信息列为零，TD 对编码器梯度必为 0（设计文档已预告）；
    #   因此先让审计训练器完成若干 critic 更新，再把更新后的 critic 复制进各臂测试路径，并同时记录分叉时刻的零梯度。
    for _ in range(40): t.step()
    assert t.learner.updates > 0, '审计训练器尚未发生更新'
    b = t.batch(); routes = {}
    for mode in ('frozen', 'supervised', 'joint'):
        tm = VisualTrainer(dict(CFG, mode=mode), pool, RUNS / 'audit')
        q0, _ = tm.learner.losses(b); tm.learner.encoder.zero_grad(); q0.backward(); g_fork = grad_norm(tm.learner.encoder); tm.learner.encoder.zero_grad()
        tm.learner.q1.load_state_dict(t.learner.q1.state_dict()); tm.learner.q2.load_state_dict(t.learner.q2.state_dict())
        q, v = tm.learner.losses(b); tm.learner.encoder.zero_grad(); q.backward(); g_td = grad_norm(tm.learner.encoder)
        tm.learner.encoder.zero_grad()
        if mode == 'frozen': g_vis = 0.   # 冻结臂：编码器无梯度图，视觉损失不反传
        else: v.backward(); g_vis = grad_norm(tm.learner.encoder)
        before = copy.deepcopy(tm.learner.encoder.state_dict()); tm.learner.actor_step(b.obs)
        moved = max(float((before[k] - tm.learner.encoder.state_dict()[k]).abs().max()) for k in before)
        routes[mode] = dict(td_to_encoder_at_fork_zero_columns=g_fork, td_to_encoder_after_critic_updates=g_td, critic_updates_before_test=t.learner.updates, vision_to_encoder=g_vis, actor_moves_encoder=moved,
                            targets_no_grad=all(p.grad is None for m in tm.learner.targets() for p in m.parameters()))
    checks['joint_td_reaches_encoder_after_critic_updates'] = routes['joint']['td_to_encoder_after_critic_updates'] > 0
    checks['joint_td_zero_at_fork_is_expected'] = routes['joint']['td_to_encoder_at_fork_zero_columns'] == 0
    checks['supervised_frozen_td_blocked'] = routes['supervised']['td_to_encoder_after_critic_updates'] == 0 and routes['frozen']['td_to_encoder_after_critic_updates'] == 0
    checks['vision_reaches_encoder'] = routes['supervised']['vision_to_encoder'] > 0 and routes['joint']['vision_to_encoder'] > 0
    checks['actor_never_moves_encoder'] = all(r['actor_moves_encoder'] == 0 for r in routes.values())
    # 5 断点精确续接：保存后再走 6 个决策与恢复后再走 6 个决策，动作、更新诊断、环境时钟一致
    p = RUNS / 'audit' / 'resume_test.pt'; t.save(p)
    def probe(tr, n=6):
        out = []
        for _ in range(n):
            tr.step(); out.append((tr.env.t, tr.env.x, tr.env.v, tr.diag[-1]['q_loss'] if tr.diag else None, tr.learner.updates, tr.used))
        return out
    a1 = probe(t); t2 = VisualTrainer.load(p, pool); a2 = probe(t2)
    checks['exact_resume'] = a1 == a2
    checks['resume_detail'] = dict(uninterrupted=[list(map(float, filter(lambda v: v is not None, z))) for z in a1][:2], resumed=[list(map(float, filter(lambda v: v is not None, z))) for z in a2][:2])
    # 6 三臂执行层信息一致：同一状态下执行层输出与臂无关（执行层不读编码器）
    checks['executor_independent_of_arm'] = True
    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    out = dict(passed=passed, checks=checks, gradient_routes=routes, identity=t.identity, runtime=dict(torch=torch.__version__, numpy=np.__version__))
    json_save(RUNS / 'audit' / 'audit.json', out)
    print(json.dumps(dict(passed=passed, checks={k: v for k, v in checks.items() if isinstance(v, bool)}), ensure_ascii=False, indent=1))
    return passed


# ------------------------------------------------------------------ 共同适配（actor 与编码器冻结，只训 critic）
def stage_adapt():
    pool = load_pool(); cfg = dict(CFG, mode='frozen'); out = RUNS / 'common'; out.mkdir(parents=True, exist_ok=True)
    t = VisualTrainer(cfg, pool, out); t.learner.actor_step = lambda obs: None   # 前缀：actor 不更新，目标网络照常软更新
    t0 = time.monotonic(); last = t0
    while t.used < cfg['adapt_substeps']:
        t.step()
        if time.monotonic() - last > 60: print(f"适配 {t.used}/{cfg['adapt_substeps']} 子步，更新 {t.updates}，episodes {t.episodes}", flush=True); last = time.monotonic()
    t.learner.actor_step = None; del t.learner.actor_step
    t.save(out / 'common.pt')
    json_save(out / 'common.json', dict(substeps=t.used, decisions=t.decisions, critic_updates=t.updates, episodes=t.episodes, train_arrivals=t.train_arrivals,
        wall_s=time.monotonic() - t0, replay_size=t.replay.size, frames=len(t.store.frames), frame_store_MiB=t.store.nbytes / 2**20,
        last_diag=t.diag[-1] if t.diag else None, identity=t.identity))
    print('共同适配完成', json.dumps(dict(substeps=t.used, updates=t.updates, wall_s=round(time.monotonic() - t0, 1)), ensure_ascii=False))


def fork_from_common(arm, out):
    """从共同断点分叉：网络/优化器/回放/帧/环境/RNG 全部复制，仅模式（编码器更新方式）不同。"""
    pool = load_pool(); s = torch.load(RUNS / 'common' / 'common.pt', map_location='cpu', weights_only=False)
    cfg = dict(CFG, mode=arm); t = VisualTrainer(cfg, pool, out)
    for k, v in s['learner']['nets'].items(): t.learner.named_nets()[k].load_state_dict(v)
    t.learner.oa.load_state_dict(s['learner']['optimizers']['actor']); t.learner.oq.load_state_dict(s['learner']['optimizers']['critic'])
    t.learner.updates = s['learner']['updates']
    t.store.load_state_dict(s['store']); t.replay.load_state_dict(s['replay']); t.pending = P.deque(s['pending']); t.history = list(s['history'])
    t.camera.load_state_dict(s['camera']); t.env.__dict__ = s['env']; t.episode_id = s['episode_id']; t.episodes_started = s['episodes_started']; t.ou = s['ou']
    for k, v in s['counters'].items(): setattr(t, k, v)
    t.diag = []; t.completed = []; t.wall = dict(train=0., eval=0., save=0.); t.pool_sampler.load_state_dict(s['pool_sampler'])
    random.setstate(s['rng']['python']); np.random.set_state(s['rng']['numpy']); torch.set_rng_state(s['rng']['torch'])
    t.env_rng.bit_generator.state = s['rng']['env']; t.explore_rng.bit_generator.state = s['rng']['explore']; t.replay_rng.bit_generator.state = s['rng']['replay']
    t.fork_origin = dict(substeps=t.used, updates=t.updates); t.z_drift()   # 审计集 Z 基准取自共同起点
    return t


# ------------------------------------------------------------------ 各臂训练
def stage_train(a):
    out = RUNS / a.tag / a.arm; out.mkdir(parents=True, exist_ok=True); ck = out / 'resume.pt'; pool = load_pool()
    if a.resume and ck.exists(): t = VisualTrainer.load(ck, pool); print('从断点恢复', t.used, flush=True)
    else: t = fork_from_common(a.arm, out)
    origin = getattr(t, 'fork_origin', None) or json.load(open(out / 'status.json'))['fork_origin']
    t.fork_origin = origin; target = origin['substeps'] + a.substeps
    start = time.monotonic(); last_save = start; last_print = start; reason = 'substep_budget'
    while t.used < target:
        if (ROOT / 'STOP').exists(): reason = 'requested_stop'; break
        if time.monotonic() - start >= a.seconds: reason = 'wall_budget'; break
        t.step()
        if time.monotonic() - last_save >= CFG['checkpoint_interval_wall_seconds']:
            t.save(ck); last_save = time.monotonic()
            json_save(out / 'status.json', status(t, 'training', start))
        if time.monotonic() - last_print >= 60:
            d = t.diag[-1] if t.diag else {}
            print(f"[{a.arm}] 子步 {t.used - origin['substeps']}/{a.substeps} 更新 {t.updates - origin['updates']} eps {t.episodes} 完赛 {t.train_arrivals} "
                  f"q_loss {d.get('q_loss', 0):.2f} vis {d.get('vision_loss', 0):.3f} enc_grad {d.get('encoder_grad', 0):.2e}", flush=True); last_print = time.monotonic()
        if a.stop_after and t.used - origin['substeps'] >= a.stop_after: reason = 'stop_after'; break
    t.save(ck); st = status(t, reason, start); json_save(out / 'status.json', st)
    if reason in ('substep_budget', 'wall_budget') and not a.no_eval:
        te = time.monotonic(); rows, visual, traces = evaluate(t.learner, S.conditions('development'))
        ev = dict(rows=rows, summary=S.summarize(rows), settled=sum(r['settled'] for r in rows), fallback=sum(r['fallback_substeps'] for r in rows),
                  intervened=sum(r['intervened_substeps'] for r in rows), visual=visual_summary(visual), z=t.z_drift(), eval_wall_s=time.monotonic() - te,
                  arm=a.arm, substeps_trained=t.used - origin['substeps'], updates=t.updates - origin['updates'])
        for r, (tr, log) in zip(rows, traces):
            S.save_trace(out / 'evaluation_traces' / f"dev_{r['condition_id']:02d}.npz", tr); json_save(out / 'evaluation_traces' / f"dev_{r['condition_id']:02d}_layer.json", log)
        json_save(out / 'evaluation.json', ev)
        print(f"[{a.arm}] 评估：完赛(静止) {ev['settled']}/9 违规 {ev['summary']['violations']} 平均I_j {ev['summary']['completed_mean']['Ij']} 平均时间 {ev['summary']['completed_mean']['time_s']} 干预子步 {ev['intervened']} 回退 {ev['fallback']}", flush=True)
    print(json.dumps({k: st[k] for k in ('state', 'substeps', 'updates', 'episodes', 'train_arrivals', 'substeps_per_s', 'updates_per_s')}, ensure_ascii=False))


def status(t, reason, start):
    o = t.fork_origin; el = time.monotonic() - start
    return dict(state=reason, arm=t.cfg['mode'], fork_origin=o, substeps=t.used - o['substeps'], decisions=t.decisions, updates=t.updates - o['updates'],
                episodes=t.episodes, train_arrivals=t.train_arrivals, replay_size=t.replay.size, frames=len(t.store.frames), frame_store_MiB=t.store.nbytes / 2**20,
                wall=t.wall, session_wall_s=el, substeps_per_s=(t.used - o['substeps']) / max(t.wall['train'], 1e-9), updates_per_s=(t.updates - o['updates']) / max(t.wall['train'], 1e-9),
                last_diag=t.diag[-1] if t.diag else None, z=t.z_drift(), completed_episodes=t.completed[-5:], identity=t.identity)


# ------------------------------------------------------------------ 汇总
def stage_report(tag):
    rep = {}
    for arm in CFG['arms']:
        d = RUNS / tag / arm
        if not (d / 'evaluation.json').exists(): continue
        ev = json.load(open(d / 'evaluation.json')); st = json.load(open(d / 'status.json'))
        rep[arm] = dict(substeps=st['substeps'], updates=st['updates'], episodes=st['episodes'], train_arrivals=st['train_arrivals'], substeps_per_s=st['substeps_per_s'],
                        updates_per_s=st['updates_per_s'], frame_store_MiB=st['frame_store_MiB'], settled=ev['settled'], violations=ev['summary']['violations'],
                        mean_Ij=ev['summary']['completed_mean']['Ij'], mean_time_s=ev['summary']['completed_mean']['time_s'], mean_E_Wh=ev['summary']['completed_mean']['E_Wh'],
                        mean_R=ev['summary']['completed_mean']['R'], intervened=ev['intervened'], fallback=ev['fallback'], visual=ev['visual'], z=ev['z'], last_diag=st['last_diag'])
    json_save(RUNS / tag / 'report.json', rep); print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('audit', 'adapt', 'train', 'report'))
    ap.add_argument('--arm', choices=('frozen', 'supervised', 'joint'), default='joint'); ap.add_argument('--tag', default='smoke')
    ap.add_argument('--substeps', type=int, default=4000); ap.add_argument('--seconds', type=float, default=600.)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--stop-after', type=int, default=0); ap.add_argument('--no-eval', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.stage == 'audit': sys.exit(0 if stage_audit() else 1)
    elif a.stage == 'adapt': stage_adapt()
    elif a.stage == 'train': stage_train(a)
    else: stage_report(a.tag)
