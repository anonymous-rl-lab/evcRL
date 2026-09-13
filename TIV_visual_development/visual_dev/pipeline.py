"""视觉 Z + 平滑执行层 + TD3 的同步采集、帧引用回放、三臂学习与全状态断点。

接入点：v19_deps/code/study.py 的 StudyEnv / command / comfort_loss / episode_metrics / conditions；
执行层：v19_deps/code/curve_layer.py（comfort_v2 r2：jerk 可行区间 + 绿灯结束可达性 + 80 km/h 上界），
其信息访问（模拟器已知信号时刻）保持原样并在三臂间相同；相机由同一位姿与同一绝对时刻相位驱动。
"""
import copy, hashlib, json, math, os, random, sys, time
from collections import deque
from pathlib import Path
os.environ['EVSIM_ROUTE'] = 'mini'          # 必须在导入 route20/env20 之前
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'v19_deps' / 'code', ROOT, ROOT / 'visual_dev'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, torch
import study as S, curve_layer as C
from renderer import SceneCamera, W, H, P2
from visual_z.contracts import VisualObservation, TransitionBatch, LEGACY_DIM, Z_DIM
from visual_z.model import VisualEncoder, InputAdapter, MLP, load_legacy_weights, perception_loss
from visual_z.learner import VisualTD3, Config, grad_norm
torch.set_num_threads(1)
STACK = 4
V19_FILES = ['study.py', 'env20.py', 'route20.py', 'models.py', 'plant.py', 'effcal.py', 'curve_layer.py', 'scn/arterial_mini_profile.npy']
DEV_FILES = ['renderer.py', 'pipeline.py', 'pretrain.py', 'run_stage.py']


def source_identity(config):
    h = {f: S.digest(ROOT / 'v19_deps' / 'code' / f) for f in V19_FILES}
    for extra in ('v19_deps/weights/short_route_A_nets.pt', config.get('pretrained_encoder', ''), config.get('pool_file', 'runs/pretrain/pool.pt')):
        if extra and (ROOT / extra).exists(): h[extra] = S.digest(ROOT / extra)   # v2：池与权重内容进入身份
    h.update({'visual_dev/' + f: S.digest(ROOT / 'visual_dev' / f) for f in DEV_FILES if (ROOT / 'visual_dev' / f).exists()})
    h.update({'visual_z/' + f.name: S.digest(f) for f in sorted((ROOT / 'visual_z').glob('*.py'))})
    return dict(sources=hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest(),
                config=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())


# ------------------------------------------------------------------ 平滑执行层环境
PHASES = ('red', 'yellow', 'green', 'off', 'unknown')   # 与 renderer.COLOR_CLASSES / ROI 头类别顺序一致


def inv_command(a):
    """S.command 的逆：m/s² → 归一化指令 [-1,1]。"""
    a = float(a); return float(np.clip(a / 2.6 if a > 0 else a / 3.5, -1., 1.))


class SmoothEnv(S.StudyEnv):
    """StudyEnv + comfort_v2 r2 执行层（与 rollout_layer.CurveEnv.project 逐行同义）。记录指令与实际动作。

    v3：signal_source 决定执行层的相位来源。'truth'：原行为，读仿真真值相位与倒计时（特权执行层）。
    'perceived'：只读 set_perceived() 给的感知相位（红/黄/绿/灭/未知），没有倒计时——绿灯按 assumed_green_remaining 规划通过，
    非绿或未知一律按不可通行（在停止线前设停车目标，保守）。红灯裁判（Route20.step 的 red_crossing）始终用真值，不受此开关影响。
    """
    def __init__(self, *args, signal_source='truth', assumed_green_remaining=30., **kw):
        self.layer_log = []; self.signal_source = signal_source; self.assumed_green_remaining = float(assumed_green_remaining)
        self.perceived = dict(phase='unknown', age_s=0., source='none')
        if signal_source not in ('truth', 'perceived'): raise ValueError(signal_source)
        super().__init__(*args, **kw)

    def set_perceived(self, phase, age_s=0., source='perception'):
        if phase not in PHASES: raise ValueError(phase)
        self.perceived = dict(phase=str(phase), age_s=float(age_s), source=source)

    def _passable(self, k, xs):   # 基类 a_safe/_stop_targets 用它决定是否在停止线前设停车目标
        if self.signal_source == 'truth': return super()._passable(k, xs)
        return self.perceived['phase'] == 'green'

    def _green_now_and_remaining(self, k, xs):
        if self.signal_source == 'truth':
            return S.R.signal_green(xs, self.t, self.offsets[k]), S.R.time_to_change(xs, self.t, self.offsets[k])
        return self.perceived['phase'] == 'green', self.assumed_green_remaining

    def project(self, cmd):
        fallback = super().project(cmd)
        targets = [(max(t[0] - self.x, 0.), t[1]) for t in self._stop_targets()]
        limit = S.R.V_FREE
        result, info = C.project(self.v, self.a, cmd, targets, fallback, vmax=limit)
        result = min(result, C.brake_bound(max(limit - self.v, 0.)))
        lower = info.get('lower'); upper = None if info.get('upper') is None else min(info['upper'], C.brake_bound(max(limit - self.v, 0.)))
        if not info['fallback']:
            for k, xs in enumerate(S.R.SIGNALS):
                green, remaining = self._green_now_and_remaining(k, xs)
                if self.x < xs and green:
                    action, extra = C.signal_project(self.v, self.a, cmd, (lower, upper), xs - self.x, remaining, limit)
                    info.update(extra)
                    if action is None: info['fallback'] = True; result = fallback
                    else: result = action
        truth_green = all(S.R.signal_green(xs, self.t, self.offsets[k]) for k, xs in enumerate(S.R.SIGNALS) if self.x < xs) if any(self.x < xs for xs in S.R.SIGNALS) else None
        self.layer_log.append(dict(t=float(self.t), x=float(self.x), command=float(cmd), applied=float(result),
                                   fallback=bool(info['fallback']), intervened=bool(abs(result - cmd) > 1e-9),
                                   lower=None if lower is None else float(lower), upper=None if upper is None else float(upper),
                                   perceived=self.perceived['phase'], truth_green=truth_green, signal_source=self.signal_source,
                                   perceived_detail=self.perceived.get('detail')))
        return result


class VisionEnv(SmoothEnv):
    """v4：执行层的道路事件信息全部来自 VisionMemory（停车目标、弯道入口目标、限速、前方信号是否可通行），无任何真值调用；
    裁判（Route20.step 的 red_crossing / 弯道超速）仍用真值。"""
    def __init__(self, *args, memory=None, assumed_green_remaining=30., **kw):
        self.memory = memory
        super().__init__(*args, signal_source='perceived', assumed_green_remaining=assumed_green_remaining, **kw)

    def _stop_targets(self):
        return self.memory.executor_targets(self.x) if self.memory is not None else []

    def _truth_green_for_log(self):   # 只用于日志/评估口径，不参与任何控制量
        ahead = [k for k, xs in enumerate(S.R.SIGNALS) if self.x < xs]
        return all(S.R.signal_green(S.R.SIGNALS[k], self.t, self.offsets[k]) for k in ahead) if ahead else None

    def _passable(self, k, xs):   # 基类不再调用（_stop_targets 已覆盖），保留以防
        g, _, _ = self.memory.executor_signal() if self.memory is not None else (None, None, None); return bool(g)

    A_COMFORT = 1.5   # 感知目标的舒适包络减速度 m/s²：对记忆中的停车/入弯目标，执行层把速度上限压到 sqrt(v_t² + 2·A_COMFORT·d)，
                      # 使停车在感知距离误差下仍可行（真值执行层靠精确距离在最后时刻制动；感知距离有 10–30 m 误差时必须提前）

    def project(self, cmd):
        fallback = S.E.Route20.project(self, cmd)   # 基类投影：jerk/执行器盒 + a_safe（来自记忆目标）
        targets = [(max(t[0] - self.x, 0.), t[1]) for t in self._stop_targets()]
        limit = self.memory.v_limit() if self.memory is not None else S.R.V_FREE
        for d_t, v_t in targets:   # 舒适包络：由感知目标推出的速度上限
            limit = min(limit, math.sqrt(max(v_t * v_t + 2. * self.A_COMFORT * d_t, 0.)))
        result, info = C.project(self.v, self.a, cmd, targets, fallback, vmax=limit)
        result = min(result, C.brake_bound(max(limit - self.v, 0.)))
        lower = info.get('lower'); upper = None if info.get('upper') is None else min(info['upper'], C.brake_bound(max(limit - self.v, 0.)))
        if not info['fallback'] and self.v > limit + 1e-6:   # 高于包络：在 jerk 可行区间内减速（vmax 只阻止加速，不会主动减速）
            a_env = max((limit - self.v) / S.E.DT, lower); result = min(result, a_env); upper = min(upper, a_env) if upper is not None else a_env
            info['envelope_braking'] = True
        green, remaining, d_line = self.memory.executor_signal(self.assumed_green_remaining) if self.memory is not None else (None, None, None)
        if not info['fallback'] and green:
            action, extra = C.signal_project(self.v, self.a, cmd, (lower, upper), d_line, remaining, limit)
            info.update(extra)
            if action is None: info['fallback'] = True; result = fallback
            else: result = action
        self.layer_log.append(dict(t=float(self.t), x=float(self.x), command=float(cmd), applied=float(result), fallback=bool(info['fallback']), intervened=bool(abs(result - cmd) > 1e-9),
                                   lower=None if lower is None else float(lower), upper=None if upper is None else float(upper),
                                   perceived=self.memory.sig['phase'] if self.memory is not None else 'unknown', truth_green=self._truth_green_for_log(), signal_source='vision_memory',
                                   v_limit=float(limit), targets=[(round(float(a_), 1), float(b_)) for a_, b_, *_ in self._stop_targets()], v=float(self.v),
                                   perceived_detail=self.perceived.get('detail'), memory=self.memory.snapshot() if self.memory is not None else None))
        return result


class Perception:
    """v3 执行层的相位来源：冻结的预训练编码器 ROI 灯色头（与臂无关、训练中不更新），只看最新帧。
    无地图关联（不知道灯在哪）或 ROI 为空时给 unknown。source='oracle' 时直接给真值颜色（用于分离规则效应与识别误差的探针）。"""
    def __init__(self, path=None, source='roi_head', green_threshold=0., min_consecutive=1, det_thr=0.5):
        """green_threshold：只有 P(green) ≥ 阈值才判为 green（否则判为 argmax 的非绿类或 'unknown'）；
        min_consecutive：连续 ≥ 该帧数满足绿灯条件才对执行层报 green（迟滞，只对绿→通行方向保守，非绿立即生效）。"""
        self.source = source; self.last = None; self.green_threshold = float(green_threshold); self.min_consecutive = int(min_consecutive); self._green_run = 0; self.det_thr = float(det_thr)
        if source == 'roi_head':
            ck = torch.load(path, map_location='cpu', weights_only=False); self.enc = VisualEncoder(STACK)
            self.enc.load_state_dict(ck['encoder']); self.enc.eval(); self.enc.requires_grad_(False)

    def observe(self, store, history, episode_id, t):
        """v4：两次前向——第一次（ROI 为空）得到四类检测与距离估计，用灯检测框构造预测 ROI 写回最新帧（roi_pred），
        第二次用预测 ROI 得到灯色概率。返回 (dets, color_probs)。"""
        from visual_z.model import decode_detections, predicted_roi
        from renderer import CLASSES, COLOR_CLASSES
        latest = store.frames[history[-1]]
        if self.source == 'oracle':   # 真值检测（只用于探针：分离记忆规则效应与识别误差）
            T = latest['labels']['truth']; L = latest['labels']
            dets = {'traffic_light': dict(score=float(L.get('visible', 0) == 1), dist_m=float(T['light_m'] or 0.), box=None),
                    'curve_sign': dict(score=float(T['curve_sign_m'] is not None and T['curve_sign_m'] > 1.5), dist_m=float(T['curve_sign_m'] or 0.), box=None),
                    'end_marker': dict(score=float(T['end_m'] is not None and T['end_m'] > 1.5), dist_m=float(T['end_m'] or 0.), box=None),
                    'release_sign': dict(score=float(T['release_sign_m'] is not None and T['release_sign_m'] > 1.5), dist_m=float(T['release_sign_m'] or 0.), box=None)}
            probs = None
            if L.get('visible', 0) == 1:
                probs = np.zeros(5, np.float32); probs[COLOR_CLASSES.index(L['color_truth'])] = 1.
            latest['roi_pred'] = (latest['roi'][0] if latest['roi'].ndim == 3 else latest['roi']).astype(np.float32)
            self.last = dict(dets=dets, probs=None if probs is None else probs.tolist(), roi_pixels=int(latest['roi_pred'].sum())); return dets, probs
        rec = dict(frame_ids=list(history), episode_id=episode_id, decision_time=float(t), legacy=np.zeros(LEGACY_DIM, np.float32), association_valid=0)
        obs = obs_from_frames(store, [rec], use_pred_roi=False); obs.signal_roi.zero_()
        with torch.no_grad(): out = self.enc(obs)
        dets = {}
        for c, name in enumerate(CLASSES):
            d = decode_detections(out, c); dets[name] = dict(score=float(d['score'][0, -1]), dist_m=float(d['dist_m'][0, -1]), box=[float(v) for v in d['box'][0, -1]])
        roi = predicted_roi(dets['traffic_light']['box'], H, W).numpy() if dets['traffic_light']['score'] >= self.det_thr * getattr(self, 'roi_ratio', 0.5) else np.zeros((H, W), np.float32)   # ROI 用较低阈值（灯色由记忆的轨迹门控决定是否采纳）
        latest['roi_pred'] = roi.astype(np.float32); probs = None
        if roi.sum() > 0:
            obs2 = obs_from_frames(store, [rec], use_pred_roi=True)
            with torch.no_grad(): probs = self.enc(obs2)['signal'][0, -1].softmax(0).numpy()
        self.last = dict(dets=dets, probs=None if probs is None else probs.tolist(), roi_pixels=int(roi.sum()))
        return dets, probs

    def phase(self, store, history, episode_id, t, env=None):
        latest = store.frames[history[-1]]
        if self.source == 'oracle':
            from renderer import signal_color
            return signal_color(t, env.offsets[0]) if latest['meta']['map_association'] and latest['labels'].get('visible') else 'unknown'
        if not latest['meta']['map_association']: return 'unknown'
        rec = dict(frame_ids=list(history), episode_id=episode_id, decision_time=float(t), legacy=np.zeros(LEGACY_DIM, np.float32),
                   association_valid=int(latest['meta']['association_valid']))
        obs = obs_from_frames(store, [rec])
        with torch.no_grad(): out = self.enc(obs); pr = out['signal'][0, -1].softmax(0)
        self.last = dict(probs=pr.tolist(), conf=float(pr.max()), roi_empty=bool(out['roi_empty'][0, -1]), px=float(latest['labels'].get('housing_px_h', 0.)),
                         d=float(latest['labels'].get('light_distance_m', -1.)), visible=int(latest['labels'].get('visible', 0)), occluded=int(latest['labels'].get('occluded', 0)))
        raw = PHASES[int(pr.argmax())]; p_green = float(pr[2])
        green_ok = raw == 'green' and p_green >= self.green_threshold
        self._green_run = self._green_run + 1 if green_ok else 0
        self.last['raw'] = raw; self.last['p_green'] = p_green; self.last['green_run'] = self._green_run
        if green_ok and self._green_run >= self.min_consecutive: return 'green'
        if raw == 'green': return 'unknown'   # 绿灯证据不足：按未知（不可通行）处理
        return raw


# ------------------------------------------------------------------ 帧存储（内存 + 引用计数）
class FrameStore:
    def __init__(self):
        self.frames = {}; self.refs = {}
    def put(self, fid, frame):
        assert fid not in self.frames, '帧 id 重复'
        self.frames[fid] = dict(rgb=frame['rgb'], roi=(frame['roi'][0] > 0).astype(np.uint8), meta=frame['meta'],
                                labels={k: v for k, v in frame['labels'].items() if k not in ('heat', 'heat_valid', 'boxes', 'box_valid')})
        self.refs[fid] = 0
    def retain(self, fids):
        for f in fids: self.refs[f] += 1
    def release(self, fids, protected):
        for f in fids:
            self.refs[f] -= 1
            if self.refs[f] <= 0 and f not in protected:
                del self.frames[f]; del self.refs[f]
    def get(self, fid, episode_id, decision_time):
        r = self.frames[fid]
        if r['meta']['episode_id'] != episode_id: raise ValueError('跨 episode 帧')
        if r['meta']['capture_time'] > decision_time + 1e-9: raise ValueError('未来帧泄漏')
        return r
    def state_dict(self): return dict(frames=self.frames, refs=self.refs)
    def load_state_dict(self, s): self.frames, self.refs = s['frames'], s['refs']
    @property
    def nbytes(self): return sum(f['rgb'].nbytes + f['roi'].nbytes for f in self.frames.values())


class Replay:
    """存帧引用与元数据，采样时从帧存储重编码；动作字段是投影前的归一化指令。"""
    def __init__(self, capacity, store):
        self.capacity, self.store, self.records, self.ptr = capacity, store, [], 0
    def add(self, rec, protected):
        for obs in (rec['obs'], rec['next_obs']): self.store.retain(obs['frame_ids'])
        if len(self.records) < self.capacity: self.records.append(rec)
        else:
            old = self.records[self.ptr]
            for obs in (old['obs'], old['next_obs']): self.store.release(obs['frame_ids'], protected)
            self.records[self.ptr] = rec
        self.ptr = (self.ptr + 1) % self.capacity
    @property
    def size(self): return len(self.records)
    def sample(self, rng, n): return [self.records[i] for i in rng.integers(0, len(self.records), n)]
    def state_dict(self): return dict(capacity=self.capacity, records=self.records, ptr=self.ptr)
    def load_state_dict(self, s): self.capacity, self.records, self.ptr = s['capacity'], s['records'], s['ptr']


# ------------------------------------------------------------------ 观测构造
def obs_from_frames(store, obs_records, use_pred_roi=True):
    """obs_records: list of dict(frame_ids, episode_id, decision_time, legacy, association_valid[, extra])。
    v4：帧若带 roi_pred（检测器预测的 ROI）且 use_pred_roi，则用它替代地图 ROI；记录若带 extra 则填入 obs.extra。"""
    b = len(obs_records)
    frames = np.zeros((b, STACK, 3, H, W), np.uint8); roi = np.zeros((b, STACK, 1, H, W), np.float32)
    valid = np.zeros((b, STACK), bool); age = np.zeros((b, STACK), np.float32); legacy = np.zeros((b, LEGACY_DIM), np.float32)
    assoc = np.zeros((b, 1), np.float32); has_extra = 'extra' in obs_records[0]; extra = np.zeros((b, len(obs_records[0]['extra'])), np.float32) if has_extra else None
    for i, o in enumerate(obs_records):
        fids = o['frame_ids']; k0 = STACK - len(fids)
        for j, fid in enumerate(fids):
            r = store.get(fid, o['episode_id'], o['decision_time'])
            frames[i, k0 + j] = r['rgb'].transpose(2, 0, 1)
            rp = r.get('roi_pred') if use_pred_roi else None
            roi[i, k0 + j, 0] = rp if rp is not None else (r['roi'][0] if r['roi'].ndim == 3 else r['roi'])
            valid[i, k0 + j] = True; age[i, k0 + j] = o['decision_time'] - r['meta']['capture_time']
        legacy[i] = o['legacy']; assoc[i, 0] = o['association_valid']
        if has_extra: extra[i] = o['extra']
    T = torch.as_tensor
    return VisualObservation(T(legacy), T(frames), T(valid), T(age), T(roi), T(assoc), torch.zeros(b, 1), None if extra is None else T(extra))


def labels_from_frames(frame_list):
    """离线监督池：frame_list 为 STACK 帧的列表的列表（含完整标签）。"""
    b = len(frame_list)
    heat = np.zeros((b, STACK, 8, *P2), np.float32); hv = np.ones((b, STACK, 1, *P2), np.float32)
    boxes = np.zeros((b, STACK, 4, *P2), np.float32); bv = np.zeros((b, STACK, 1, *P2), np.float32)
    sig = np.full((b, STACK), -100, np.int64)
    has_dist = 'dist' in frame_list[0][0]['labels']
    dist = np.zeros((b, STACK, 1, *P2), np.float32); dv = np.zeros((b, STACK, 1, *P2), np.float32)
    for i, seq in enumerate(frame_list):
        for j, f in enumerate(seq):
            L = f['labels']; heat[i, j] = L['heat']; boxes[i, j] = L['boxes']; bv[i, j] = L['box_valid']   # uint8/bool 存储在此处无损转回 float32
            sig[i, j] = L['signal'] if f['meta']['map_association'] else -100   # v2：地图有灯即给标签（不可见=unknown）；无灯 -100
            if has_dist: dist[i, j] = L['dist']; dv[i, j] = L['dist_valid']
    T = torch.as_tensor
    out = dict(heat=T(heat), heat_valid=T(hv), boxes=T(boxes), box_valid=T(bv), signal=T(sig))
    if has_dist: out.update(dist=T(dist), dist_valid=T(dv), presence=T((heat[:, :, :4].reshape(b, STACK, 4, -1).sum(-1) > 0).astype(np.float32)))
    return out


def pool_observation(frame_list, legacy=None):
    b = len(frame_list)
    frames = np.stack([np.stack([f['rgb'].transpose(2, 0, 1) for f in seq]) for seq in frame_list])
    roi = np.stack([np.stack([(f['roi'][0] if f['roi'].ndim == 3 else f['roi']) for f in seq]) for seq in frame_list])[:, :, None].astype(np.float32)
    age = np.tile(np.arange(STACK - 1, -1, -1, dtype=np.float32) * S.E.DT, (b, 1))
    assoc = np.array([[seq[-1]['meta']['association_valid']] for seq in frame_list], np.float32)
    T = torch.as_tensor
    return VisualObservation(T(np.zeros((b, LEGACY_DIM), np.float32) if legacy is None else legacy), T(frames),
                             torch.ones(b, STACK, dtype=torch.bool), T(age), T(roi), T(assoc), torch.zeros(b, 1))


class PoolSampler:
    """三臂共用的离线视觉监督采样计划：同一种子、同一顺序。"""
    def __init__(self, pool, batch, seed):
        self.pool, self.batch, self.rng = pool, batch, np.random.default_rng(seed); self.count = 0
    def next(self):
        idx = self.rng.integers(0, len(self.pool['sequences']), self.batch); self.count += 1
        seqs = [self.pool['sequences'][i] for i in idx]
        return pool_observation(seqs), labels_from_frames(seqs)
    def state_dict(self): return dict(rng=self.rng.bit_generator.state, count=self.count)
    def load_state_dict(self, s): self.rng.bit_generator.state = s['rng']; self.count = s['count']


# ------------------------------------------------------------------ 训练器
class VisualTrainer:
    def __init__(self, cfg, pool, out):
        self.cfg = copy.deepcopy(cfg); self.out = Path(out); self.identity = source_identity(cfg)
        seed = cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.env_rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
        self.explore_rng = np.random.default_rng(np.random.SeedSequence([seed, 22]))
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 33]))
        self.learner = VisualTD3(Config(mode=cfg['mode'], information_mode=('vision_memory' if cfg.get('world', 'v2') == 'v4' else 'camera_map'), stack=STACK,
            actor_lr=cfg['actor_lr'], critic_lr=cfg['critic_lr'], encoder_lr=cfg['encoder_lr'],
            vision_weight=cfg['vision_weight'], lambda_c=cfg['lambda_c'], tau=cfg['tau'],
            policy_delay=cfg['policy_delay'], target_noise=cfg['target_noise'], target_clip=cfg['target_clip'],
            critic_action=cfg.get('critic_action', 'command'), actor_clip_ste=bool(cfg.get('actor_clip_ste', False))))
        self.signal_source = cfg.get('signal_source', 'truth'); self.world = cfg.get('world', 'v2')
        self.perception = Perception(ROOT / cfg['pretrained_encoder'], cfg.get('perception_source', 'roi_head'), green_threshold=cfg.get('green_threshold', 0.), min_consecutive=cfg.get('min_consecutive', 1), det_thr=(cfg['det_thr'].get('traffic_light', 0.5) if isinstance(cfg.get('det_thr'), dict) else cfg.get('det_thr', 0.5))) if (self.signal_source == 'perceived' or self.world == 'v4') else None
        from vision_state import VisionMemory
        self.memory = VisionMemory(det_thr=cfg.get('det_thr', 0.5), hold_s=cfg.get('hold_s', 3.), init_votes=cfg.get('init_votes', 2), consistency_m=cfg.get('consistency_m', 40.)) if self.world == 'v4' else None
        if self.perception is not None and isinstance(cfg.get('det_thr'), dict): self.perception.det_thr = float(cfg['det_thr'].get('traffic_light', 0.5))
        # 共同起点：预训练编码器 + 冻结 4 km A 臂 actor/critic 的第一层迁移（新增 Z/元信息列初始化为零）
        enc = torch.load(ROOT / cfg['pretrained_encoder'], map_location='cpu', weights_only=False)
        self.learner.encoder.load_state_dict(enc['encoder'])
        nets = torch.load(ROOT / 'v19_deps' / 'weights' / 'short_route_A_nets.pt', map_location='cpu', weights_only=False)['networks']
        load_legacy_weights(self.learner.actor, nets['actor']); load_legacy_weights(self.learner.q1, nets['q1'], critic=True)
        load_legacy_weights(self.learner.q2, nets['q2'], critic=True)
        scale = float(cfg.get('actor_output_scale', 1.))   # v3：缩小 tanh 前的预激活，使迁移来的饱和 actor 回到有梯度的区域（1.0 = 原行为）
        if scale != 1.:
            with torch.no_grad(): self.learner.actor.f[4].weight.mul_(scale); self.learner.actor.f[4].bias.mul_(scale)
        self.learner.synchronize_targets()
        if cfg['mode'] == 'frozen': self.learner.encoder.requires_grad_(False)
        self.store = FrameStore(); self.replay = Replay(cfg['replay_capacity'], self.store)
        self.pool_sampler = PoolSampler(pool, cfg.get('pool_batch', 8), seed=cfg['pool_seed']); self.pool = pool
        self.camera = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=cfg['camera_seed'], world=cfg.get('world', 'v2'))
        self.pending = deque(); self.history = []; self.ou = 0.
        self.used = self.decisions = self.updates = self.episodes = self.train_arrivals = 0
        self.diag = []; self.wall = dict(train=0., eval=0., save=0.); self.completed = []
        self.audit_z0 = None
        self.env = None; self.new_episode()

    # ---------------------------------------------------------- 采集
    def new_episode(self):
        soc, temp = S.PACKS[int(self.env_rng.integers(0, 3))]
        v0 = float(self.env_rng.uniform(12., 20.)); off = float(self.env_rng.uniform(0., 90.))
        if self.world == 'v4':
            self.memory.reset(); self.env = VisionEnv(soc, temp, off, memory=self.memory, assumed_green_remaining=self.cfg.get('assumed_green_remaining', 30.))
        else: self.env = SmoothEnv(soc, temp, off, signal_source=self.signal_source, assumed_green_remaining=self.cfg.get('assumed_green_remaining', 30.))
        self.env.reset(v0); self.episodes_started = getattr(self, 'episodes_started', 0) + 1
        self.episode_id = f"ep{self.episodes_started:06d}"; self.camera.new_episode(self.episode_id)
        self.history = []; self.ou = 0.; self.capture()

    def capture(self):
        f = self.camera.capture(episode_id=self.episode_id, sim_time=self.env.t, pose=dict(x=self.env.x, v=self.env.v, offsets=self.env.offsets))
        fid = f"{self.episode_id}_{f['meta']['frame_index']:05d}"; self.store.put(fid, f)
        self.history.append(fid); self.history = self.history[-STACK:]
        if self.world == 'v4':   # v4：每帧感知 → 记忆更新（弯道/信号/终点），三方共享
            dets, probs = self.perception.observe(self.store, self.history, self.episode_id, self.env.t)
            self.memory.update(dets, probs, self.env.v, S.E.DT if len(self.history) > 1 else 0.)
            self.env.set_perceived(self.memory.sig['phase'], self.memory.sig['age'] if self.memory.sig['seen'] else 0., 'vision_memory')
        elif self.perception is not None:   # v3：每采一帧就更新执行层的感知相位
            self.env.set_perceived(self.perception.phase(self.store, self.history, self.episode_id, self.env.t, self.env), 0., self.perception.source)

    def current_obs_record(self):
        latest = self.store.frames[self.history[-1]]
        if self.world == 'v4':   # v4：13 维来自视觉记忆 + 车辆传感器，不调用 env.obs()（地图/真值）
            e = self.env; leg = self.memory.legacy(e.v, e.a, e.soc, e.T, e.t_end, e.t, e.t_budget)
            return dict(frame_ids=list(self.history), episode_id=self.episode_id, decision_time=float(e.t), legacy=leg,
                        association_valid=self.memory.light_detected(), extra=self.memory.extra())
        return dict(frame_ids=list(self.history), episode_id=self.episode_id, decision_time=float(self.env.t),
                    legacy=np.asarray(self.env.obs(), np.float32), association_valid=int(latest['meta']['association_valid']))

    def policy_state(self, obs_rec):
        obs = obs_from_frames(self.store, [obs_rec])
        with torch.no_grad():
            z = self.learner.encoder(obs)['z']; state = self.learner.adapter(obs, z)
        return obs, state

    def step(self):
        c = self.cfg; t0 = time.monotonic()
        rec = self.current_obs_record(); _, state = self.policy_state(rec)
        with torch.no_grad(): u = float(self.learner.actor(state)[0, 0])
        self.ou += -c['ou_theta'] * self.ou + c['ou_sigma'] * self.explore_rng.normal()
        u = float(np.clip(u + self.ou, -1., 1.)); reward = 0.; n = 0
        n0 = len(self.env.layer_log); t0_trace = len(self.env.trace)
        for _ in range(c['repeat']):
            _, r, done, info = self.env.step(S.command(u)); reward += r; self.used += 1; n += 1
            self.capture()
            if self.world == 'v4' and self.env.x > S.R.LENGTH + 100.: done = True; info = dict(info, overshoot=True)   # v4：冲出终点 100 m 终止
            if done: break
        jw = float(c.get('jerk_weight', 0.))
        if jw > 0.:   # v3 可选：学习信号加累计 jerk 惩罚（环境回报与轨迹记录不变）
            reward -= jw * float(sum(row[9] ** 2 * (row[1] - row[0]) for row in self.env.trace[t0_trace:]))
        lg = self.env.layer_log[n0]   # 决策首子步的执行层记录：实际动作与可行区间（归一化到指令空间）
        u_applied = inv_command(lg['applied']); u_lo = -1. if lg['lower'] is None else inv_command(lg['lower']); u_hi = 1. if lg['upper'] is None else inv_command(lg['upper'])
        nxt = self.current_obs_record()
        self.pending.append(dict(obs=rec, u=u, r=reward, nxt=nxt, done=float(done), n=n, u_applied=u_applied, u_lo=u_lo, u_hi=u_hi))
        if len(self.pending) >= c['nstep'] or done:
            while self.pending:
                first, last = self.pending[0], self.pending[-1]
                self.replay.add(dict(obs=first['obs'], next_obs=last['nxt'], u_command=first['u'], return_n=sum(p['r'] for p in self.pending),
                                     bootstrap_discount=(1. - last['done']) * c['gamma'] ** sum(p['n'] for p in self.pending),
                                     actual_n=len(self.pending), executed_trace_ref=f"{self.episode_id}:{len(self.env.trace)}",
                                     u_applied=first['u_applied'], u_lo=first['u_lo'], u_hi=first['u_hi']),
                                protected=set(self.history))
                self.pending.popleft()
                if not done: break
        self.decisions += 1
        if done:
            self.episodes += 1; self.train_arrivals += int(info['arrived'])
            m = S.episode_metrics(self.env); m.update(fallback_substeps=sum(l['fallback'] for l in self.env.layer_log),
                intervened_substeps=sum(l['intervened'] for l in self.env.layer_log), episode=self.episodes, settled=bool(info['arrived'] and self.env.v <= 1e-6 and abs(self.env.a) <= 1e-6))
            self.completed.append(m); self.new_episode()
        if self.used >= c['warmup'] and self.replay.size >= c['batch']: self.update()
        self.wall['train'] += time.monotonic() - t0

    # ---------------------------------------------------------- 更新
    def batch(self):
        recs = self.replay.sample(self.replay_rng, self.cfg['batch'])
        obs = obs_from_frames(self.store, [r['obs'] for r in recs]); nxt = obs_from_frames(self.store, [r['next_obs'] for r in recs])
        T = torch.as_tensor
        u = T(np.array([[r['u_command']] for r in recs], np.float32)); ret = T(np.array([[r['return_n']] for r in recs], np.float32))
        bd = T(np.array([[r['bootstrap_discount']] for r in recs], np.float32))
        ua = T(np.array([[r.get('u_applied', r['u_command'])] for r in recs], np.float32))
        lo = T(np.array([[r.get('u_lo', -1.)] for r in recs], np.float32)); hi = T(np.array([[r.get('u_hi', 1.)] for r in recs], np.float32))
        vis_obs, labels = self.pool_sampler.next()
        return TransitionBatch(obs, nxt, u, ret, bd, labels, vis_obs, ua, lo, hi)

    def update(self):
        b = self.batch(); d = self.learner.update(b); d['step'] = self.used; self.updates = self.learner.updates
        if self.updates % self.cfg.get('diag_every', 50) == 0 or not self.diag: self.diag.append(d)

    # ---------------------------------------------------------- 诊断：固定审计图像集上的 Z 漂移
    def z_drift(self):
        obs = pool_observation(self.pool['audit'])
        with torch.no_grad(): z = self.learner.encoder(obs)['z']
        if self.audit_z0 is None: self.audit_z0 = z.clone()
        return dict(z_drift=float((z - self.audit_z0).norm(dim=1).mean()), z_std=float(z.std(dim=0).mean()))

    # ---------------------------------------------------------- 断点
    def state(self):
        return dict(schema=1, identity=self.identity, cfg=self.cfg, learner=self.learner.state_dict(),
            store=self.store.state_dict(), replay=self.replay.state_dict(), pending=list(self.pending), history=list(self.history),
            camera=self.camera.state_dict(), env=copy.deepcopy(self.env.__dict__), episode_id=self.episode_id, episodes_started=self.episodes_started,
            ou=self.ou, counters=dict(used=self.used, decisions=self.decisions, updates=self.updates, episodes=self.episodes, train_arrivals=self.train_arrivals),
            diag=self.diag, wall=self.wall, completed=self.completed, pool_sampler=self.pool_sampler.state_dict(), audit_z0=self.audit_z0,
            fork_origin=getattr(self, 'fork_origin', None),
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     env=self.env_rng.bit_generator.state, explore=self.explore_rng.bit_generator.state, replay=self.replay_rng.bit_generator.state),
            runtime=dict(torch=torch.__version__, numpy=np.__version__))

    def save(self, path):
        t0 = time.monotonic(); p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + '.tmp')
        with tmp.open('wb') as f: torch.save(self.state(), f); f.flush(); os.fsync(f.fileno())
        if p.exists(): os.replace(p, p.with_name(p.stem + '.previous' + p.suffix))
        os.replace(tmp, p); self.wall['save'] += time.monotonic() - t0

    @classmethod
    def load(cls, path, pool):
        s = torch.load(path, map_location='cpu', weights_only=False)   # 仅信任本地生成的研究断点
        t = cls(s['cfg'], pool, Path(path).parent)
        if s['identity'] != t.identity: raise ValueError('源码/配置/数据身份不一致，拒绝恢复')
        if s['runtime'] != dict(torch=torch.__version__, numpy=np.__version__): raise ValueError('运行时版本不一致，拒绝恢复')
        t.fork_origin = s.get('fork_origin')
        t.learner.load_state_dict(s['learner']); t.store.load_state_dict(s['store']); t.replay.load_state_dict(s['replay'])
        t.pending = deque(s['pending']); t.history = list(s['history']); t.camera.load_state_dict(s['camera'])
        t.env.__dict__ = s['env']; t.episode_id = s['episode_id']; t.episodes_started = s['episodes_started']; t.ou = s['ou']
        for k, v in s['counters'].items(): setattr(t, k, v)
        t.diag, t.wall, t.completed, t.audit_z0 = s['diag'], s['wall'], s['completed'], s['audit_z0']
        t.pool_sampler.load_state_dict(s['pool_sampler'])
        random.setstate(s['rng']['python']); np.random.set_state(s['rng']['numpy']); torch.set_rng_state(s['rng']['torch'])
        t.env_rng.bit_generator.state = s['rng']['env']; t.explore_rng.bit_generator.state = s['rng']['explore']; t.replay_rng.bit_generator.state = s['rng']['replay']
        return t


# ------------------------------------------------------------------ 评估（固定 actor，相机驱动，平滑执行层）
def evaluate(learner, conditions, camera_seed=1000, signal_source='truth', perception=None, assumed_green_remaining=30., world='v2', memory_kw=None, hide=()):
    """world='v4'：相机 v4 世界，执行层为 VisionEnv（记忆驱动），观测由记忆生成；hide 可含 'curve_sign'/'light_color'/'release_sign'/'end_marker'
    用于“驾驶是否依赖视觉”的消融（画面里不画该物体 / 灯色改为灭）。"""
    """固定工况闭环评估。v2e：每个工况用独立的相机外观种子（camera_seed*1000+i）与独立噪声流（camera_seed*1000+500+i），
    外观不再依赖之前工况采了多少帧——三臂在同一工况下看到的光照/雾/噪声/遮挡逐位相同（v2d 及以前共用一个生成器，
    轨迹长度不同的臂从第 3 个工况起外观不同）。"""
    rows = []; visual = []; traces = []; store = FrameStore()
    from vision_state import VisionMemory
    for i, (soc, temp, v0, off) in enumerate(conditions):
        camera = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=camera_seed * 1000 + i, noise_seed=camera_seed * 1000 + 500 + i, world=world); camera.hide = set(hide)
        if world == 'v4':
            memory = VisionMemory(**(memory_kw or {})); env = VisionEnv(soc, temp, off, memory=memory, assumed_green_remaining=assumed_green_remaining)
        else: memory = None; env = SmoothEnv(soc, temp, off, signal_source=signal_source, assumed_green_remaining=assumed_green_remaining)
        env.reset(v0); eid = f"eval{i:02d}"; camera.new_episode(eid); hist = []
        def cap():
            f = camera.capture(episode_id=eid, sim_time=env.t, pose=dict(x=env.x, v=env.v, offsets=env.offsets))
            fid = f"{eid}_{f['meta']['frame_index']:05d}"; store.put(fid, f); hist.append(fid); del hist[:-STACK]
            if world == 'v4':
                dets, probs = perception.observe(store, hist, eid, env.t); memory.update(dets, probs, env.v, S.E.DT if len(hist) > 1 else 0.)
                env.set_perceived(memory.sig['phase'], memory.sig['age'] if memory.sig['seen'] else 0., 'vision_memory'); env.perceived['detail'] = dict(perception.last, memory=memory.snapshot())
            elif perception is not None:
                env.set_perceived(perception.phase(store, hist, eid, env.t, env), 0., perception.source); env.perceived['detail'] = perception.last
            return f
        cap(); done = False
        while not done and env.t < 600:
            if world == 'v4':
                rec = dict(frame_ids=list(hist), episode_id=eid, decision_time=float(env.t), legacy=memory.legacy(env.v, env.a, env.soc, env.T, env.t_end, env.t, env.t_budget),
                           association_valid=memory.light_detected(), extra=memory.extra())
            else:
                rec = dict(frame_ids=list(hist), episode_id=eid, decision_time=float(env.t), legacy=np.asarray(env.obs(), np.float32),
                           association_valid=int(store.frames[hist[-1]]['meta']['association_valid']))
            obs = obs_from_frames(store, [rec])
            with torch.no_grad():
                out = learner.encoder(obs); u = float(learner.actor(learner.adapter(obs, out['z']))[0, 0])
            latest = store.frames[hist[-1]]['labels']
            lm = store.frames[hist[-1]]['meta']
            if lm['map_association'] and latest.get('light_distance_m') is not None:
                visual.append(dict(pred=int(out['signal'][0, -1].argmax()), pred_z=int(out['signal_z'][0].argmax()), truth=int(latest['signal']), px=float(latest['housing_px_h']),
                                   lamp_px_d=float(latest['lamp_px_d']), d=float(latest['light_distance_m']), roi_nonempty=int(lm['roi_pixels'] > 0), visible=int(latest['visible']), occluded=int(latest['occluded'])))
            cmd = S.command(u)
            for _ in range(4):
                if env.x >= 3999 and env.v <= .3: cmd = 0.
                _, r, d, info = env.step(cmd); cap()
                overshoot = env.x > S.R.LENGTH + 100.   # v4：无地图时可能冲过终点；驶出终点 100 m 视为失败终止
                done = bool(info['red_crossing'] or (info['arrived'] and env.v <= 1e-6 and abs(env.a) <= 1e-6) or env.t >= 600 or overshoot)
                if done: break
        m = S.episode_metrics(env); m.update(condition_id=i, settled=bool(info['arrived'] and env.v <= 1e-6 and abs(env.a) <= 1e-6), overshoot=bool(env.x > S.R.LENGTH + 100.),
            fallback_substeps=sum(l['fallback'] for l in env.layer_log), intervened_substeps=sum(l['intervened'] for l in env.layer_log),
            mean_abs_command_gap=float(np.mean([abs(l['applied'] - l['command']) for l in env.layer_log])),
            signal_source=('vision_memory' if world == 'v4' else signal_source), perception_source=None if perception is None else perception.source, world=world, hide=list(hide),
            curve_excess_penalty=float(getattr(env, 'e_curve_pen', 0.)), offroad_substeps=int(getattr(env, 'n_offroad', 0)),
            perceived_counts={ph: int(sum(l['perceived'] == ph for l in env.layer_log)) for ph in PHASES},
            perceived_green_truth_red_substeps=int(sum(l['perceived'] == 'green' and l['truth_green'] is False for l in env.layer_log)),
            perceived_nongreen_truth_green_substeps=int(sum(l['perceived'] != 'green' and l['truth_green'] is True for l in env.layer_log)),
            camera_appearance_seed=camera_seed * 1000 + i, camera_appearance={k: (list(v) if isinstance(v, tuple) else v) for k, v in camera.appearance.items()})
        rows.append(m); traces.append((np.asarray(env.trace, np.float64), list(env.layer_log))); store.frames.clear(); store.refs.clear()
    return rows, visual, traces


def _bin_stats(rows, key='pred'):
    rows = [v for v in rows if v.get(key) is not None]   # v2c：Z 头只对最新帧有定义（None 跳过）
    if not rows: return None
    known = [v for v in rows if v['truth'] != 4]; unk = [v for v in rows if v['truth'] == 4]
    return dict(n=len(rows), acc=float(np.mean([v[key] == v['truth'] for v in rows])),
        red_to_green=int(sum(v['truth'] == 0 and v[key] == 2 for v in rows)), green_to_nongreen=int(sum(v['truth'] == 2 and v[key] != 2 for v in rows)),
        unknown_rate=float(np.mean([v[key] == 4 for v in rows])), known_acc=float(np.mean([v[key] == v['truth'] for v in known])) if known else None,
        unknown_recall=float(np.mean([v[key] == 4 for v in unk])) if unk else None, n_known=len(known), n_unknown=len(unk))


def visual_summary(visual):
    """灯色识别分层报告（v2）：按名义灯箱像素高与按实际渲染光斑直径两种分层；ROI 头与 Z 头分别报告。"""
    out = {}
    for name, key, bins in (('housing_px_h', 'px', [(0, 4), (4, 8), (8, 16), (16, 1e9)]), ('lamp_px_d', 'lamp_px_d', [(0, 1e-9), (1e-9, 3), (3, 6), (6, 1e9)])):
        for lo, hi in bins:
            rows = [v for v in visual if lo <= v[key] < hi]
            if not rows: continue
            label = f"{name}:{lo:g}-{hi if hi < 1e9 else 'inf'}"
            out[label] = dict(roi_head=_bin_stats(rows, 'pred'), z_head=_bin_stats(rows, 'pred_z'))
    out['all'] = dict(roi_head=_bin_stats(visual, 'pred'), z_head=_bin_stats(visual, 'pred_z'))
    return out


# ------------------------------------------------------------------ v2 新增诊断
def z_decodability(encoder, sequences, seed=0, steps=400):
    """Z 是否保留灯色：在固定序列集上取 Z（最新帧有地图关联者），训练线性 softmax 探针（一半训练一半测试），并报告 signal_z 头的准确率。"""
    seqs = [s for s in sequences if s[-1]['meta']['map_association']]
    Z, y, yz = [], [], []
    for i in range(0, len(seqs), 32):
        chunk = seqs[i:i + 32]; obs = pool_observation(chunk)
        with torch.no_grad(): out = encoder(obs)
        Z.append(out['z']); yz.append(out['signal_z'].argmax(-1)); y += [int(s[-1]['labels']['signal']) for s in chunk]
    Z = torch.cat(Z); y = torch.as_tensor(y); yz = torch.cat(yz); n = len(y)
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g); tr, te = perm[: n // 2], perm[n // 2:]
    probe = torch.nn.Linear(Z.shape[1], 5); opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(probe(Z[tr]), y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pred = probe(Z[te]).argmax(-1)
    known = y[te] != 4
    return dict(n=n, probe_test_acc=float((pred == y[te]).float().mean()), probe_test_known_acc=float((pred[known] == y[te][known]).float().mean()) if known.any() else None,
                probe_train_acc=float((probe(Z[tr]).argmax(-1) == y[tr]).float().mean()), signal_z_head_acc=float((yz == y).float().mean()),
                chance=float(torch.bincount(y, minlength=5).max() / n), class_counts=torch.bincount(y, minlength=5).tolist())


def image_swap_sensitivity(learner, seeds=(0, 1, 2), distances=range(40, 420, 20), v=15., t=100., truth_offsets=(0., 30.)):
    """同一车辆状态、同一外观、同一噪声下把灯色强制为 red / green，比较 actor 指令、Z，以及经同一执行层投影后的实际动作
    与随后一个决策周期（4 子步）的回报：策略是否对图像灯色有响应，响应经执行层后是否保留。仿真真值相位取两种（offset 0 → 真值绿，offset 30 → 真值红），
    按真值分层汇总（by_truth），共 3 种子 × 19 距离 × 2 相位 = 114 对。
    v2e：每个 (seed, d) 只调用一次 new_episode（外观），两种颜色各自把噪声流重置到同一种子；v2d 及以前先 new_episode 再重置噪声，
    第一对（40 m）的外观在两色之间不同。仿真真值信号相位对两色相同（由 offset=0、t 决定），记录在 truth_green 字段。"""
    rows = []
    for seed in seeds:
      for off in truth_offsets:   # 仿真真值相位：offset 0 → t=100 s 时真值绿灯；offset 30 → 真值红灯。图像灯色与真值无关地被强制
        cam = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=1000 + seed, noise_seed=seed)
        for d in distances:
            x = S.R.SIGNALS[0] + 14. - d; cam.new_episode(f'swap{seed}_{d}')
            out = {}
            for color in ('red', 'green'):
                cam.noise_rng = np.random.default_rng(seed)   # 相同噪声种子；外观由本 (seed, d) 的 new_episode 固定
                env = SmoothEnv(.85, 288.15, off); env.reset(v); env.x = x; env.t = t; env.a = 0.
                store = FrameStore(); fids = []
                for k in range(STACK):
                    f = cam.capture(episode_id=cam.episode_id, sim_time=t - (STACK - 1 - k) * S.E.DT, pose=dict(x=x - (STACK - 1 - k) * v * S.E.DT, v=v, offsets=[off]), force_color=color)
                    fid = f"{cam.episode_id}_{color}_{k}"; store.put(fid, f); fids.append(fid)
                rec = dict(frame_ids=fids, episode_id=cam.episode_id, decision_time=t, legacy=np.asarray(env.obs(), np.float32), association_valid=int(store.frames[fids[-1]]['meta']['association_valid']))
                obs = obs_from_frames(store, [rec])
                with torch.no_grad():
                    o = learner.encoder(obs); st = learner.adapter(obs, o['z']); u = float(learner.actor(st)[0, 0])
                cmd = S.command(u); rsum = 0.
                for _ in range(4):
                    _, r, _, info = env.step(cmd); rsum += float(r)
                tr = np.asarray(env.trace, np.float64)[-4:]
                out[color] = dict(u=u, z=o['z'][0], pred=int(o['signal'][0, -1].argmax()), pred_z=int(o['signal_z'][0].argmax()),
                                  a=float(env.layer_log[0]['applied']), a_mean=float(np.mean([l['applied'] for l in env.layer_log])), r=rsum,
                                  fallback_substeps=int(sum(l['fallback'] for l in env.layer_log)), jerk_max=float(np.abs(tr[:, 9]).max()),
                                  stop_target_m=float(min([t[0] - x for t in env._stop_targets() if t[0] > x] or [np.inf])))
            truth_green = bool(S.R.signal_green(S.R.SIGNALS[0], t, off))
            rows.append(dict(seed=seed, d=float(d), truth_offset=float(off), truth_green=truth_green, u_red=out['red']['u'], u_green=out['green']['u'], du=out['green']['u'] - out['red']['u'],
                             cmd_red=S.command(out['red']['u']), cmd_green=S.command(out['green']['u']),
                             a_red=out['red']['a'], a_green=out['green']['a'], da=out['green']['a'] - out['red']['a'],
                             a_mean_red=out['red']['a_mean'], a_mean_green=out['green']['a_mean'], da_mean=out['green']['a_mean'] - out['red']['a_mean'],
                             r_red=out['red']['r'], r_green=out['green']['r'], dr=out['green']['r'] - out['red']['r'],
                             fallback_red=out['red']['fallback_substeps'], fallback_green=out['green']['fallback_substeps'], jerk_max_red=out['red']['jerk_max'], jerk_max_green=out['green']['jerk_max'],
                             feasible=bool(out['red']['fallback_substeps'] == 0 and out['green']['fallback_substeps'] == 0),
                             dz=float((out['green']['z'] - out['red']['z']).norm()), pred_red=out['red']['pred'], pred_green=out['green']['pred'],
                             predz_red=out['red']['pred_z'], predz_green=out['green']['pred_z']))
    def summ(rs):
        du = np.array([r['du'] for r in rs]); dz = np.array([r['dz'] for r in rs]); da = np.array([r['da'] for r in rs]); dr = np.array([r['dr'] for r in rs]); dam = np.array([r['da_mean'] for r in rs])
        return dict(n=len(rs), n_infeasible=int(sum(not r['feasible'] for r in rs)), max_jerk=float(max(max(r['jerk_max_red'], r['jerk_max_green']) for r in rs)),
                    mean_abs_du=float(np.abs(du).mean()), max_abs_du=float(np.abs(du).max()), frac_abs_du_gt_0_05=float(np.mean(np.abs(du) > .05)),
                    mean_du_green_minus_red=float(du.mean()), mean_dz=float(dz.mean()),
                    first_substep_mean_abs_da=float(np.abs(da).mean()), first_substep_frac_changed=float(np.mean(np.abs(da) > 1e-9)),
                    mean_abs_da_4substep=float(np.abs(dam).mean()), max_abs_da_4substep=float(np.abs(dam).max()), frac_changed_4substep=float(np.mean(np.abs(dam) > 1e-9)), frac_abs_da_4substep_gt_0_05=float(np.mean(np.abs(dam) > .05)),
                    mean_da_4substep_green_minus_red=float(dam.mean()),
                    mean_abs_da=float(np.abs(da).mean()), max_abs_da=float(np.abs(da).max()), frac_abs_da_gt_0_05=float(np.mean(np.abs(da) > .05)),
                    mean_da_green_minus_red=float(da.mean()), mean_dr_green_minus_red=float(dr.mean()), frac_dr_positive=float(np.mean(dr > 1e-9)), frac_dr_negative=float(np.mean(dr < -1e-9)),
                    roi_head_color_correct=float(np.mean([r['pred_red'] == 0 and r['pred_green'] == 2 for r in rs])),
                    z_head_color_correct=float(np.mean([r['predz_red'] == 0 and r['predz_green'] == 2 for r in rs])))
    res = summ(rows); res['truth_green_frac'] = float(np.mean([r['truth_green'] for r in rows]))
    res['by_truth'] = {'green': summ([r for r in rows if r['truth_green']]), 'red': summ([r for r in rows if not r['truth_green']])}
    feas = [r for r in rows if r['feasible']]; res['feasible_only'] = summ(feas) if feas else None
    res['note'] = ('a/da 为首子步实际动作；a_mean/da_mean 为四子步平均实际动作（指令保持四子步，执行约束可能只在首子步把差异压为 0）。'
                   'feasible=False 的对（真值红、距灯 40 m、v=15 m/s：停车目标仅 24 m，执行层 stop_distance_infeasible 触发紧急回退，jerk 达 7）单列，feasible_only 为剔除后的汇总。'
                   '真值相位只影响执行层与回报；actor 的输入在两种真值下相同（通道 7/8 被屏蔽），故 Δu 与真值无关是输入构造决定的，不能用来判断是否利用了灯色语义。')
    res['rows'] = rows; return res


def executor_independence_check(n=50, seed=0):
    """执行层只依赖环境状态与指令：同一状态的两个副本对随机指令给出相同动作；且 project 的代码不引用编码器/学习器。"""
    import copy as _copy, dis
    rng = np.random.default_rng(seed); env = SmoothEnv(.9, 263.15, 20.); env.reset(16.)
    for _ in range(60): env.step(float(rng.uniform(-2, 2)))
    twin = _copy.deepcopy(env); same = True
    for _ in range(n):
        cmd = float(rng.uniform(-3.5, 2.6)); same &= env.project(cmd) == twin.project(cmd)
    names = set(SmoothEnv.project.__code__.co_names)
    return dict(same_output_on_twin_state=bool(same), references_encoder_or_policy=bool(names & {'encoder', 'learner', 'actor', 'z'}))
