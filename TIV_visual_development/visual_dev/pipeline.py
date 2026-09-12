"""视觉 Z + 平滑执行层 + TD3 的同步采集、帧引用回放、三臂学习与全状态断点。

接入点：v19_deps/code/study.py 的 StudyEnv / command / comfort_loss / episode_metrics / conditions；
执行层：v19_deps/code/curve_layer.py（comfort_v2 r2：jerk 可行区间 + 绿灯结束可达性 + 80 km/h 上界），
其信息访问（模拟器已知信号时刻）保持原样并在三臂间相同；相机由同一位姿与同一绝对时刻相位驱动。
"""
import copy, hashlib, json, os, random, sys, time
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
    h.update({'visual_dev/' + f: S.digest(ROOT / 'visual_dev' / f) for f in DEV_FILES if (ROOT / 'visual_dev' / f).exists()})
    h.update({'visual_z/' + f.name: S.digest(f) for f in sorted((ROOT / 'visual_z').glob('*.py'))})
    return dict(sources=hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest(),
                config=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())


# ------------------------------------------------------------------ 平滑执行层环境
class SmoothEnv(S.StudyEnv):
    """StudyEnv + comfort_v2 r2 执行层（与 rollout_layer.CurveEnv.project 逐行同义）。记录指令与实际动作。"""
    def __init__(self, *args, **kw):
        self.layer_log = []
        super().__init__(*args, **kw)

    def project(self, cmd):
        fallback = super().project(cmd)
        targets = [(max(t[0] - self.x, 0.), t[1]) for t in self._stop_targets()]
        limit = S.R.V_FREE
        result, info = C.project(self.v, self.a, cmd, targets, fallback, vmax=limit)
        result = min(result, C.brake_bound(max(limit - self.v, 0.)))
        if not info['fallback']:
            for k, xs in enumerate(S.R.SIGNALS):
                if self.x < xs and S.R.signal_green(xs, self.t, self.offsets[k]):
                    remaining = S.R.time_to_change(xs, self.t, self.offsets[k])
                    action, extra = C.signal_project(self.v, self.a, cmd,
                        (info['lower'], min(info['upper'], C.brake_bound(max(limit - self.v, 0.)))), xs - self.x, remaining, limit)
                    info.update(extra)
                    if action is None: info['fallback'] = True; result = fallback
                    else: result = action
        self.layer_log.append(dict(t=float(self.t), x=float(self.x), command=float(cmd), applied=float(result),
                                   fallback=bool(info['fallback']), intervened=bool(abs(result - cmd) > 1e-9)))
        return result


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
def obs_from_frames(store, obs_records):
    """obs_records: list of dict(frame_ids, episode_id, decision_time, legacy, association_valid)。"""
    b = len(obs_records)
    frames = np.zeros((b, STACK, 3, H, W), np.uint8); roi = np.zeros((b, STACK, 1, H, W), np.float32)
    valid = np.zeros((b, STACK), bool); age = np.zeros((b, STACK), np.float32); legacy = np.zeros((b, LEGACY_DIM), np.float32)
    assoc = np.zeros((b, 1), np.float32)
    for i, o in enumerate(obs_records):
        fids = o['frame_ids']; k0 = STACK - len(fids)
        for j, fid in enumerate(fids):
            r = store.get(fid, o['episode_id'], o['decision_time'])
            frames[i, k0 + j] = r['rgb'].transpose(2, 0, 1); roi[i, k0 + j, 0] = r['roi']
            valid[i, k0 + j] = True; age[i, k0 + j] = o['decision_time'] - r['meta']['capture_time']
        legacy[i] = o['legacy']; assoc[i, 0] = o['association_valid']
    T = torch.as_tensor
    return VisualObservation(T(legacy), T(frames), T(valid), T(age), T(roi), T(assoc), torch.zeros(b, 1))


def labels_from_frames(frame_list):
    """离线监督池：frame_list 为 STACK 帧的列表的列表（含完整标签）。"""
    b = len(frame_list)
    heat = np.zeros((b, STACK, 8, *P2), np.float32); hv = np.ones((b, STACK, 1, *P2), np.float32)
    boxes = np.zeros((b, STACK, 4, *P2), np.float32); bv = np.zeros((b, STACK, 1, *P2), np.float32)
    sig = np.full((b, STACK), -100, np.int64)
    for i, seq in enumerate(frame_list):
        for j, f in enumerate(seq):
            L = f['labels']; heat[i, j] = L['heat']; boxes[i, j] = L['boxes']; bv[i, j] = L['box_valid']   # uint8/bool 存储在此处无损转回 float32
            sig[i, j] = L['signal'] if f['meta']['association_valid'] else -100
    T = torch.as_tensor
    return dict(heat=T(heat), heat_valid=T(hv), boxes=T(boxes), box_valid=T(bv), signal=T(sig))


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
        self.learner = VisualTD3(Config(mode=cfg['mode'], information_mode='camera_map', stack=STACK,
            actor_lr=cfg['actor_lr'], critic_lr=cfg['critic_lr'], encoder_lr=cfg['encoder_lr'],
            vision_weight=cfg['vision_weight'], lambda_c=cfg['lambda_c'], tau=cfg['tau'],
            policy_delay=cfg['policy_delay'], target_noise=cfg['target_noise'], target_clip=cfg['target_clip']))
        # 共同起点：预训练编码器 + 冻结 4 km A 臂 actor/critic 的第一层迁移（新增 Z/元信息列初始化为零）
        enc = torch.load(ROOT / cfg['pretrained_encoder'], map_location='cpu', weights_only=False)
        self.learner.encoder.load_state_dict(enc['encoder'])
        nets = torch.load(ROOT / 'v19_deps' / 'weights' / 'short_route_A_nets.pt', map_location='cpu', weights_only=False)['networks']
        load_legacy_weights(self.learner.actor, nets['actor']); load_legacy_weights(self.learner.q1, nets['q1'], critic=True)
        load_legacy_weights(self.learner.q2, nets['q2'], critic=True); self.learner.synchronize_targets()
        if cfg['mode'] == 'frozen': self.learner.encoder.requires_grad_(False)
        self.store = FrameStore(); self.replay = Replay(cfg['replay_capacity'], self.store)
        self.pool_sampler = PoolSampler(pool, cfg.get('pool_batch', 8), seed=cfg['pool_seed']); self.pool = pool
        self.camera = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=cfg['camera_seed'])
        self.pending = deque(); self.history = []; self.ou = 0.
        self.used = self.decisions = self.updates = self.episodes = self.train_arrivals = 0
        self.diag = []; self.wall = dict(train=0., eval=0., save=0.); self.completed = []
        self.audit_z0 = None
        self.env = None; self.new_episode()

    # ---------------------------------------------------------- 采集
    def new_episode(self):
        soc, temp = S.PACKS[int(self.env_rng.integers(0, 3))]
        v0 = float(self.env_rng.uniform(12., 20.)); off = float(self.env_rng.uniform(0., 90.))
        self.env = SmoothEnv(soc, temp, off); self.env.reset(v0); self.episodes_started = getattr(self, 'episodes_started', 0) + 1
        self.episode_id = f"ep{self.episodes_started:06d}"; self.camera.new_episode(self.episode_id)
        self.history = []; self.ou = 0.; self.capture()

    def capture(self):
        f = self.camera.capture(episode_id=self.episode_id, sim_time=self.env.t, pose=dict(x=self.env.x, v=self.env.v, offsets=self.env.offsets))
        fid = f"{self.episode_id}_{f['meta']['frame_index']:05d}"; self.store.put(fid, f)
        self.history.append(fid); self.history = self.history[-STACK:]

    def current_obs_record(self):
        latest = self.store.frames[self.history[-1]]
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
        for _ in range(c['repeat']):
            _, r, done, info = self.env.step(S.command(u)); reward += r; self.used += 1; n += 1
            self.capture()
            if done: break
        nxt = self.current_obs_record()
        self.pending.append(dict(obs=rec, u=u, r=reward, nxt=nxt, done=float(done), n=n))
        if len(self.pending) >= c['nstep'] or done:
            while self.pending:
                first, last = self.pending[0], self.pending[-1]
                self.replay.add(dict(obs=first['obs'], next_obs=last['nxt'], u_command=first['u'], return_n=sum(p['r'] for p in self.pending),
                                     bootstrap_discount=(1. - last['done']) * c['gamma'] ** sum(p['n'] for p in self.pending),
                                     actual_n=len(self.pending), executed_trace_ref=f"{self.episode_id}:{len(self.env.trace)}"),
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
        vis_obs, labels = self.pool_sampler.next()
        return TransitionBatch(obs, nxt, u, ret, bd, labels, vis_obs)

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
        if s['identity'] != t.identity: raise ValueError('源码/配置身份不一致，拒绝恢复')
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
def evaluate(learner, conditions, camera_seed=1000):
    rows = []; visual = []; traces = []
    camera = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=camera_seed); store = FrameStore()
    for i, (soc, temp, v0, off) in enumerate(conditions):
        env = SmoothEnv(soc, temp, off); env.reset(v0); eid = f"eval{i:02d}"; camera.new_episode(eid); hist = []
        def cap():
            f = camera.capture(episode_id=eid, sim_time=env.t, pose=dict(x=env.x, v=env.v, offsets=env.offsets))
            fid = f"{eid}_{f['meta']['frame_index']:05d}"; store.put(fid, f); hist.append(fid); del hist[:-STACK]; return f
        cap(); done = False
        while not done and env.t < 600:
            rec = dict(frame_ids=list(hist), episode_id=eid, decision_time=float(env.t), legacy=np.asarray(env.obs(), np.float32),
                       association_valid=int(store.frames[hist[-1]]['meta']['association_valid']))
            obs = obs_from_frames(store, [rec])
            with torch.no_grad():
                out = learner.encoder(obs); u = float(learner.actor(learner.adapter(obs, out['z']))[0, 0])
            latest = store.frames[hist[-1]]['labels']
            if latest['light_distance_m'] is not None and rec['association_valid']:
                visual.append(dict(pred=int(out['signal'][0, -1].argmax()), truth=int(latest['signal']), px=float(latest['light_px_h']), d=float(latest['light_distance_m'])))
            cmd = S.command(u)
            for _ in range(4):
                if env.x >= 3999 and env.v <= .3: cmd = 0.
                _, r, d, info = env.step(cmd); cap()
                done = bool(info['red_crossing'] or (info['arrived'] and env.v <= 1e-6 and abs(env.a) <= 1e-6) or env.t >= 600)
                if done: break
        m = S.episode_metrics(env); m.update(condition_id=i, settled=bool(info['arrived'] and env.v <= 1e-6 and abs(env.a) <= 1e-6),
            fallback_substeps=sum(l['fallback'] for l in env.layer_log), intervened_substeps=sum(l['intervened'] for l in env.layer_log),
            mean_abs_command_gap=float(np.mean([abs(l['applied'] - l['command']) for l in env.layer_log])))
        rows.append(m); traces.append((np.asarray(env.trace, np.float64), list(env.layer_log))); store.frames.clear(); store.refs.clear()
    return rows, visual, traces


def visual_summary(visual):
    """按灯箱像素高分层的灯色识别：正确率、红→绿误判、green→非绿、unknown 率。"""
    bins = [(0, 4), (4, 8), (8, 16), (16, 1e9)]; out = {}
    for lo, hi in bins:
        rows = [v for v in visual if lo <= v['px'] < hi]
        if not rows: continue
        known = [v for v in rows if v['truth'] != 4]
        out[f'{lo}-{hi if hi < 1e9 else "inf"}px'] = dict(n=len(rows), acc=float(np.mean([v['pred'] == v['truth'] for v in rows])),
            red_to_green=int(sum(v['truth'] == 0 and v['pred'] == 2 for v in rows)), green_to_nongreen=int(sum(v['truth'] == 2 and v['pred'] != 2 for v in rows)),
            unknown_rate=float(np.mean([v['pred'] == 4 for v in rows])), known_acc=float(np.mean([v['pred'] == v['truth'] for v in known])) if known else None)
    return out
