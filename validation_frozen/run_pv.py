"""R1/R2 runner: frozen-inference visual evaluation (P-V protocol) with explicit memory-profile injection.

Derived from visual/visual_dev/pipeline.py::evaluate (current protocol: the true-position terminal cmd=0 branch
is NOT present). Camera, frame store, perception, learner adapter, VisionEnv executor and physics are the
archived objects; only the memory object is replaced by the packaged VisionMemory (software/EvcRL) constructed
from a frozen profile and fed the actual substep displacement. Under profile paper_v25 that memory is the
archived rule set (legacy propagation, no guard); under guard_v26 it propagates by the actual displacement
and rejects an inward fused crossing of the 15 m range-freeze threshold.

Per episode (atomic, resumable): trace.npz (archived 24-column substep trace), layer.json (executor records),
substeps.json.gz (augmented per-substep log: pre/post state, displacement, detections, phase probabilities,
predicted/candidate/accepted range, guard flag, memory snapshot, targets, interval, request/applied, fallback,
judge flags, buffered margins), summary.json, DONE.
"""
import argparse, copy, gzip, hashlib, json, math, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parent; VIS = REPO / 'visual'
sys.path.insert(0, str(VIS / 'visual_dev')); sys.path.insert(0, str(REPO / 'software' / 'EvcRL' / 'src'))
import numpy as np, torch
import pipeline as P
from pipeline import S, STACK, FrameStore, obs_from_frames, VisionEnv, Perception
from renderer import SceneCamera
from reevaluate import learner_from_final_nets
import curve_layer as C
from evcrl.memory import VisionMemory as ProfileMemory
from evcrl.profiles import get_profile, profile_identity
torch.set_num_threads(1)

RUNS = {0: 'v4r', 1: 'v4r_s1', 2: 'v4r_s2'}
CAMERA_BASE = 1000
J = float(C.J); DT = float(S.E.DT)
assert J == 2.0 and float(S.E.J_MAX) == 2.0 and DT == 0.5


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model_hash(learner):
    h = hashlib.sha256()
    for name in ('encoder', 'actor'):
        for k, v in sorted(learner.named_nets()[name].state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class ConstantLearner:
    """u = +1 constant normalized command; the frozen S0 encoder/adapter are kept so that the observation path is unchanged (its Z is not used)."""
    def __init__(self, base, u=1.0):
        self.encoder = base.encoder; self.adapter = base.adapter; self.u = float(u); self.cfg = base.cfg
        self._dummy = torch.nn.Linear(1, 1)   # parameter-free policy: a fixed placeholder so that the before/after model hash covers the encoder only
    def actor(self, st): return torch.full((st.shape[0], 1), self.u)
    def named_nets(self): return {'encoder': self.encoder, 'actor': self._dummy}


def load_policy(seed_index):
    ck_path = VIS / 'runs' / RUNS[seed_index] / 'v4_pilot' / 'supervised' / 'final_nets.pt'
    learner, ck = learner_from_final_nets(ck_path); cfg = ck['cfg']; assert cfg['world'] == 'v4'
    for m in learner.named_nets().values(): m.eval(); m.requires_grad_(False)
    return learner, cfg, ck_path


def make_perception(cfg):
    thr = cfg.get('det_thr', .5); det = VIS / cfg['pretrained_encoder']
    per = Perception(det, 'roi_head', det_thr=thr.get('traffic_light', .5) if isinstance(thr, dict) else thr)
    return per, det


def make_memory(profile, cfg):
    rules = get_profile(profile)['memory']
    assert rules['det_thr'] == cfg['det_thr'] and rules['hold_s'] == cfg.get('hold_s', 3.), 'profile thresholds differ from the checkpoint configuration'
    return ProfileMemory(**rules, route_length=S.R.LENGTH)


def B(d): return max(.85 * d - 6., 0.)


def margins(diag, v, a):
    if 'predicted' not in diag: return None
    D = C.backup_distance(v, a)
    def m(d): return None if d is None else (None if not math.isfinite(D) else B(d) - D)
    return dict(D_backup=(None if not math.isfinite(D) else D), backup_infinite=not math.isfinite(D),
                m_minus=m(diag['predicted']), m_cand=m(diag['candidate']), m_plus=m(diag['accepted']))


def run_episode(job, learner, cfg, perception, out, fingerprint, extra_dim):
    i = int(job['condition_id']); cond = S.conditions('development')[i]; soc, temp, v0, off = cond
    assert (soc, temp, v0, off) == (float(job['soc']), float(job['temperature_K']), float(job['initial_speed_mps']), float(job['signal_offset_s']))
    app_seed, noise_seed = CAMERA_BASE * 1000 + i, CAMERA_BASE * 1000 + 500 + i
    assert app_seed == int(job['camera_appearance_seed']) and noise_seed == int(job['camera_noise_seed'])
    camera = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=app_seed, noise_seed=noise_seed, world='v4')
    memory = make_memory(job['profile'], cfg)
    env = VisionEnv(soc, temp, off, memory=memory, assumed_green_remaining=float(cfg.get('assumed_green_remaining', 30.)))
    env.reset(v0); eid = f'eval{i:02d}'; camera.new_episode(eid); store = FrameStore(); hist = []
    h_before = model_hash(learner); t_wall = time.monotonic()

    def cap(ds, dt):
        f = camera.capture(episode_id=eid, sim_time=env.t, pose=dict(x=env.x, v=env.v, offsets=env.offsets))
        fid = f"{eid}_{f['meta']['frame_index']:05d}"; store.put(fid, f); hist.append(fid); del hist[:-STACK]
        dets, probs = perception.observe(store, hist, eid, env.t); memory.update(dets, probs, env.v, dt, odom_ds=ds)
        env.set_perceived(memory.sig['phase'], memory.sig['age'] if memory.sig['seen'] else 0., 'vision_memory'); env.perceived['detail'] = dict(perception.last, memory=memory.snapshot())
        return f

    cap(0., 0.); done = False; sub = []; n_dec = 0; fallback_creep = fallback_move = 0
    while not done and env.t < 600:
        rec = dict(frame_ids=list(hist), episode_id=eid, decision_time=float(env.t), legacy=memory.legacy(env.v, env.a, env.soc, env.T, env.t_end, env.t, env.t_budget),
                   association_valid=memory.light_detected(), extra=memory.extra(extended=extra_dim == 10))
        obs = obs_from_frames(store, [rec])
        with torch.no_grad(): out_enc = learner.encoder(obs); u = float(learner.actor(learner.adapter(obs, out_enc['z']))[0, 0])
        cmd = S.command(u)
        for j in range(4):
            pre = dict(t=float(env.t), x=float(env.x), v=float(env.v), a=float(env.a), soc=float(env.soc))
            _, r, d, info = env.step(cmd); ds = env.x - pre['x']; cap(ds, DT)
            lay = env.layer_log[-1]; tr = env.trace[-1]; diag = dict(memory.update_diagnostic)
            jerk = (env.a - pre['a']) / DT
            if lay['fallback']:
                if pre['v'] < .5: fallback_creep += 1
                else: fallback_move += 1
            sub.append(dict(k=len(sub), decision=n_dec, substep=j, pre=pre, command_u=u, command_a=float(cmd), applied_a=float(env.a),
                            interval=[lay['lower'], lay['upper']], fallback=bool(lay['fallback']), intervened=bool(lay['intervened']), targets=lay['targets'], v_limit=lay['v_limit'],
                            perceived_pre=lay['perceived'], truth_green_pre=lay['truth_green'], reward=float(r),
                            post=dict(t=float(env.t), x=float(env.x), v=float(env.v), a=float(env.a), soc=float(env.soc)), odom_ds=float(ds), jerk=float(jerk),
                            override_flag=bool(tr[17]), numeric_exceedance=bool(abs(jerk) > J + 1e-7),
                            judge=dict(red_crossing=bool(info['red_crossing']), arrived=bool(info['arrived']), offroad=bool(info['offroad']), deadline=bool(info['deadline']), truth_green_post=env._truth_green_for_log()),
                            dets=perception.last['dets'], probs=perception.last['probs'], update=diag, memory=memory.snapshot(), margins=margins(diag, env.v, env.a)))
            overshoot = env.x > S.R.LENGTH + 100.; settled_now = info['arrived'] and env.v <= .05 and abs(env.a) <= .05
            done = bool(info['red_crossing'] or settled_now or env.t >= 600 or overshoot)
            if done: break
        n_dec += 1
    m = S.episode_metrics(env); h_after = model_hash(learner)
    m.update(condition_id=i, settled=bool(info['arrived'] and env.v <= .05 and abs(env.a) <= .05), overshoot=bool(env.x > S.R.LENGTH + 100.),
             fallback_substeps=int(sum(l['fallback'] for l in env.layer_log)), intervened_substeps=int(sum(l['intervened'] for l in env.layer_log)),
             fallback_creeping_substeps=fallback_creep, fallback_noncreeping_substeps=fallback_move,
             override_episode=bool(m['jerk_override_steps'] > 0), numeric_exceedance_steps=int(sum(s['numeric_exceedance'] for s in sub)),
             guard_rejections=int(sum(1 for s in sub if s['update'].get('range_guard'))), offroad_substeps=int(env.n_offroad), curve_excess_penalty=float(env.e_curve_pen),
             decisions=n_dec, substeps=len(sub), wall_s=time.monotonic() - t_wall, model_hash_before=h_before, model_hash_after=h_after, model_unchanged=h_before == h_after,
             fingerprint=fingerprint, job_id=job['job_id'], profile=job['profile'], profile_identity=profile_identity(job['profile']), policy=job['policy'],
             camera_appearance_seed=app_seed, camera_noise_seed=noise_seed, episode_id=eid, training_steps=0)
    out.mkdir(parents=True, exist_ok=True)
    S.save_trace(out / 'trace.npz', env.trace)
    for name, obj in (('layer.json', env.layer_log), ('summary.json', m)):
        tmp = out / (name + '.tmp'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, out / name)
    tmp = out / 'substeps.json.gz.tmp'
    with gzip.open(tmp, 'wt') as f: json.dump(sub, f)
    os.replace(tmp, out / 'substeps.json.gz'); (out / 'DONE').write_text(fingerprint + '\n')
    return m


def complete(out, fingerprint):
    if not all((out / n).exists() for n in ('trace.npz', 'layer.json', 'substeps.json.gz', 'summary.json', 'DONE')): return False
    return (out / 'DONE').read_text().strip() == fingerprint and json.load(open(out / 'summary.json')).get('fingerprint') == fingerprint


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', default=str(HERE / 'jobs.csv')); ap.add_argument('--select', default='R1,R2', help='experiments and/or job_id substrings, comma separated')
    ap.add_argument('--policy', default=None, help='restrict to a policy label (S0/S1/S2/constant_u_plus_1)'); ap.add_argument('--out', default=str(HERE / 'results')); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); lock = json.load(open(HERE / 'protocol_lock.json')); fp = lock['fingerprint']
    import csv; jobs = [r for r in csv.DictReader(open(a.jobs)) if r['experiment'] in ('R1', 'R2')]
    sel = a.select.split(','); jobs = [r for r in jobs if r['experiment'] in sel or any(s in r['job_id'] for s in sel)]
    if a.policy: jobs = [r for r in jobs if r['policy'] == a.policy]
    cache = {}; n_done = 0
    for job in jobs:
        out = Path(a.out) / job['experiment'] / job['job_id']
        if complete(out, fp): print('skip (complete)', job['job_id'], flush=True); continue
        if a.limit and n_done >= a.limit: break
        seed_index = 0 if job['policy'] == 'constant_u_plus_1' else int(job['policy_seed_index'])
        if seed_index not in cache:
            learner, cfg, ck = load_policy(seed_index); per, det = make_perception(cfg); cache[seed_index] = (learner, cfg, per)
        learner, cfg, per = cache[seed_index]
        if job['policy'] == 'constant_u_plus_1': learner = ConstantLearner(learner, 1.0)
        m = run_episode(job, learner, cfg, per, out, fp, int(cfg.get('extra_dim', 6))); n_done += 1
        print(json.dumps({k: m[k] for k in ('job_id', 'settled', 'violations', 'offroad_substeps', 'jerk_override_steps', 'fallback_substeps', 'guard_rejections', 'Ij', 'time_s', 'E_Wh', 'R', 'wall_s', 'model_unchanged')}), flush=True)


if __name__ == '__main__': main()
