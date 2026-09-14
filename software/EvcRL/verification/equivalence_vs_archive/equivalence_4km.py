"""Numerical equivalence of the packaged environment against the archived research code.

Runs in the research environment (torch present, EVSIM_ROUTE=mini set before importing study):
  * structured mode: evcrl Backend vs env20.Route20 (mini: StudyEnv with t_budget 480; 20 km: Route20)
  * camera mode paper_v25: evcrl EcoDriveEnv+OracleDetector vs pipeline.VisionEnv + vision_state.VisionMemory
    driven by the archived 'oracle' perception rule, same camera seed, same command sequence.
Compares x, v, a, reward per substep and memory snapshots.
"""
import os, sys, json, math, copy
from pathlib import Path
import numpy as np
ROOT = str(Path(__file__).resolve().parents[4] / 'visual')   # repository layout: <repo>/visual
os.environ['EVSIM_ROUTE'] = 'mini'
for p in (ROOT + '/v19_deps/code', ROOT, ROOT + '/visual_dev'): sys.path.insert(0, p)
sys.path.insert(0, sys.argv[1])   # packaged src
import evcrl, numpy as np
import study as S, curve_layer as C, env20 as E
from pipeline import VisionEnv
from vision_state import VisionMemory as OrigMemory
from renderer import SceneCamera as OrigCamera, COLOR_CLASSES

def commands(n, seed):
    rng = np.random.default_rng(seed); return [float(v) for v in np.clip(rng.normal(0.2, 0.6, n), -1, 1)]

def cmp(name, a, b, tol=0.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(np.max(np.abs(a - b))) if a.size else 0.
    ok = a.shape == b.shape and d <= tol
    print('%-40s n=%5d max|diff|=%.3e %s' % (name, a.size, d, 'OK' if ok else 'MISMATCH'))
    return ok

# ---------------------------------------------------------------- structured mini
def structured_mini(seed=0, offset=17., soc=.8, T=280., v0=16.):
    orig = S.StudyEnv(soc, T, offset); orig.reset(v0)
    env = evcrl.EcoDriveEnv(mode='structured', route_length=4000.); env.reset(seed=seed, options=dict(soc=soc, temperature=T, speed=v0, signal_offset=offset))
    A = []; B = []; obsA = []; obsB = []
    for u in commands(400, seed):
        cmd = S.command(u); done = False
        for _ in range(4):
            o, r, d, info = orig.step(cmd); A.append((orig.x, orig.v, orig.a, r, orig.e_batt)); obsA.append(o)
            if d: done = True; break
        o2, r2, t, tr, info2 = env.step(np.array([u], np.float64))
        for q in info2['substeps']: B.append((None, q['v1'], q['applied_a'], q['reward'], None))
        obsB.append(o2)
        # patch x, e_batt from env internals (truth is available to the harness)
        B[-1] = (env.vehicle.x, env.vehicle.v, env.vehicle.a, B[-1][3], env.vehicle.e_batt)
        if done or t or tr:
            print('  ended: orig done=%s at t=%.1f; pkg terminated=%s truncated=%s reason=%s t=%.1f' % (done, orig.t, t, tr, info2['termination_reason'], info2['time_s'])); break
    ok = cmp('structured-mini x/v/a/r final', [A[-1][0], A[-1][1], A[-1][2], A[-1][4]], [B[-1][0], B[-1][1], B[-1][2], B[-1][4]])
    ok &= cmp('structured-mini substep reward', [a[3] for a in A], [b[3] for b in B])
    ok &= cmp('structured-mini substep v', [a[1] for a in A], [b[1] for b in B])
    ok &= cmp('structured-mini substep a', [a[2] for a in A], [b[2] for b in B])
    oa = np.array(obsA)[3::4]; ob = np.array(obsB); mm = min(len(oa), len(ob)); ok &= cmp('structured-mini obs (13-dim, per decision)', oa[:mm], ob[:mm])
    if mm and np.max(np.abs(oa[:mm] - ob[:mm])) > 0: print('   worst channel:', int(np.argmax(np.max(np.abs(oa[:mm] - ob[:mm]), axis=0))))
    return ok

# ---------------------------------------------------------------- camera / oracle
def oracle_dets(labels):
    T = labels['truth']; L = labels
    dets = {'traffic_light': dict(score=float(L.get('visible', 0) == 1), dist_m=float(T['light_m'] or 0.), box=None),
            'curve_sign': dict(score=float(T['curve_sign_m'] is not None and T['curve_sign_m'] > 1.5), dist_m=float(T['curve_sign_m'] or 0.), box=None),
            'end_marker': dict(score=float(T['end_m'] is not None and T['end_m'] > 1.5), dist_m=float(T['end_m'] or 0.), box=None),
            'release_sign': dict(score=float(T['release_sign_m'] is not None and T['release_sign_m'] > 1.5), dist_m=float(T['release_sign_m'] or 0.), box=None)}
    probs = None
    if L.get('visible', 0) == 1:
        probs = np.zeros(5, np.float32); probs[COLOR_CLASSES.index(L['color_truth'])] = 1.
    return dets, probs

def camera_oracle(seed=0, offset=40., soc=.85, T=288.15, v0=16., cam_seed=1234, n=600):
    mem = OrigMemory(); orig = VisionEnv(soc, T, offset, memory=mem); orig.reset(v0)
    cam = OrigCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=cam_seed, world='v4'); cam.new_episode('e'); nframes = [0]
    def cap():
        f = cam.capture(episode_id='e', sim_time=orig.t, pose=dict(x=orig.x, v=orig.v, offsets=orig.offsets))
        dets, probs = oracle_dets(f['labels']); mem.update(dets, probs, orig.v, S.E.DT if nframes[0] > 0 else 0.); nframes[0] += 1
        orig.set_perceived(mem.sig['phase'], mem.sig['age'] if mem.sig['seen'] else 0., 'vision_memory')
        return f
    cap()
    env = evcrl.EcoDriveEnv(mode='camera', profile='paper_v25', detector=evcrl.OracleDetector())
    env.reset(seed=seed, options=dict(soc=soc, temperature=T, speed=v0, signal_offset=offset, camera_seed=cam_seed))
    A = []; B = []; snapsA = []; snapsB = []; rgbA = []; rgbB = []; done = False; ended = None
    for u in commands(n, seed):
        cmd = S.command(u)
        for _ in range(4):
            _, r, d, info = orig.step(cmd); f = cap(); A.append((orig.x, orig.v, orig.a, r, orig.e_batt, orig.layer_log[-1]['applied'], orig.layer_log[-1]['fallback'])); snapsA.append(json.dumps(mem.snapshot(), sort_keys=True)); rgbA.append(f['rgb'])
            overshoot = orig.x > S.R.LENGTH + 100.; settled = info['arrived'] and orig.v <= .05 and abs(orig.a) <= .05
            if info['red_crossing'] or settled or orig.t >= 600 or overshoot: done = True; ended = ('red' if info['red_crossing'] else 'settled' if settled else 'deadline' if orig.t >= 600 else 'overshoot'); break
        o, r2, t, tr, info2 = env.step(np.array([u], np.float64))
        for q in info2['substeps']: B.append((None, q['v1'], q['applied_a'], q['reward'], None, q['applied_a'], q['fallback']))
        B[-1] = (env.vehicle.x, env.vehicle.v, env.vehicle.a, B[-1][3], env.vehicle.e_batt, B[-1][5], B[-1][6]); snapsB.append(json.dumps(env.memory.snapshot(), sort_keys=True)); rgbB.append(o['rgb'])
        if done or t or tr:
            print('  ended: orig=%s t=%.1f x=%.1f | pkg reason=%s t=%.1f x=%.1f' % (ended, orig.t, orig.x, info2['termination_reason'], info2['time_s'], env.vehicle.x)); break
    m = min(len(A), len(B))
    ok = len(A) == len(B); print('  substeps orig=%d pkg=%d' % (len(A), len(B)))
    for i, nm in enumerate(('x', 'v', 'a', 'reward', 'e_batt', 'applied', 'fallback')):
        if nm in ('x', 'e_batt'): ok &= cmp('camera-oracle %s (decision boundaries)' % nm, [a[i] for a in A[3::4][:len([b for b in B if b[0] is not None])]], [b[i] for b in B if b[0] is not None][:len(A[3::4])])
        else: ok &= cmp('camera-oracle %s' % nm, [a[i] for a in A[:m]], [b[i] for b in B[:m]])
    same_snap = sum(a == b for a, b in zip(snapsA, snapsB[:len(snapsA)] if False else snapsB)); print('  memory snapshots identical at decision boundaries: checking last'); ok &= snapsA[-1] == snapsB[-1] if A and B else ok
    ok &= cmp('camera-oracle final rgb frame', rgbA[-1].astype(float), rgbB[-1].astype(float))
    return ok

if __name__ == '__main__':
    results = {}
    results['structured_mini'] = structured_mini()
    for off in (0., 40., 70.):
        results['camera_oracle_off%d' % int(off)] = camera_oracle(offset=off)
    print(json.dumps(results, indent=1)); sys.exit(0 if all(results.values()) else 1)
