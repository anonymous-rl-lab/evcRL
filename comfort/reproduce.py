"""P-C (fixed-actor complete-executor comparison): recompute the paper's numbers from the archived traces.

Record mode (default) needs NumPy only. It recomputes main Table III, Supplementary Tables S3/S4 and the
original-arrival-cut secondary numbers (77.861 -> 61.879, 49.083 -> 32.205) from
  reports/settled_traces/{A,B}_xx.npz   original executor, terminal settling appended (baseline)
  runs/frozen_s7_r2/{A,B}_xx.npz        complete executor (r2)
and checks them against the values printed in the manuscript.

--rollout additionally re-runs both executors with the extracted frozen actors (frozen_actor_replay/actor_{A,B}.pt)
through the archived code (deps/study.py, code/curve_layer.py, code/rollout_layer.py) and compares the regenerated
trajectories with the archived traces. It needs PyTorch (CPU).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code')); sys.path.insert(0, str(HERE / 'deps'))
import summarize   # archived metric definitions (Ij, RMS, peaks, settled, overspeed); importing has no side effects

PAPER = {   # main Table III / Table S4 / secondary cut, as printed
    'A': dict(orig=dict(Ij=86.776, peak=6.483, time=270.889, E=450.821, arms=0.643, brake=3.50, below=6.50), comp=dict(Ij=63.781, peak=2.000, time=271.000, E=446.432, arms=0.618, brake=3.50, below=6.94), cut=(77.861, 61.879, 20.53)),
    'B': dict(orig=dict(Ij=60.690, peak=6.809, time=303.167, E=363.387, arms=0.514, brake=3.50, below=5.33), comp=dict(Ij=34.249, peak=2.000, time=303.556, E=360.970, arms=0.490, brake=3.50, below=5.50), cut=(49.083, 32.205, 34.39))}
S3 = {'A': [(22.22, 0.5, -0.89), (27.90, 0.0, -1.37), (28.48, 0.5, -0.27), (20.92, 0.5, -0.77), (21.80, -1.0, -1.21), (17.60, 1.0, -1.28), (28.27, -1.0, -0.01), (36.63, 1.0, -1.97), (30.84, -0.5, -0.72)],
      'B': [(48.59, -0.5, -0.14), (43.30, 0.5, -0.32), (41.71, 0.5, -1.53), (52.88, -0.5, -1.02), (45.17, 0.5, -1.47), (9.80, 2.5, 1.30), (53.96, -0.5, -0.88), (47.06, 0.5, -0.79), (40.19, 0.5, -0.99)]}


def load(p): return np.load(p, allow_pickle=False)['values']


def accel_diag(a):
    dt = a[:, 1] - a[:, 0]; acc = a[:, 8]
    return dict(arms=float(np.sqrt(np.sum(acc ** 2 * dt) / dt.sum())), brake=float(-acc.min()), below=float(dt[acc < -2.].sum()))


def record_mode():
    ok = True; out = {}
    def check(name, got, exp, tol):
        nonlocal ok; good = abs(got - exp) <= tol; ok &= good
        print('  %-42s %12.3f  paper %10.3f  %s' % (name, got, exp, 'OK' if good else 'MISMATCH'))
    for arm in 'AB':
        orig = [summarize.metrics(load(HERE / 'reports/settled_traces' / f'{arm}_{i:02d}.npz')) for i in range(9)]
        comp = [summarize.metrics(load(HERE / 'runs/frozen_s7_r2' / f'{arm}_{i:02d}.npz')) for i in range(9)]
        do = [accel_diag(load(HERE / 'reports/settled_traces' / f'{arm}_{i:02d}.npz')) for i in range(9)]
        dc = [accel_diag(load(HERE / 'runs/frozen_s7_r2' / f'{arm}_{i:02d}.npz')) for i in range(9)]
        print(f'Actor {arm}: Table III (nine settled development trips)')
        for lab, rows, diag, key in (('original', orig, do, 'orig'), ('complete', comp, dc, 'comp')):
            exp = PAPER[arm][key]
            check(f'{lab}: mean I_j', np.mean([r['Ij'] for r in rows]), exp['Ij'], 5e-4)
            check(f'{lab}: peak jerk', max(r['jerk_max'] for r in rows), exp['peak'], 5e-4)
            check(f'{lab}: mean time', np.mean([r['time_s'] for r in rows]), exp['time'], 5e-4)
            check(f'{lab}: mean energy', np.mean([r['E_Wh'] for r in rows]), exp['E'], 5e-4)
            check(f'{lab}: settled trips', sum(r['settled'] for r in rows), 9, 0)
            check(f'{lab}: acceleration RMS (S4)', np.mean([d['arms'] for d in diag]), exp['arms'], 5e-4)
            check(f'{lab}: largest braking (S4)', max(d['brake'] for d in diag), exp['brake'], 5e-3)
            check(f'{lab}: time below -2 m/s2 (S4)', np.mean([d['below'] for d in diag]), exp['below'], 5e-3)
        print(f'Actor {arm}: Table S3 (paired condition effects)')
        for i, (o, c) in enumerate(zip(orig, comp)):
            red = 100 * (1 - c['Ij'] / o['Ij']); dtm = c['time_s'] - o['time_s']; de = 100 * (c['E_Wh'] / o['E_Wh'] - 1)
            e = S3[arm][i]; check(f'condition {i}: I_j reduction %', red, e[0], 5e-3); check(f'condition {i}: time change s', dtm, e[1], 1e-9); check(f'condition {i}: energy change %', de, e[2], 5e-3)
        cut = []
        for i in range(9):
            a = load(HERE / 'runs/frozen_s7_r2' / f'{arm}_{i:02d}.npz'); k = int(np.flatnonzero(a[:, 22])[0]); cut.append(summarize.metrics(a[:k + 1]))
        archived = json.load(open(HERE / 'actor_evaluations' / f'{arm}_step_300000_development.json'))['summary']['completed_mean']['Ij']
        print(f'Actor {arm}: original arrival cut (secondary)')
        check('original executor (archived evaluation)', archived, PAPER[arm]['cut'][0], 5e-4)
        check('complete executor, cut at arrival', np.mean([r['Ij'] for r in cut]), PAPER[arm]['cut'][1], 5e-4)
        check('reduction %', 100 * (1 - np.mean([r['Ij'] for r in cut]) / archived), PAPER[arm]['cut'][2], 5e-3)
        out[arm] = dict(original=[{k: (bool(r[k]) if k == 'settled' else float(r[k])) for k in ('Ij', 'jerk_max', 'time_s', 'E_Wh', 'settled')} for r in orig], complete=[{k: (bool(r[k]) if k == 'settled' else float(r[k])) for k in ('Ij', 'jerk_max', 'time_s', 'E_Wh', 'settled')} for r in comp])
    print('RECORD CHECKS', 'PASS' if ok else 'FAIL'); return ok, out


def rollout_mode():
    import torch; torch.set_num_threads(1)
    import study as S, rollout_layer as RL
    ok = True; report = {}
    for arm in 'AB':
        ck = torch.load(HERE / 'frozen_actor_replay' / f'actor_{arm}.pt', map_location='cpu', weights_only=False)
        actor = S.MLP(13, True); actor.load_state_dict(ck['actor']); actor.eval()
        for i, cond in enumerate(S.conditions('development')):
            m, env = RL.rollout(actor, cond); got = np.asarray(env.trace, float)
            ref = load(HERE / 'runs/frozen_s7_r2' / f'{arm}_{i:02d}.npz'); d_c = float(np.max(np.abs(got - ref))) if got.shape == ref.shape else float('inf')
            soc, temp, v0, off = cond; env0 = S.StudyEnv(soc, temp, off); o = env0.reset(v0)   # original executor (archived study.rollout loop)
            while True:
                with torch.no_grad(): u = S.command(float(actor(torch.as_tensor(o).unsqueeze(0))[0, 0]))
                for _ in range(4):
                    o, r, done, info = env0.step(u)
                    if done: break
                if done: break
            while (env0.v > 1e-9 or abs(env0.a) > 1e-9) and len(env0.trace) < 5000: env0.step(0.)   # terminal settling as in code/diagnose.py
            got0 = np.asarray(env0.trace, float); ref0 = load(HERE / 'reports/settled_traces' / f'{arm}_{i:02d}.npz'); d_o = float(np.max(np.abs(got0 - ref0))) if got0.shape == ref0.shape else float('inf')
            print(f'{arm} condition {i}: complete executor max|diff| {d_c:.2e} ({got.shape[0]} substeps) | original+settling max|diff| {d_o:.2e} ({got0.shape[0]} substeps)')
            # The archived settled traces were produced by code/diagnose.py, which restores the terminal state from the saved
            # trace rows before appending the settling tail; the live environment differs from that reconstruction at 1e-10.
            ok &= d_c == 0. and d_o <= 1e-8; report[f'{arm}_{i}'] = dict(complete=d_c, original=d_o)
    print('ROLLOUT CHECKS', 'PASS (complete executor bit-identical; original executor within 1e-8 of the archived settled traces)' if ok else 'FAIL', json.dumps({'torch': torch.__version__, 'numpy': np.__version__}))
    return ok, report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rollout', action='store_true'); ap.add_argument('--out', default=None); a = ap.parse_args()
    ok, out = record_mode(); res = dict(record_checks=bool(ok), per_condition=out)
    if a.rollout:
        import torch; ok2, rep = rollout_mode(); res.update(rollout_checks=bool(ok2), rollout_max_abs_diff=rep, runtime=dict(torch=torch.__version__, numpy=np.__version__))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    sys.exit(0 if all(v for k, v in res.items() if k.endswith('checks')) else 1)
