"""P-V: regenerate the visual outcome tables from the archived evaluation records and check them against the paper.

Inputs (no network inference, no vehicle stepping):
  runs/{v4r,v4r_s1,v4r_s2}/v4_pilot/{frozen,supervised,joint,joint_head}/evaluation.json   nine development conditions per arm
  runs/probes/v4r_probe_closed_loop.json                                                  historical constant-command probe (u=1)
Outputs: main Table V, Supplementary Tables S6 and S7, printed and written to tables.json.
Seed indices 0/1/2 are run directories v4r / v4r_s1 / v4r_s2 (training seeds 7/8/9). F/S/J records use the current
protocol (no true-position terminal override); JH records retain the legacy override and are reported separately.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
RUNS = ['v4r', 'v4r_s1', 'v4r_s2']; ARMS = dict(F='frozen', S='supervised', J='joint', JH='joint_head')

PAPER_V = {'F': (24, 2, 0, -42.780, 225.771, 625.824, 96.311, 12), 'S': (27, 0, 0, -39.423, 223.722, 597.914, 88.135, 9)}
PAPER_PROBE = (9, -39.246, 217.889, 605.974, 100.954)
PAPER_S6 = {(0, 'F'): (9, -40.265, 99.238, 220.556, 628.351), (0, 'S'): (9, -39.147, 86.998, 218.889, 601.001), (0, 'J'): (9, -40.622, 98.563, 222.167, 634.682), (0, 'JH'): (9, -39.081, 92.333, 217.778, 601.643),
            (1, 'F'): (9, -39.970, 86.128, 225.278, 609.665), (1, 'S'): (9, -38.874, 80.026, 217.944, 595.525), (1, 'J'): (9, -39.420, 114.105, 220.556, 604.877), (1, 'JH'): (9, -38.875, 79.829, 217.000, 597.653),
            (2, 'F'): (6, -48.105, 107.194, 234.333, 646.271), (2, 'S'): (9, -40.246, 97.382, 234.333, 597.217), (2, 'J'): (8, -44.263, 149.347, 248.500, 674.051), (2, 'JH'): (7, -44.400, 150.506, 243.571, 642.908)}
PAPER_S7 = {('S', 0): ('0,1,2,3,4,5,6,7,8', 9, -12.240, -1.667, -27.350, 1.118), ('S', 1): ('0,1,2,3,4,5,6,7,8', 9, -6.101, -7.333, -14.140, 1.096), ('S', 2): ('0,1,3,4,6,7', 6, 4.144, 15.167, -41.915, 0.296),
            ('J', 0): ('0,1,2,3,4,5,6,7,8', 9, -0.674, 1.611, 6.331, -0.357), ('J', 1): ('0,1,2,3,4,5,6,7,8', 9, 27.977, -4.722, -4.788, 0.550), ('J', 2): ('1,3,4,6,7', 5, 45.832, 13.900, 9.343, -1.448)}


def rows(seed, arm):
    d = json.load(open(HERE / 'runs' / RUNS[seed] / 'v4_pilot' / ARMS[arm] / 'evaluation.json'))['rows']
    assert len(d) == 9 and [r['condition_id'] for r in d] == list(range(9)); return d


def summary(rs):
    done = [r for r in rs if r['settled']]
    return dict(n=len(rs), settled=len(done), signal=sum(r['violations'] for r in rs), curve=sum(1 for r in rs if r['offroad_substeps'] > 0),
                R=float(np.mean([r['R'] for r in rs])), time=float(np.mean([r['time_s'] for r in done])) if done else None,
                energy=float(np.mean([r['E_Wh'] for r in done])) if done else None, Ij=float(np.mean([r['Ij'] for r in done])) if done else None,
                override_episodes=sum(1 for r in rs if r['jerk_override_steps'] > 0), peak_jerk=max(r['jerk_max'] for r in rs))


def main():
    ok = True
    def check(name, got, exp, tol=5e-4):
        nonlocal ok; good = got is not None and abs(got - exp) <= tol; ok &= good
        print('  %-48s %10.3f  paper %10.3f  %s' % (name, got if got is not None else float('nan'), exp, 'OK' if good else 'MISMATCH'))
    out = {}
    print('Table V (pooled over seeds 0/1/2, nine conditions each)')
    for arm in ('F', 'S'):
        s = summary([r for seed in range(3) for r in rows(seed, arm)]); out['tableV_' + arm] = s; e = PAPER_V[arm]
        check(f'{arm}: settled', s['settled'], e[0], 0); check(f'{arm}: signal violations', s['signal'], e[1], 0); check(f'{arm}: curve violations', s['curve'], e[2], 0)
        check(f'{arm}: mean R (all 27)', s['R'], e[3]); check(f'{arm}: time (completed)', s['time'], e[4]); check(f'{arm}: energy (completed)', s['energy'], e[5]); check(f'{arm}: I_j (completed)', s['Ij'], e[6]); check(f'{arm}: override episodes', s['override_episodes'], e[7], 0)
    p = json.load(open(HERE / 'runs/probes/v4r_probe_closed_loop.json'))['vision']; out['tableV_constant_probe'] = {k: p[k] for k in ('settled', 'violations', 'offroad', 'R', 'time_s', 'E_Wh', 'Ij', 'max_jerk')}
    check('constant u=1 probe: settled', p['settled'], PAPER_PROBE[0], 0); check('constant u=1 probe: mean R', p['R'], PAPER_PROBE[1]); check('constant u=1 probe: time', p['time_s'], PAPER_PROBE[2]); check('constant u=1 probe: energy', p['E_Wh'], PAPER_PROBE[3]); check('constant u=1 probe: I_j', p['Ij'], PAPER_PROBE[4])
    print('Table S6 (every seed and arm)'); out['tableS6'] = {}
    for seed in range(3):
        for arm in ('F', 'S', 'J', 'JH'):
            s = summary(rows(seed, arm)); out['tableS6'][f'{seed}/{arm}'] = s; e = PAPER_S6[(seed, arm)]
            check(f'seed {seed} {arm}: completed', s['settled'], e[0], 0); check(f'seed {seed} {arm}: mean R', s['R'], e[1]); check(f'seed {seed} {arm}: I_j', s['Ij'], e[2]); check(f'seed {seed} {arm}: time', s['time'], e[3]); check(f'seed {seed} {arm}: energy', s['energy'], e[4])
    print('Table S7 (common-success contrasts relative to F, current protocol)'); out['tableS7'] = {}
    for arm in ('S', 'J'):
        for seed in range(3):
            f = {r['condition_id']: r for r in rows(seed, 'F') if r['settled']}; g = {r['condition_id']: r for r in rows(seed, arm) if r['settled']}
            ids = sorted(set(f) & set(g)); e = PAPER_S7[(arm, seed)]
            d = {k: float(np.mean([g[i][k] - f[i][k] for i in ids])) for k in ('Ij', 'time_s', 'E_Wh', 'R')}; out['tableS7'][f'{arm}-F seed {seed}'] = dict(ids=ids, n=len(ids), **d)
            check(f'{arm}-F seed {seed}: condition ids', 0 if ','.join(map(str, ids)) == e[0] else 1, 0, 0); check(f'{arm}-F seed {seed}: n', len(ids), e[1], 0)
            check(f'{arm}-F seed {seed}: delta I_j', d['Ij'], e[2]); check(f'{arm}-F seed {seed}: delta time', d['time_s'], e[3]); check(f'{arm}-F seed {seed}: delta energy', d['E_Wh'], e[4]); check(f'{arm}-F seed {seed}: delta R', d['R'], e[5])
    out['checks_pass'] = ok; (HERE / 'tables.json').write_text(json.dumps(out, indent=1) + '\n')
    print('TABLE CHECKS', 'PASS' if ok else 'FAIL'); return ok


if __name__ == '__main__': sys.exit(0 if main() else 1)
