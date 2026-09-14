"""Baseline reproduction check: every R1 paper_v25 episode against the archived S evaluation records (trace and metrics), exact."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; VIS = HERE.parent / 'visual'
RUNS = {'S0': 'v4r', 'S1': 'v4r_s1', 'S2': 'v4r_s2'}; out = dict(episodes=[], all_identical=True)
for pol, run in RUNS.items():
    arch = json.load(open(VIS / 'runs' / run / 'v4_pilot' / 'supervised' / 'evaluation.json'))['rows']
    for c in range(9):
        d = HERE / 'results' / 'R1' / f'R1_{pol}_c{c:02d}_paper_v25'
        new = np.load(d / 'trace.npz')['values']; old = np.load(VIS / 'runs' / run / 'v4_pilot' / 'supervised' / 'evaluation_traces' / f'dev_{c:02d}.npz')['values']
        s = json.load(open(d / 'summary.json')); a = arch[c]
        diff = float(np.max(np.abs(new - old))) if new.shape == old.shape else float('inf')
        m = {k: (s[k], a[k]) for k in ('R', 'Ij', 'time_s', 'E_Wh', 'jerk_max', 'jerk_override_steps', 'violations', 'settled', 'fallback_substeps')}
        same = diff == 0. and all(x == y for x, y in m.values())
        out['episodes'].append(dict(policy=pol, condition_id=c, trace_shape=list(new.shape), max_abs_trace_diff=diff, metrics_identical=all(x == y for x, y in m.values()), identical=same))
        out['all_identical'] &= same
out['archived_totals'] = dict(settled='27/27', violations=0, override_episodes='9/27')
(HERE / 'baseline_check.json').write_text(json.dumps(out, indent=1) + '\n'); print('baseline reproduction exact for all 27:', out['all_identical'])
