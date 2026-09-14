"""P-T (executor timing-use intervention): check main Table IV and Supplementary Table S5 against the archived pilot records.

Reads reports/pilot.json and reports/pilot_pairs.csv (written by code/analyze.py pilot from runs/development/*_layer.json).
No vehicle rollout; NumPy only.  `python code/audit.py` checks the frozen source/weight identity and
`python code/analyze.py pilot` regenerates the two report files from the raw records.
"""
import csv, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PAPER_IV = dict(color=dict(settled=8, violations=1, fallback=4, peak=7.738), timing=dict(settled=9, violations=0, fallback=0, peak=2.000))
PAPER_S5 = [(5.516, 5.516), (2.270, 2.270), (63.518, 63.518), (0.524, 0.524), (24.647, 24.647), (33.555, 33.555), (4.113, 4.113), (46.300, 36.940), (8.636, 8.636)]
EVENT = dict(t=177.5, x=2914.128, color_action=0.408, timing_action=0.160, timing_steps=1498, tests=466, action_diff=5)

ok = True
def check(name, got, exp, tol=5e-4):
    global ok; good = abs(got - exp) <= tol; ok &= good
    print('  %-44s %10.3f  paper %10.3f  %s' % (name, got, exp, 'OK' if good else 'MISMATCH'))

d = json.load(open(HERE / 'reports/pilot.json')); agg = d['aggregate']
print('Table IV')
for arm in ('color', 'timing'):
    a = agg[arm]; e = PAPER_IV[arm]
    check(f'{arm}: settled trips', a['settled'], e['settled'], 0); check(f'{arm}: signal violations', a['totals']['violations'], e['violations'], 0)
    check(f'{arm}: fallback substeps', a['totals']['layer_fallback_steps'], e['fallback'], 0); check(f'{arm}: peak jerk', a['peak_jerk'], e['peak'])
check('timing: countdown-visible substeps', agg['timing']['totals']['countdown_visible_steps'], EVENT['timing_steps'], 0)
check('timing: green-end tests', agg['timing']['totals']['timing_test_steps'], EVENT['tests'], 0)
check('paired action differences (substeps)', agg['timing']['totals']['paired_action_difference_steps'], EVENT['action_diff'], 0)
print('Table S5 (signal-window squared jerk 2200-3300 m)')
rows = list(csv.DictReader(open(HERE / 'reports/pilot_pairs.csv'))); assert len(rows) == 9
for r, (c, t) in zip(rows, PAPER_S5):
    i = r['condition_id']; check(f'condition {i}: current color', float(r['color_signal_Ij']), c); check(f'condition {i}: color + timing', float(r['timing_signal_Ij']), t)
    identical = r['first_divergence'] == ''
    check(f'condition {i}: identical trajectories', 1.0 if identical else 0.0, 0.0 if i == '7' else 1.0, 0)
ev = eval(rows[7]['first_divergence'])
print('Condition 7 first difference')
check('time', ev['t'], EVENT['t'], 1e-9); check('position', ev['x'], EVENT['x'], 5e-4); check('current-color action', ev['color_action'], EVENT['color_action']); check('timing action', ev['timing_action'], EVENT['timing_action'])
json.dump(dict(checks_pass=ok, source_hash=d['source_hash'], stage=d['stage']), open(HERE / 'reports/table_checks.json', 'w'), indent=1)
print('TABLE CHECKS', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
