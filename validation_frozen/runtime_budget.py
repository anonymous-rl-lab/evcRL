"""Measured runtime per episode class and the resulting estimate; written to runtime_budget.json."""
import csv, json, os
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
eps = list(csv.DictReader(open(HERE / 'episodes.csv')))
w1 = [float(e['wall_s']) for e in eps if e['experiment'] == 'R1']; w2 = [float(e['wall_s']) for e in eps if e['experiment'] == 'R2']
r3 = json.load(open(HERE / 'results' / 'R3' / 'progress.json'))['rows']; w3 = [float(r['wall_s']) for r in r3]
lay = sum(os.path.getsize(p) for p in (HERE / 'results').rglob('*') if p.is_file())
out = dict(machine=dict(cpu_count=os.cpu_count(), gpu='none', price='unknown (no billing rate available for this container)', external_compute_charge=0),
           smoke=dict(episodes=['R1_S0_c00_paper_v25', 'R1_S0_c00_guard_v26'], wall_s_budget=600, note='complete formal-protocol episodes; reused in the formal queue'),
           pilot_validation=dict(cumulative_wall_s_budget=3600, used='the formal queue itself (single process); no separate timing runs'),
           measured=dict(R1=dict(n=len(w1), mean_wall_s=float(np.mean(w1)), min=min(w1), max=max(w1), total_s=sum(w1)), R2=dict(n=len(w2), mean_wall_s=float(np.mean(w2)), total_s=sum(w2)),
                         R3=dict(n=len(w3), mean_wall_s=float(np.mean(w3)), total_s=sum(w3)), results_bytes=lay, per_episode_log_bytes_R1=int(lay / max(1, len(eps)))),
           formal_estimate=dict(R1_54=54 * float(np.mean(w1)), R2_9=9 * float(np.mean(w2)), R3_36=36 * float(np.mean(w3)), total_s=54 * float(np.mean(w1)) + 9 * float(np.mean(w2)) + 36 * float(np.mean(w3))),
           actual=dict(all_99_completed=len(eps) == 63 and len(r3) == 36, single_process_sequential=True, memory='< 2 GB resident per process (frame store keeps 4 frames)'),
           resume=dict(unit='episode (all four raw outputs + DONE flag with protocol fingerprint); R3 additionally resumes inside an episode from decision-boundary snapshots (probe.py)',
                       command='cd validation_frozen && OMP_NUM_THREADS=1 python3 run_pv.py --select R1,R2 && OMP_NUM_THREADS=1 python3 run_pt.py && python3 metrics.py'))
(HERE / 'runtime_budget.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out['measured'], indent=1)); print(out['formal_estimate'])
