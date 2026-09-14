"""R3 runner: P-T timing-use experiment on the 18 prespecified reporting conditions (extension conditions).

Reuses timing/code/probe.py unchanged (ProbeEnv, execute, rollout with decision-boundary resume snapshots,
extra_metrics, frozen source hash) and the frozen actor timing/weights/actor_A.pt. The archived `reporting`
stage of probe.py stays gated by its failed pilot gate; this entry point is registered separately as the
'extension-condition evaluation' authorized by validation_frozen/protocol_lock.json and writes only under
validation_frozen/results/R3. Both arms keep the archived P-T terminal branch and settle criteria.
"""
import argparse, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parent; TIM = REPO / 'timing'
sys.path.insert(0, str(TIM / 'code'))
import numpy as np, torch
import probe as P
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(HERE / 'results' / 'R3')); ap.add_argument('--jobs', default=str(HERE / 'jobs.csv')); a = ap.parse_args()
    lock = json.load(open(HERE / 'protocol_lock.json')); fp = P.source_hash()
    freeze = json.loads((P.ROOT / 'audit' / 'freeze.json').read_text()); assert freeze['source_hash'] == fp, 'frozen P-T code/protocol/weights changed'
    assert lock['R3']['pt_source_hash'] == fp and json.loads((P.ROOT / 'audit' / 'audit.json').read_text())['passed']
    assert not (P.ROOT / 'runs' / 'reporting').exists(), 'archived reporting outputs exist; prior usage must be re-checked'
    conds = P.S.conditions('reporting'); assert len(conds) == 18
    import csv; jobs = [r for r in csv.DictReader(open(a.jobs)) if r['experiment'] == 'R3']
    for r in jobs:
        c = conds[int(r['condition_id'])]; assert (float(r['soc']), float(r['temperature_K']), float(r['initial_speed_mps']), float(r['signal_offset_s'])) == tuple(map(float, c)), r['job_id']
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); pf = out / 'progress.json'
    progress = json.loads(pf.read_text()) if pf.exists() else dict(source_hash=fp, rows=[])
    assert progress['source_hash'] == fp
    completed = {(r['mode'], r['condition_id']) for r in progress['rows']}
    actor = P.load_actor(); start = time.monotonic()
    for r in jobs:
        i = int(r['condition_id']); mode = r['timing_mode']
        if (mode, i) in completed: print('skip (complete)', r['job_id'], flush=True); continue
        stem = f'{mode}_{i:02d}'; cp = out / (stem + '.resume.pkl')
        m, env = P.rollout(actor, conds[i], mode, cp, fp)
        m.update(mode=mode, condition_id=i, condition=list(conds[i]), job_id=r['job_id'], fingerprint=lock['fingerprint'], training_steps=0)
        P.S.save_trace(out / (stem + '.npz'), env.trace); P.S.json_save(out / (stem + '_layer.json'), env.layer_log)
        progress['rows'].append(m); P.S.json_save(pf, progress); cp.unlink(missing_ok=True)
        print(json.dumps({k: m[k] for k in ('job_id', 'settled', 'signal_Ij', 'signal_window_complete', 'Ij', 'time_s', 'E_Wh', 'violations', 'local_overspeed_time_s', 'layer_fallback_steps', 'paired_action_difference_steps', 'wall_s')}), flush=True)
    P.S.json_save(out / 'runtime.json', dict(complete=len(progress['rows']) == 36, wall_s=time.monotonic() - start, source_hash=fp))


if __name__ == '__main__': main()
