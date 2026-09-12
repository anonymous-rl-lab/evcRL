"""Select checkpoints on validation offsets, report only matching held-out conditions."""
import argparse
import json
from pathlib import Path
import numpy as np

VAL_OFFSETS = {0.0, 45.0}
REP_OFFSETS = {22.5, 67.5}


def subset(grid, offsets):
    rows = [r for r in grid if float(r['offset']) in offsets]
    if len(rows) != 6 or len({(r['soc'], r['T'], r['offset']) for r in rows}) != 6:
        raise ValueError('Expected six unique pack/offset conditions per split')
    return rows


def score(entry, offsets):
    g = subset(entry['grid'], offsets)
    return float(np.mean([r['R'] for r in g])), sum(bool(r['arr']) for r in g)


def reference_means(refs, rows):
    selected = [refs['grid'][f"{r['soc']}_{r['T']}_{r['offset']}"] for r in rows]
    out = {n: float(np.mean([r['humans'][n]['R'] for r in selected]))
           for n in ('attentive', 'normal', 'distracted')}
    out['dp'] = float(np.mean([r['ref']['R'] for r in selected]))
    return out


def summarize(data):
    curve = data['curve']
    # First checkpoint wins ties. Reporting data never enters checkpoint selection.
    k = max(range(len(curve)), key=lambda j: score(curve[j], VAL_OFFSETS)[0])
    selected, last = curve[k], curve[-1]
    rep, arrivals = score(selected, REP_OFFSETS)
    refs = reference_means(data['refs'], subset(selected['grid'], REP_OFFSETS))
    rr = dict(selected_step=selected['step'], validation_R=score(selected, VAL_OFFSETS)[0],
              reporting_R=rep, reporting_arrivals=arrivals, reporting_n=6,
              last_reporting_R=score(last, REP_OFFSETS)[0],
              last_reporting_arrivals=score(last, REP_OFFSETS)[1],
              last3_reporting_R=float(np.mean([score(x, REP_OFFSETS)[0] for x in curve[-3:]])),
              last3_count=min(3, len(curve)), references=refs,
              total_steps=last['step'], wall_s=last['wall_s'])
    rr['beats_attentive'] = bool(arrivals == 6 and rep > refs['attentive'])
    rr['beats_normal'] = bool(arrivals == 6 and rep > refs['normal'])
    rr['reporting_E_Wh'] = float(np.mean([x['E_Wh'] for x in subset(selected['grid'], REP_OFFSETS)]))
    rr['reporting_t'] = float(np.mean([x['t'] for x in subset(selected['grid'], REP_OFFSETS)]))
    rr['reporting_within_1050s'] = sum(bool(x['arr'] and x['t'] <= 1050.) for x in subset(selected['grid'], REP_OFFSETS))
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(Path(__file__).resolve().parent / 'out'))
    ap.add_argument('--min-step', type=int, default=0)
    ap.add_argument('--include-smoke', action='store_true')
    ap.add_argument('--json', default='')
    a = ap.parse_args(); rows = []
    for path in sorted(Path(a.input).glob('*.json')):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f'SKIP {path.name}: incomplete JSON (a live writer may still be saving it)')
            continue
        if not isinstance(data, dict) or not data.get('curve'): continue
        if not a.include_smoke and any(x in path.stem for x in ('smoke', 'tiny')): continue
        if data['curve'][-1]['step'] < a.min_step: continue
        if data.get('refs', {}).get('schema') != 2:
            print(f'SKIP {path.name}: missing current per-condition references'); continue
        row = summarize(data); row['run'] = path.stem; row['args'] = data['args']
        rows.append(row)
        print(f"{path.stem:24s} selected@{row['selected_step']:8d} REP {row['reporting_R']:9.3f} "
              f"arr {row['reporting_arrivals']}/6 | last {row['last_reporting_R']:9.3f} "
              f"last3 {row['last3_reporting_R']:9.3f} (n={row['last3_count']}) "
              f"| matching attentive {row['references']['attentive']:.3f}")
    if not rows: print('No matching completed evaluation points.')
    # Do not pool different algorithms/hyperparameters as interchangeable seeds.
    if a.json: Path(a.json).write_text(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
