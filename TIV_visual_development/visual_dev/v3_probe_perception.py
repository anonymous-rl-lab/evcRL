"""v3 门 2 探针：给定编码器（执行层感知源）与一个已训练 actor，在 9 个开发工况上比较
truth / perceived(oracle) / perceived(ROI 头, 阈值+连续帧) 三种执行层相位来源的闭环结果，并给出 ROI 头按距离分层的闭环识别质量。
用法：python visual_dev/v3_probe_perception.py --encoder runs/pretrain_v3/encoder.pt --actor runs/pilot_v2/frozen/final_nets.pt [--gate]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P, study as S
from pipeline import Perception, evaluate
from reevaluate import learner_from_final_nets
from renderer import signal_color
torch.set_num_threads(1)
BINS = ((0, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 450))


def by_distance(rows, traces):
    recs = []
    for r, (tr, log) in zip(rows, traces):
        for l in log:
            d = l.get('perceived_detail')
            if d and d.get('visible'): recs.append(dict(dist=3000. - l['x'], truth=signal_color(l['t'], r['offset_s']), raw=d['raw'], p_green=d['p_green']))
    out = {}
    for lo, hi in BINS:
        b = [x for x in recs if lo <= x['dist'] < hi]; g = [x for x in b if x['truth'] == 'green']; y = [x for x in b if x['truth'] == 'yellow']; rd = [x for x in b if x['truth'] == 'red']
        f = lambda L: (float(np.mean([x['raw'] == 'green' for x in L])) if L else None)
        out[f'{lo}-{hi}m'] = dict(n_green=len(g), green_recall=f(g), green_p_median=(float(np.median([x['p_green'] for x in g])) if g else None), n_yellow=len(y), yellow_to_green=f(y), n_red=len(rd), red_to_green=f(rd))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--encoder', default='runs/pretrain_v3/encoder.pt'); ap.add_argument('--actor', default='runs/pilot_v2/frozen/final_nets.pt')
    ap.add_argument('--green-threshold', type=float, default=0.9); ap.add_argument('--min-consecutive', type=int, default=2); ap.add_argument('--gate', action='store_true'); ap.add_argument('--out', default='runs/probes/v3_probe_perception.json')
    a = ap.parse_args(); L, _ = learner_from_final_nets(ROOT / a.actor); res = {}
    variants = [('truth', 'truth', None), ('perceived_oracle', 'perceived', Perception(source='oracle')),
                ('perceived_roi_raw', 'perceived', Perception(ROOT / a.encoder, 'roi_head')),
                (f'perceived_roi_thr{a.green_threshold}_k{a.min_consecutive}', 'perceived', Perception(ROOT / a.encoder, 'roi_head', a.green_threshold, a.min_consecutive))]
    for name, src, per in variants:
        t0 = time.time(); rows, vis, traces = evaluate(L, S.conditions('development'), signal_source=src, perception=per); sm = S.summarize(rows); cm = sm['completed_mean'] or {}
        res[name] = dict(summary=sm, arrived=int(sum(r['arrived'] for r in rows)), settled=int(sum(r['settled'] for r in rows)), violations=int(sm['violations']), fallback=int(sum(r['fallback_substeps'] for r in rows)),
                         per_condition=[dict(id=r['condition_id'], violations=r['violations'], fallback=r['fallback_substeps'], Ij=r['Ij'], time_s=r['time_s'], E_Wh=r['E_Wh'], R=r['R'], arrived=r['arrived']) for r in rows],
                         false_green_substeps=int(sum(r['perceived_green_truth_red_substeps'] for r in rows)), conservative_substeps=int(sum(r['perceived_nongreen_truth_green_substeps'] for r in rows)),
                         unknown_substeps=int(sum(r['perceived_counts']['unknown'] for r in rows)), by_distance=(by_distance(rows, traces) if per is not None and per.source == 'roi_head' else None))
        print(f"[{name}] 到达 {res[name]['arrived']}/9 静止 {res[name]['settled']} 违规 {res[name]['violations']} 回退子步 {res[name]['fallback']} 最坏jerk {sm['max_jerk']:.2f} | I_j {cm.get('Ij', float('nan')):.2f} 时间 {cm.get('time_s', float('nan')):.1f} 能耗 {cm.get('E_Wh', float('nan')):.1f} R {cm.get('R', float('nan')):.3f} | 感知绿/真值红 {res[name]['false_green_substeps']} 感知非绿/真值绿 {res[name]['conservative_substeps']} unknown {res[name]['unknown_substeps']} ({time.time() - t0:.0f}s)", flush=True)
        if res[name]['by_distance']:
            for k, v in res[name]['by_distance'].items(): print(f"    {k:>9}: 绿 n={v['n_green']} 召回 {v['green_recall']} P中位 {None if v['green_p_median'] is None else round(v['green_p_median'], 3)} | 黄 n={v['n_yellow']} →绿 {v['yellow_to_green']} | 红 n={v['n_red']} →绿 {v['red_to_green']}")
    json.dump(res, open(ROOT / a.out, 'w'), indent=1, ensure_ascii=False)
    if a.gate:
        g = res[variants[-1][0]]; ok = g['violations'] == 0 and g['arrived'] == 9 and g['false_green_substeps'] <= 2 and g['fallback'] == 0
        print(f"门 2 判据：违规 {g['violations']}==0、到达 {g['arrived']}==9、回退 {g['fallback']}==0、感知绿/真值红 {g['false_green_substeps']}≤2 -> {'通过' if ok else '不通过'}"); raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
