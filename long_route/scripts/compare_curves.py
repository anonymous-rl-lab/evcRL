"""把本次重训曲线与归档八种子曲线逐检查点比较。

用法：python scripts/compare_curves.py [results/train/repro_s0.json ...]
输出每个种子在新旧两次运行中的验证集/报告集均值、完赛数与 critic 偏差，
按论文协议（analyze.py 规则：仅用验证偏移 0/45 选检查点）给出新运行的报告集分数，
并判断是否超过工况匹配的 attentive 驾驶员基线与 18.5 m/s 定速规则。
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as P
sys.path.insert(0, str(P.EVSIM))
from analyze import score, reference_means, subset, VAL_OFFSETS, REP_OFFSETS

rule = json.load(open(P.REFERENCE / 'cruise_baseline.json'))['reporting']['R']
paths = [Path(p) for p in sys.argv[1:]] or sorted((P.RESULTS / 'train').glob('*_s*.json'))
paths = [p for p in paths if not p.name.startswith('analyze')]
summary = {}
for path in paths:
    new = json.load(open(path)); seed = new['args']['seed']
    old = json.load(open(P.REFERENCE / 'bench_curves' / f'bench_s{seed}.json'))
    print(f"\n### 种子 {seed}：{path.name}（本次）对比 bench_s{seed}.json（归档）")
    print(f"{'序号':>4} | {'本次步数':>8} {'验证R':>8} {'报告R':>8} {'完赛':>3} {'Q偏差':>6} | {'归档步数':>8} {'验证R':>8} {'报告R':>8} {'完赛':>3} {'Q偏差':>6} | {'Δ报告R':>7}")
    for i, c in enumerate(new['curve']):
        nv, _ = score(c, VAL_OFFSETS); nr, na = score(c, REP_OFFSETS)
        line = f"{i:>4} | {c['step']:>8} {nv:8.2f} {nr:8.2f} {na:>3} {c['Q_bias']:+6.1f} |"
        if i < len(old['curve']):
            o = old['curve'][i]; ov, _ = score(o, VAL_OFFSETS); orr, oa = score(o, REP_OFFSETS)
            line += f" {o['step']:>8} {ov:8.2f} {orr:8.2f} {oa:>3} {o['Q_bias']:+6.1f} | {nr-orr:+7.2f}"
        print(line)
    curve = new['curve']
    k = max(range(len(curve)), key=lambda j: score(curve[j], VAL_OFFSETS)[0])
    sel = curve[k]; rep, arr = score(sel, REP_OFFSETS)
    refs = reference_means(new['refs'], subset(sel['grid'], REP_OFFSETS))
    late = [score(c, REP_OFFSETS)[0] for c in curve[-8:]]
    old_res = {r['args']['seed']: r for r in json.load(open(P.REFERENCE / 'bench_results.json'))}[seed]
    summary[f's{seed}'] = dict(steps=curve[-1]['step'], selected_step=sel['step'], reporting_R=rep, reporting_arrivals=arr,
        late8_mean=float(np.mean(late)), late8_sd=float(np.std(late, ddof=1)) if len(late) > 1 else None,
        attentive=refs['attentive'], rule=rule, beats_attentive=bool(arr == 6 and rep > refs['attentive']),
        beats_rule=bool(arr == 6 and rep > rule), archived_selected_step=old_res['selected_step'],
        archived_reporting_R=old_res['reporting_R'], archived_late8_mean=None, wall_s=curve[-1]['wall_s'])
    s = summary[f's{seed}']
    print(f"验证集选中 @{s['selected_step']}：报告集 R {rep:.3f}，完赛 {arr}/6 | 末 8 检查点均值 {s['late8_mean']:.3f} | attentive {refs['attentive']:.3f}，定速规则 {rule:.3f} "
          f"| 归档：选中 @{s['archived_selected_step']}，报告集 R {s['archived_reporting_R']:.3f}")
out = P.RESULTS / 'train' / 'compare_curves.json'; out.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(out, 'w'), indent=1); print(f"\n已写入 {out}")
