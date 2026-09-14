"""绘制重训曲线与归档曲线的对比图（验证集/报告集均值回报、报告集完赛数、critic 起点偏差）。

用法：python scripts/plot_curves.py [results/train/repro_s0.json ...]   （默认：results/train 下全部）
图内文字为中文；若系统无 CJK 字体（脚本检测 WenQuanYi/Noto CJK），自动退回英文字体以免出现方框。
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_cjk=[f.name for f in font_manager.fontManager.ttflist if any(k in f.name for k in ('WenQuanYi','Noto Sans CJK','Noto Sans SC','SimHei'))]
if _cjk: plt.rcParams['font.family']=_cjk[0]; plt.rcParams['axes.unicode_minus']=False
ZH=bool(_cjk)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as P
VAL = {0., 45.}; REP = {22.5, 67.5}
def sc(c, offs):
    g = [r for r in c['grid'] if float(r['offset']) in offs]
    return float(np.mean([r['R'] for r in g])), sum(bool(r['arr']) for r in g)
paths = [Path(p) for p in sys.argv[1:]] or sorted(p for p in (P.RESULTS / 'train').glob('*_s*.json') if not p.name.startswith('analyze'))
refs = json.load(open(P.REFERENCE / 'refs_fixed.json'))
attentive = np.mean([refs['grid'][k]['humans']['attentive']['R'] for k in refs['grid'] if float(k.split('_')[-1]) in REP])
rule = json.load(open(P.REFERENCE / 'cruise_baseline.json'))['reporting']['R']
fig, axes = plt.subplots(len(paths), 3, figsize=(15, 3.6 * len(paths)), squeeze=False)
for row, path in zip(axes, paths):
    new = json.load(open(path)); seed = new['args']['seed']
    old = json.load(open(P.REFERENCE / 'bench_curves' / f'bench_s{seed}.json'))
    for label, run, style in ((('归档（论文）' if ZH else 'archived'), old, dict(color='0.5', ls='--')), (('本次重训' if ZH else 're-trained'), new, dict(color='C0'))):
        steps = np.array([c['step'] for c in run['curve']]) / 1e6
        val = [sc(c, VAL)[0] for c in run['curve']]; rep = [sc(c, REP)[0] for c in run['curve']]
        arr = [sc(c, REP)[1] for c in run['curve']]; bias = [c['Q_bias'] for c in run['curve']]
        row[0].plot(steps, rep, marker='o', ms=3, label=f'{label}：报告集' if ZH else f'{label}: reporting', **style)
        row[0].plot(steps, val, marker='x', ms=3, alpha=.6, label=f'{label}：验证集' if ZH else f'{label}: validation', **{**style, 'ls': ':'})
        row[1].plot(steps, arr, marker='o', ms=3, label=label, **style)
        row[2].plot(steps, bias, marker='o', ms=3, label=label, **style)
    row[0].axhline(attentive, color='C3', lw=.8, label=(f'attentive 驾驶员 {attentive:.2f}' if ZH else f'attentive {attentive:.2f}'))
    row[0].axhline(rule, color='C2', lw=.8, label=(f'18.5 m/s 定速规则 {rule:.2f}' if ZH else f'rule {rule:.2f}'))
    row[0].set_title((f'种子 {seed}：平均回报' if ZH else f'seed {seed}: mean return')); row[0].set_ylim(-230, -165); row[0].legend(fontsize=6)
    row[1].set_title(('报告集完赛数 / 6' if ZH else 'reporting arrivals / 6')); row[1].set_ylim(-.3, 6.3); row[1].legend(fontsize=7)
    row[2].set_title(('critic 起点偏差 Q(s0) - MC' if ZH else 'critic bias')); row[2].legend(fontsize=7)
    for ax in row: ax.set_xlabel(('积分步（百万）' if ZH else 'integration steps (M)')); ax.grid(alpha=.3)
plt.tight_layout(); out = P.RESULTS / 'train' / 'curves_vs_archived.png'; out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130); print('已写入', out)
