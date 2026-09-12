"""逐检查点训练诊断表（先升后降现象），可用于归档曲线或本次新曲线。

列：验证集/报告集均值回报、报告集完赛数、12 个起点状态上的 critic 值与实际蒙特卡洛回报及其偏差
（Q_s0、MC、bias）、回放池构成（完赛/超时 episode 占比）、critic 损失、双 critic 差、巡航段均速。
用法：python scripts/diagnose_curves.py [curve.json ...]   （默认：归档 bench_s0..7）
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as P
VAL = {0., 45.}; REP = {22.5, 67.5}
def sc(c, offs):
    g = [r for r in c['grid'] if float(r['offset']) in offs]
    return float(np.mean([r['R'] for r in g])), sum(bool(r['arr']) for r in g)
paths = [Path(p) for p in sys.argv[1:]] or [P.REFERENCE / 'bench_curves' / f'bench_s{s}.json' for s in range(8)]
lines = []
for path in paths:
    d = json.load(open(path)); curve = d['curve']
    vals = [sc(c, VAL)[0] for c in curve]; k = int(np.argmax(vals))
    lines.append(f"\n### {path.name}（种子 {d['args']['seed']}）：验证集峰值在第 {k} 个检查点（步 {curve[k]['step']}），"
                 f"其后还有 {len(curve)-1-k} 个检查点；峰值处报告集 R {sc(curve[k],REP)[0]:.2f}，末尾 {sc(curve[-1],REP)[0]:.2f}")
    lines.append(f"{'步数':>8} {'验证R':>8} {'报告R':>8} {'完赛':>3} {'Q_s0':>7} {'MC':>7} {'偏差':>6} {'池完赛比':>7} {'池超时比':>6} {'Q损失':>7} {'双Q差':>5} {'饱和':>4} {'违规':>4} {'停车':>5} {'巡航v':>8}")
    for c in curve:
        v, _ = sc(c, VAL); r, a = sc(c, REP)
        lines.append(f"{c['step']:>8} {v:8.2f} {r:8.2f} {a:>3} {c['Q_s0']:7.1f} {c['MC_s0']:7.1f} {c['Q_bias']:+6.1f} {c['buf_arrived']:7.2f} {c['buf_timeout']:6.2f} "
                     f"{c['qloss']:7.1f} {c['twin']:5.2f} {c['sat']:4.2f} {c['viol']:4.2f} {c['stops']:5.2f} {c['cruise_v']:8.1f}")
text = '\n'.join(lines); print(text)
out = P.RESULTS / ('curve_diagnostics_' + ('archived' if len(sys.argv) == 1 else 'custom') + '.md')
out.write_text('```\n' + text + '\n```\n'); print(f"\n已写入 {out}")
