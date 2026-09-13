"""生成 reports/REPORT_v4_zh.md 的数据段（探针、检测器、小规模、规模运行）。用法：python visual_dev/v4_report_draft.py"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
pr = json.load(open(ROOT / 'runs/probes/v4_probe_closed_loop.json')); thr = json.load(open(ROOT / 'runs/pretrain_v4/thresholds.json'))
rep = json.load(open(ROOT / 'runs/pretrain_v4/pretrain_report.json')); v = rep['dev_v4']
ZH = {'frozen': 'F 冻结', 'supervised': 'S 仅监督', 'joint': 'J 联合(整网)', 'joint_head': 'JH 联合(冻结骨干只训末端)'}
f1 = lambda x: 'nan' if x is None else f"{x:.1f}"; f3 = lambda x: 'nan' if x is None else f"{x:.3f}"
def prow(name, r): return f"| {name} | {r['arrived']}/9 | {r['violations']} | {r['offroad']} | {r['fallback']} | {f1(r['Ij'])} | {f1(r['time_s'])} | {f1(r['E_Wh'])} | {f3(r['R'])} |"
probe_rows = "\n".join(prow(n, pr[k]) for n, k in (('oracle 记忆', 'oracle'), ('oracle / 灯色灭', 'oracle_hide_light_color'), ('视觉记忆', 'vision'), ('视觉 / 隐藏警示牌', 'vision_hide_curve_sign'), ('视觉 / 灯色灭', 'vision_hide_light_color'), ('视觉 / 隐藏解除牌', 'vision_hide_release_sign'), ('视觉 / 隐藏终点标志', 'vision_hide_end_marker')))
det = "\n".join(f"| {c} | {v[c]['n']} | {v[c]['recall']:.3f} | {v[c]['false_alarm']:.4f} | {f1(v[c]['dist_mae'])} | {thr['recommended'][c]} | {thr['table'][c]['recall_at_recommended']:.3f} / {thr['table'][c]['false_alarm_at_recommended']:.4f} |" for c in ('traffic_light', 'curve_sign', 'end_marker', 'release_sign'))
def arm_table(tag, runs='v4'):
    rows = []
    for a in ('frozen', 'supervised', 'joint', 'joint_head'):
        p = ROOT / 'runs' / runs / tag / a / 'evaluation.json'
        if not p.exists(): continue
        e = json.load(open(p)); cm = e['summary']['completed_mean'] or {}
        abl = ' / '.join(f"{k.replace('hide_', '')}: {x['arrived']},{x['violations']},{x['offroad']}" for k, x in (e.get('ablation') or {}).items())
        rows.append(f"| {ZH[a]} | {e.get('arrived', '-')}/9 | {e['summary']['violations']} | {e.get('offroad', '-')} | {e['fallback']} | {f1(cm.get('Ij'))} | {f1(cm.get('time_s'))} | {f1(cm.get('E_Wh'))} | {f3(cm.get('R'))} | {e['z']['z_drift']:.3f} | {e['z_decodability_dev']['probe_test_acc']:.3f} | {abl} |")
    return "\n".join(rows)
hdr = "| 臂 | 到达 | 闯红灯 | 弯道超速 | 回退 | I_j | 时间 s | 能耗 Wh | R | Z 漂移 | Z 探针 | 消融（到达,闯红灯,弯道超速）警示牌/灯色/终点 |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
out = dict(probe_rows=probe_rows, det=det, small=arm_table('v4_small'), hdr=hdr, teacher=f"{v['light_color']['teacher_roi_known_acc']:.3f}", zhead=f"{rep['dev_z_head_latest_frame']['known_acc']:.3f}", ij_vis=f"{pr['vision']['Ij']:.0f}", ij_or=f"{pr['oracle']['Ij']:.0f}")
extra = {}
for s, runs in (('0', 'v4'), ('1', 'v4_s1'), ('2', 'v4_s2')):
    t = arm_table('v4_pilot', runs); extra[f'pilot_s{s}'] = t
out.update(extra)
json.dump(out, open(ROOT / 'reports' / '_v4_tables.json', 'w'), ensure_ascii=False, indent=1); print({k: len(v_) for k, v_ in out.items()})
