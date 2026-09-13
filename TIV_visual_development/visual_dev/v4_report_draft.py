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
for s, runs in (('0', 'v4r'), ('1', 'v4r_s1'), ('2', 'v4r_s2'), ('0_v4enc_oldrules_reeval', 'v4')):
    t = arm_table('v4_pilot', runs); extra[f'pilot_s{s}'] = t
out.update(extra)
# 三种子汇总（审计整改后的口径）：
#  主表 = 全部 27 工况（失败工况计入分母）：到达/违规/弯道超速、全工况平均 R、jerk 越界回合数与峰值、执行层干预子步占比；
#  副表 = 与冻结臂 F 在“同种子、双方都到达且停稳”的共同工况上的配对差，逐种子标明 n 与工况号（按结果筛选的条件描述，不是无偏处理效应）；
#  参照行 = 常量命令 u=+1 + 同一冻结感知/记忆执行层的探针（runs/probes/v4r_probe_closed_loop.json）。
import numpy as np
seeds = {}
for s, runs in (('0', 'v4r'), ('1', 'v4r_s1'), ('2', 'v4r_s2')):
    d = {}
    for a in ('frozen', 'supervised', 'joint', 'joint_head'):
        p = ROOT / 'runs' / runs / 'v4_pilot' / a / 'evaluation.json'
        if p.exists():
            e = json.load(open(p)); d[a] = dict(rows={r['condition_id']: r for r in e['rows']}, z_drift=e['z']['z_drift'], intervened=e['intervened'])
    if d: seeds[s] = d
def allcond(a):
    rs = [r for s in seeds if a in seeds[s] for r in seeds[s][a]['rows'].values()]
    return rs
agg = []
for a in ('frozen', 'supervised', 'joint', 'joint_head'):
    rs = allcond(a)
    if not rs: continue
    n = len(rs); arrived = sum(r['settled'] for r in rs); viol = sum(r['violations'] for r in rs); off = sum(r.get('offroad_substeps', 0) for r in rs)
    R = np.mean([r['R'] for r in rs]); jo = sum(r['jerk_override_steps'] > 0 for r in rs); jmax = max(r['jerk_max'] for r in rs)
    interv = np.mean([r['intervened_substeps'] / max(r['trace_steps'], 1) for r in rs]); zd = np.mean([seeds[s][a]['z_drift'] for s in seeds if a in seeds[s]])
    agg.append(f"| {ZH[a]} | {arrived}/{n} | {viol} | {off} | {R:.3f} | {jo}/{n} | {jmax:.1f} | {interv * 100:.1f}% | {zd:.2f} |")
pr_ = pr.get('vision') or {}
probe_row = ''
try:
    prv = json.load(open(ROOT / 'runs/probes/v4r_probe_closed_loop.json'))['vision']
    probe_row = f"| 参照：常量 u=+1 + 冻结感知/记忆执行层（探针，9 工况） | {prv['arrived']}/9 | {prv['violations']} | {prv['offroad']} | {prv['R']:.3f} | – | {prv['max_jerk']:.1f} | – | – |"
except Exception: pass
agg_hdr = "| 臂（三种子 × 9 工况，失败工况计入分母） | 到达且停稳 | 信号违规 | 弯道超速子步 | 全工况平均 R | jerk 越界回合 | jerk 峰值 m/s³ | 执行层干预子步占比 | Z 漂移 |\n|---|---|---|---|---|---|---|---|---|"
paired = []
for a in ('supervised', 'joint', 'joint_head'):
    per = []
    for s in seeds:
        if a not in seeds[s] or 'frozen' not in seeds[s]: continue
        A = seeds[s][a]['rows']; F = seeds[s]['frozen']['rows']; common = [i for i in sorted(A) if A[i]['settled'] and F[i]['settled']]
        if not common: continue
        dd = {k: float(np.mean([A[i][k] - F[i][k] for i in common])) for k in ('Ij', 'R', 'E_Wh', 'time_s')}
        per.append((s, common, dd))
    for k, name, better in (('Ij', 'I_j（低为好）', -1), ('R', 'R（高为好）', 1), ('E_Wh', '能耗 Wh（低为好）', -1), ('time_s', '时间 s（低为好）', -1)):
        cells = ' / '.join(f"{dd[k]:+.2f} (n={len(c)})" for s_, c, dd in per); m = np.mean([dd[k] for _, _, dd in per])
        signs = [np.sign(dd[k]) for _, _, dd in per]; cons = '同号' if len(set(signs)) == 1 else '异号'
        paired.append(f"| {ZH[a]} − F | {name} | {cells} | {m:+.2f} | {cons} |")
    paired.append(f"| {ZH[a]} − F | 共同成功工况号 | " + ' / '.join(f"种子{s_}: {c}" for s_, c, _ in per) + " | – | – |")
paired_hdr = "| 配对（同种子、双方都到达且停稳的工况） | 指标 | 逐种子差 (n) | 等权均值 | 符号一致性 |\n|---|---|---|---|---|"
# 完赛均值副表（各臂自己的成功工况；分母不同，仅作描述）
cm_rows = []
for a in ('frozen', 'supervised', 'joint', 'joint_head'):
    rs = [r for r in allcond(a) if r['settled']]
    if rs: cm_rows.append(f"| {ZH[a]} | {len(rs)} | {np.mean([r['Ij'] for r in rs]):.1f} | {np.mean([r['time_s'] for r in rs]):.1f} | {np.mean([r['E_Wh'] for r in rs]):.1f} | {np.mean([r['R'] for r in rs]):.3f} |")
cm_hdr = "| 臂 | 成功工况数 | I_j | 时间 s | 能耗 Wh | R |\n|---|---|---|---|---|---|"
# 失败工况归因表（visual_dev/v4_violation_analysis.py 输出）
fail_rows = []
try:
    F = json.load(open(ROOT / 'reports/_v4_failures.json')); ZS = {'v4r': '0', 'v4r_s1': '1', 'v4r_s2': '2'}; ZA = {'frozen': 'F', 'supervised': 'S', 'joint': 'J', 'joint_head': 'JH'}
    for r in F:
        k = r.get('kind', '')
        if k.startswith('闯红灯（黄灯'):
            fail_rows.append(f"| {ZS[r['runs']]}/{ZA[r['arm']]} | {r['condition']} | {k} | {r['d_line_at_onset']} / {r['v_at_onset']} / {r['min_stop_dist']} | {'是' if r['stoppable_ideal'] else '否'} | {r['mem_phase_at_onset']} / {r['mem_d_line_at_onset']} | {r['substeps_onset_to_cross']} | {r['crossing_speed']} | – |")
        elif k.startswith('红灯蠕行') or k.startswith('闯红灯'):
            fail_rows.append(f"| {ZS[r['runs']]}/{ZA[r['arm']]} | {r['condition']} | {k} | – | – | 过线时 {r.get('mem_phase_at_cross')} / 锁存 {r.get('mem_hold_at_cross')} | – | {r['crossing_speed']} | – |")
        else:
            fail_rows.append(f"| {ZS[r['runs']]}/{ZA[r['arm']]} | {r['condition']} | {k} | – | – | 停车时 {r.get('mem_phase_at_stop')} / d 估计 {r.get('mem_d_line_at_stop')} | – | – | 线前 {-r['stop_rel_line']:.1f} m，静止 {r['stopped_substeps']} 子步，真值绿而感知非绿 {r['perceived_nongreen_while_truth_green']} 子步 |")
except Exception as e: fail_rows = [f'| 归因文件缺失：{e} |']
fail_hdr = "| 种子/臂 | 工况 | 类型 | 黄灯起始（子步前状态）d m / v m/s / v²/7 m | 理想可停 | 记忆相位 / d 估计 | 起始到过线子步 | 过线速度 m/s | 停车细节 |\n|---|---|---|---|---|---|---|---|---|"
out.update(fail_hdr=fail_hdr, failures="\n".join(fail_rows))
out.update(agg_hdr=agg_hdr, agg="\n".join(agg + ([probe_row] if probe_row else [])), paired_hdr=paired_hdr, paired="\n".join(paired), cm_hdr=cm_hdr, cm="\n".join(cm_rows), n_seeds=len(seeds))
json.dump(out, open(ROOT / 'reports' / '_v4_tables.json', 'w'), ensure_ascii=False, indent=1); print({k: (len(v_) if hasattr(v_, "__len__") else v_) for k, v_ in out.items()})
