"""三臂小规模汇总：控制指标、视觉指标、编码诊断、训练速率，以及训练诊断曲线图。

用法：python visual_dev/summarize_pilot.py --tag pilot
输出：runs/<tag>/summary.json、runs/<tag>/summary_zh.md、runs/<tag>/diagnostics.png
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_cjk = [f.name for f in font_manager.fontManager.ttflist if any(k in f.name for k in ('WenQuanYi', 'Noto Sans CJK', 'Noto Sans SC'))]
if _cjk: plt.rcParams['font.family'] = _cjk[0]; plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parents[1]
ARMS = ('frozen', 'supervised', 'joint', 'joint_head'); ZH = {'frozen': 'F 冻结', 'supervised': 'S 仅监督', 'joint': 'J 联合', 'joint_head': 'JH 联合(冻结骨干)'}


RUNS_SUB = ''


def load_arm(tag, arm):
    d = ROOT / 'runs' / RUNS_SUB / tag / arm
    if not (d / 'evaluation.json').exists(): return None
    st = json.load(open(d / 'status.json')); ev = json.load(open(d / 'evaluation.json'))
    diag = completed = None
    if (d / 'resume.pt').exists():
        s = torch.load(d / 'resume.pt', map_location='cpu', weights_only=False); diag = s['diag']; completed = s['completed']
    return dict(status=st, eval=ev, diag=diag, completed=completed)


def main(tag):
    arms = {a: load_arm(tag, a) for a in ARMS}; arms = {a: v for a, v in arms.items() if v}
    if not arms: sys.exit('没有可汇总的臂')
    summary = {}; lines = [f"### 三臂小规模汇总（tag={tag}）\n", "| 臂 | 子步 | 更新 | 训练 episode/到达 | 子步/s | 更新/s | 帧存储 MiB | 评估静止完赛/9 | 红灯 | 回退 | 干预子步 | 平均 I_j | 最坏 jerk | 平均时间 s | 平均能耗 Wh | 平均 R | Z 漂移 | Z 标准差 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, v in arms.items():
        st, ev = v['status'], v['eval']; cm = ev['summary']['completed_mean']
        summary[a] = dict(substeps=st['substeps'], updates=st['updates'], episodes=st['episodes'], train_arrivals=st['train_arrivals'], substeps_per_s=st['substeps_per_s'],
            updates_per_s=st['updates_per_s'], train_wall_s=st['wall']['train'], save_wall_s=st['wall']['save'], frame_store_MiB=st['frame_store_MiB'], settled=ev['settled'],
            violations=ev['summary']['violations'], fallback=ev['fallback'], intervened=ev['intervened'], mean_Ij=cm['Ij'], jerk_max=ev['summary']['max_jerk'], mean_time_s=cm['time_s'],
            mean_E_Wh=cm['E_Wh'], mean_R=cm['R'], z=ev['z'], visual=ev['visual'], per_condition=[dict(id=r['condition_id'], Ij=r['Ij'], time_s=r['time_s'], E_Wh=r['E_Wh'], R=r['R'], settled=r['settled'], violations=r['violations'], intervened=r['intervened_substeps'], gap=r['mean_abs_command_gap']) for r in ev['rows']])
        lines.append(f"| {ZH[a]} | {st['substeps']} | {st['updates']} | {st['episodes']}/{st['train_arrivals']} | {st['substeps_per_s']:.2f} | {st['updates_per_s']:.2f} | {st['frame_store_MiB']:.0f} | {ev['settled']} | {ev['summary']['violations']} | {ev['fallback']} | {ev['intervened']} | {cm['Ij']:.2f} | {ev['summary']['max_jerk']:.3f} | {cm['time_s']:.1f} | {cm['E_Wh']:.1f} | {cm['R']:.2f} | {ev['z']['z_drift']:.3f} | {ev['z']['z_std']:.3f} |")
    lines.append("\n视觉（行驶中控制灯色识别；每格 = n / 已知类正确率 / unknown 召回 / 红→绿；ROI 头 ‖ Z 头）\n")
    bins = sorted({b for v in arms.values() for b in v['eval']['visual']})
    lines.append("| 臂 | " + " | ".join(bins) + " |"); lines.append("|---|" + "---|" * len(bins))
    def cell(m):
        if not m: return '—'
        def one(h): return '—' if not h else f"{h['n']}/{('%.2f' % h['known_acc']) if h['known_acc'] is not None else '无已知'}/{('%.2f' % h['unknown_recall']) if h['unknown_recall'] is not None else '无unk'}/{h['red_to_green']}"
        return one(m.get('roi_head')) + ' ‖ ' + one(m.get('z_head'))
    for a, v in arms.items():
        lines.append(f"| {ZH[a]} | " + " | ".join(cell(v['eval']['visual'].get(b)) for b in bins) + " |")
    if all(v['eval'].get('z_decodability_dev') for v in arms.values()):
        lines.append("\nZ 可解码性（dev 固定序列集，线性探针，一半训练一半测试）与同状态换图响应（红↔绿，同外观同噪声）\n")
        lines.append("| 臂 | 探针测试准确率 | 已知类准确率 | 机会水平 | signal_z 头准确率 | 换图 \|Δu\| 均值 | 最大 | >0.05 比例 | Δu(绿−红) 均值 | ‖ΔZ‖ 均值 | ROI 头颜色对 | Z 头颜色对 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a, v in arms.items():
            zd = v['eval']['z_decodability_dev']; sw = v['eval']['image_swap']
            lines.append(f"| {ZH[a]} | {zd['probe_test_acc']:.3f} | {zd['probe_test_known_acc'] if zd['probe_test_known_acc'] is None else '%.3f' % zd['probe_test_known_acc']} | {zd['chance']:.3f} | {zd['signal_z_head_acc']:.3f} | {sw['mean_abs_du']:.4f} | {sw['max_abs_du']:.4f} | {sw['frac_abs_du_gt_0_05']:.2f} | {sw['mean_du_green_minus_red']:+.4f} | {sw['mean_dz']:.3f} | {sw['roi_head_color_correct']:.2f} | {sw['z_head_color_correct']:.2f} |")
    lines.append("\n逐工况（I_j / 时间 s / 能耗 Wh，静止完赛=✓）\n")
    ids = sorted({r['id'] for v in summary.values() for r in v['per_condition']})
    lines.append("| 工况 | " + " | ".join(ZH[a] for a in arms) + " |"); lines.append("|---|" + "---|" * len(arms))
    for i in ids:
        cells = []
        for a in arms:
            r = next(x for x in summary[a]['per_condition'] if x['id'] == i)
            cells.append(f"{r['Ij']:.1f} / {r['time_s']:.1f} / {r['E_Wh']:.0f} {'✓' if r['settled'] else '✗'}")
        lines.append(f"| {i} | " + " | ".join(cells) + " |")
    # 配对差：J−S 与 S−F（主比较 J 对 S）
    if all(a in summary for a in ARMS):
        def m(a, k): return summary[a][k]
        lines.append(f"\n配对差：J−S 平均 I_j {m('joint','mean_Ij')-m('supervised','mean_Ij'):+.2f}，时间 {m('joint','mean_time_s')-m('supervised','mean_time_s'):+.1f} s，能耗 {m('joint','mean_E_Wh')-m('supervised','mean_E_Wh'):+.1f} Wh，R {m('joint','mean_R')-m('supervised','mean_R'):+.2f}；"
                     f"S−F 平均 I_j {m('supervised','mean_Ij')-m('frozen','mean_Ij'):+.2f}，时间 {m('supervised','mean_time_s')-m('frozen','mean_time_s'):+.1f} s，能耗 {m('supervised','mean_E_Wh')-m('frozen','mean_E_Wh'):+.1f} Wh，R {m('supervised','mean_R')-m('frozen','mean_R'):+.2f}。"
                     "单开发种子、单条路线、小预算：这些是接口与闭环行为的记录，不是性能增益证明。")
    # 训练诊断曲线
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, v in arms.items():
        if not v['diag']: continue
        u = [d['updates'] for d in v['diag']]
        ax[0].plot(u, [d['q_loss'] for d in v['diag']], label=ZH[a]); ax[1].plot(u, [d['vision_loss'] for d in v['diag']], label=ZH[a])
        ax[2].plot(u, [d['encoder_grad'] for d in v['diag']], label=ZH[a])
        if v['completed']:
            ax[3].plot([c['episode'] for c in v['completed']], [c['Ij'] for c in v['completed']], marker='o', ms=3, label=ZH[a])
    for x, t in zip(ax, ('critic TD 损失', '视觉监督损失', '编码器总梯度范数（TD+视觉）', '训练 episode 的 I_j（含探索噪声）')):
        x.set_title(t); x.grid(alpha=.3); x.legend(fontsize=8)
    ax[0].set_xlabel('更新数'); ax[1].set_xlabel('更新数'); ax[2].set_xlabel('更新数'); ax[2].set_yscale('symlog'); ax[3].set_xlabel('episode')
    plt.tight_layout(); out = ROOT / 'runs' / RUNS_SUB / tag; plt.savefig(out / 'diagnostics.png', dpi=120)
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False); (out / 'summary_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines)); print('已写入', out / 'summary_zh.md', out / 'diagnostics.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='pilot'); ap.add_argument('--runs', default='', help='runs 子目录，如 v3'); a = ap.parse_args()
    RUNS_SUB = a.runs; main(a.tag)
