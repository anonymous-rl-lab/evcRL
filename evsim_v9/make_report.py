"""Build the debugging handoff from actual saved results, without pooling variants."""
import json
from pathlib import Path
from analyze import summarize, subset, REP_OFFSETS, VAL_OFFSETS, score

HERE=Path(__file__).resolve().parent


def main():
    names=['pilot_n1_s0','pilot_n1_s1','pilot_n20_s0','pilot_n20_s1','fifo_s0','fullbuf_s0']
    data={name:json.loads((HERE/'out'/f'{name}.json').read_text()) for name in names}
    if not all(d.get('final') for d in data.values()):raise RuntimeError('Wait for every bounded run to finish')
    rows={name:summarize(d) for name,d in data.items()}
    selected={}
    lines=['# 20 km 自动驾驶节能 RL：调试运行报告','',
           '本报告仅引用本轮实际运行。原附件不含历史训练权重，原 README 的 500 万步成绩不能据此独立复现。','',
           '**结论：确定的物理、指标和运行错误已修复。100 万步同种子对照中，FIFO 改善了最终与最后三次回报，但仍未超过 attentive 基线；critic 偏差和后期到达波动没有消除。不能宣称已经得到稳定节能 RL。**','',
           '## 1. 已修复的确定错误','',
           '| 问题 | 原版复现实证 | 修复及验证 |','|---|---|---|',
           '| 低速牵引漏算 | 1.5 m/s、2.6 m/s² 加速，车轮输出 7730.41 W，电池却只记录辅助用电 194.19 W | 再生截止只用于制动；起步计入牵引损耗；跨 2 m/s 连续性通过 |',
           '| 非物理制动 | 红灯前 10 m、22 m/s，投影输出 −32.13 m/s² | 强制 −3.5～2.6 m/s²；无法停车则记录违规，禁止瞬移 |',
           '| 信号穿越判定 | 原代码只使用积分步开始时刻信号 | 根据连续运动求停止线穿越时间，正确识别步内变红 |',
           '| Jerk 指标漏报 | 原版以 2 秒动作窗口的首尾加速度估计，遗漏 0.5 秒内冲击 | 按物理积分步累计 RMS、最大值及舒适性限值覆盖次数 |',
           '| 基线工况错配 | 原报告集 attentive 应为 −177.415，却比较全网格 −176.204；normal 应为 −179.363，却比较 −178.044 | 从对应工况读取基线；当前物理模型所有基线重新跑，保留原参数 |',
           '| 时间观测遗漏 | 仅剩余截止时间不足以表示旧 P 奖励中的绝对超时依赖 | 追加绝对时钟，观测 12→13 维；主训练维持 L 奖励 |',
           '| FIFO 与步数 | 覆盖槽位残留旧到达标签；步内提前终止仍加固定 repeat；短训练可能无 checkpoint | 覆盖清零并按完整插入计数标记；统计实际步数；最后不足日志间隔也保存 |',
           '| 执行及图表 | 12 个后台长进程无失败传播；绘图机器路径固定且部分数据写死 | 前台有界批处理、退出码传播、输出保护；图表从日志生成 |','',
           '安全边界：加速度上限是硬约束；紧急制动可能覆盖舒适 jerk 限值，已记录，不能宣称任意策略都满足全部舒适/交通约束。','',
           '## 2. 运行规模与验证','',
           '- 设备：本机 CPU，无 CUDA；具体 Python/NumPy/PyTorch 版本见 runtime.json。',
           '- 先检查指标与代码，再运行 40,000 步冒烟；随后进行 300,000 步双种子更新方式对照。',
           '- 最后一组固定种子 0、1,000,000 步，仅比较 replay 125,000 与 1,250,000 条决策转移。',
           '- repeat=4；1 条决策转移约对应 4 个积分步。小缓存约从 500,000 积分步开始覆盖。',
           '- 原来的 7 项检查通过；新增 8 项确定故障回归检查通过。另实跑 403 步、8 条容量、20 步回报，验证短训练与回绕。',
           '- 不运行原默认的 12 个种子 × 5,000,000 步批次，不推断 5M 稳定性。','',
           '## 3. 30 万步：更新方式对照','',
           '选择检查点只使用验证偏移 0、45 秒；下表只报告偏移 22.5、67.5 秒的 6 个工况。R 越大越好。本轮是已查看曲线的开发诊断，不是完全未触碰的最终测试。','',
           '| 方法 | 种子 | 选中步数 | 报告 R | 到达 | 最后 R | 最后三次 R |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in names[:4]:
        r=rows[name];kind='一步 TD3' if '_n1_' in name else '原式 20 步回报'
        lines.append(f"| {kind} | {data[name]['args']['seed']} | {r['selected_step']:,} | {r['reporting_R']:.3f} | {r['reporting_arrivals']}/6 | {r['last_reporting_R']:.3f} | {r['last3_reporting_R']:.3f} |")
    lines+=['','一步更新在本预算内未解决长时域价值传播问题；不能作为已经成功的修复。20 步更新学得更快，但存在探索轨迹与目标策略不一致。','',
            '## 4. 100 万步：缓存容量的同种子对照','',
            '| 缓存 | 选中步数 | 报告 R | 到达 | 最后 R | 最后三次 R | 选中时 Q−MC（全网格） |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name in names[4:]:
        r=rows[name];p=next(p for p in data[name]['curve'] if p['step']==r['selected_step'])
        lab='FIFO 125k 决策转移' if name=='fifo_s0' else '保留全部（容量 1250k）'
        lines.append(f"| {lab} | {r['selected_step']:,} | {r['reporting_R']:.3f} | {r['reporting_arrivals']}/6 | {r['last_reporting_R']:.3f} | {r['last3_reporting_R']:.3f} | {p['Q_bias']:.3f} |")
    a,b=rows['fifo_s0'],rows['fullbuf_s0']
    common=[(x,y) for x,y in zip(data['fifo_s0']['curve'],data['fullbuf_s0']['curve']) if x['step']<500000 and y['step']<500000]
    pre=max(abs(x['R']-y['R']) for x,y in common)
    lines+=['',f'开始覆盖前全网格回报最大差异：{pre:.12g}；这确认前半段训练一致。',
            f"覆盖后最终报告集差异 FIFO − 全保留 = {a['last_reporting_R']-b['last_reporting_R']:+.3f}；验证选中模型差异 = {a['reporting_R']-b['reporting_R']:+.3f}。",
            '这是一组同种子、预先固定比较，不足以声称跨种子稳定，也不足以把全部退化归因于缓存。','',
            '## 5. 节能与到达必须分别看','',
            '| 实验选中模型 | 报告均值能耗 Wh | 均值时间 s | 1312.5 s 内到达 | 1050 s 内到达 | 相同工况 attentive R |',
            '|---|---:|---:|---:|---:|---:|']
    for name in names[4:]:
        r=rows[name]
        lines.append(f"| {name} | {r['reporting_E_Wh']:.1f} | {r['reporting_t']:.1f} | {r['reporting_arrivals']}/6 | {r['reporting_within_1050s']}/6 | {r['references']['attentive']:.3f} |")
    lines+=['', '没有完成路线的低能耗不能当作节能优势。L 奖励同时给时间定价，R 的改善也不能直接写成同百分比的电量节省。FIFO 选中模型虽完成 6/6，但 1050 秒内为 0/6；平均 1141.4 秒，比同工况 attentive 的 989.0 秒慢。',
            '这里的到达时间边界沿用原任务；没有为了提高到达率放宽截止、降低罚分或重新挑选工况。','',
            '## 6. 冻结策略的 20 步污染探针','',
            '固定 30 万步 seed 0 模型，沿实际访问状态固定第一动作，只给后面 19 次决策加入原 OU 噪声，再恢复无噪声策略直至终止；末端价值用真实回放替代 critic。每个状态 8 次。','',
            '| 位置 m | 无噪声真实后续回报 | 加噪 20 步目标的平均变化 | 标准差 |','|---:|---:|---:|---:|']
    probe=json.loads((HERE/'out'/'nstep_probe.json').read_text())
    for p in probe['rows']:
        lines.append(f"| {p['x']:.0f} | {p['true_Q']:.3f} | {p['mean_shift']:+.3f} | {p['sd']:.3f} |")
    lines+=['', '探针证明：未修正的多步行为轨迹会改变正常策略的训练目标，部分起点偏移约 −9 个回报单位；也可能因赶上绿灯而出现正偏移。前两个状态的样本标准差约为 27，8 次样本不足以把 −9 当成精确总体偏差；它不是全部 critic 偏差的因果分解。TD3 的目标平滑、函数逼近、有限样本和未观测的远处信号也仍影响 Q−MC。','',
            '## 7. 可复现交付与剩余问题','',
            '- selected_checkpoints.json 按每个实验自己的验证集指定模型，没有根据报告集挑一个最好种子。',
            '- 权重、逐工况 JSON、完整日志、重算基线、原包备份和源代码修改 diff 均包含在交付包。',
            '- 工程缺陷已修复；是否值得扩大训练，应看本表的最终、最后三次和到达率，不能只挑最优一次。',
            '- 若要声称“稳定节能 RL”，仍须在明确冻结的学习规则上通过多种子长训练；本轮不作该声明。',
            '- 本轮没有加入新算法模块、行为克隆或奖励放宽。一步目标、缓存容量均作为显式对照，不将未成功的尝试标记为修复成功。',
            '- 当前输出为速度/能耗纵向仿真，使用 SUMO 导出的路段 profile；不是闭环 SUMO 背景交通训练。','']
    for name,r in rows.items():
        selected[name]=dict(checkpoint=f"out/ckpt_{name}_{r['selected_step']}steps.pt",**r)
    (HERE/'out'/'selected_checkpoints.json').write_text(json.dumps(selected,indent=2,allow_nan=False))
    (HERE/'DEBUG_REPORT_zh.md').write_text('\n'.join(lines))
    print('Wrote DEBUG_REPORT_zh.md and out/selected_checkpoints.json')


if __name__=='__main__':main()
