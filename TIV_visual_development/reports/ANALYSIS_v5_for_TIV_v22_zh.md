# v5 时机实验对 TIV_v22 论文的意义：能支持什么、不能支持什么、怎么写

日期：2026-09-14。对象：`TIV_v22`（Manuscript_v22.md / Supplementary_v22.md）与 `REPORT_v5_timing_zh.md`（36 条正式分支 + 72 条事后扫描）。

## 1. 论文当前的证据缺口正是 v5 填的

v22 第 VII-C 节明写："No paired camera-cue-retention by jerk-bound intervention or complete event-wise recognition-to-preparation latency study is available. These are limits on mechanism attribution." Table II 的"Timing use"行注明是 SPaT 执行层时机使用，"not visual-encoding delay"；第 VI-B 节的时机证据只有 9 对里的 1 个事件。

v5 正是"配对的视觉线索保留 × jerk 上限干预"，并逐事件记录了识别→采纳→动作的时延与余量：同一冻结状态、同一冻结感知/记忆/执行层、三个冻结 S 权重，只改灯出现位置与 jerk 上限。它把 Table II 里"视觉驾驶"（能完成）与"时机使用"（执行层）两行之间的空档接上：**在实际视觉链路里，执行约束决定了同一条视觉线索必须多早被采纳。**

## 2. v5 支持的论文主张（可以写进正文）

1. **执行条件收紧要求更早的信息**（对应 C1 的机制、Table I 第 1 行"stopping infeasible"边界的车辆级对应）。需停车情境：采纳距离 187 m 时两种 jerk 上限、三权重全部高速段平顺；采纳距离 109 m 时 jerk 2 在采纳后的第一个决策（d = 98 m，v = 21.5 m/s）进入应急制动、jerk 4 三权重全部平顺；采纳距离 75 m 时两者都应急（d = 65 m）。事后扫描把边界定在 jerk 2：109 m 失去、jerk 4：98 m 仍保持、87 m 失去，与备份制动距离推出的阈值（109.3 / 98.1 m）一致，差正好一帧（0.5 s，11 m）。三权重逐值相同。
2. **恢复可行但有代价**（对应摘要与 VII-A "recovery can be feasible and still costly"）。较晚采纳时车仍停在线前、0 违规（恢复可行），但靠应急回退（脱离平顺执行族，jerk 峰值 7.0）并多耗能：需停车分支能耗 87–99 → 121–126 → 131–133 Wh（正常 → 中等 → 较晚），三权重一致。
3. **"准备"的代价出现在通行分支**（对应定理的 pass-branch 惩罚的经验对应物，但方向要说清楚）。允许通行情境：正常可见时三个 S 策略看到灯就减速（过线前最低速度 15.7–18.9 m/s，I_j 14–34），耗时比灯较晚出现的分支多 0.5–2.5 s、多耗 2.4–6.3 Wh（种子 1/jerk 4 因近线灯色抖动多 23 Wh，单列）；灯较晚出现时车不作准备、直接通过（I_j ≈ 0）。这是"早知道就准备、准备在通行分支付出时间"的直接观测——但请注意第 3 节第 2 条：这里付代价的是**早**观察者（策略保守准备），不是定理里的延迟观察者。
4. **四个时刻可测且顺序正确**：线索开放 → 检出（同帧）→ 采纳（+0.5 s）→ 动作（同拍或下一决策）；首次应急回退紧随采纳时刻（中等延迟：采纳 6.5 s、回退 7.0 s），差异发生在响应机会消失之前或附近，满足论文对"时序记录"的要求。
5. **重复性与无泄漏**：三权重重复；隐藏期检出分数 ≤ 0.033、记忆无轨迹；jerk 三处参数统一；同一分支两次运行逐子步一致。

## 3. v5 不支持的、不能写的

1. **不能把 v5 的边界数值与 Table I 的 ρ 阈值等同**。定理是三段等长斜坡族（Δ、s、r、ρ），v5 的执行层是 curve_layer 的"斜坡到 −3.5 m/s² 持续再释放"备份，执行层目标还带 0.15d+8 m 的余量；两者的边界只在"更严的 jerk 上限要求更早采纳"这一定性结论上对应。论文里写"车辆级的对应关系"，不写"验证了阈值"。
2. **v5 没有实现定理中的"共同准备"**。v4r 执行层是反应式的：首见即绿假定剩余 30 s、见红才停；它不会为两种情境做共同准备。所以 v5 观察到的是"不准备 → 通行分支快、停车分支只能应急"与"早看到 → 策略保守减速 → 通行分支慢、停车分支平顺"的权衡，不是式 (9) 的最优延迟惩罚。固定策略下的行程差按论文 V-E 的口径记为"该策略下的配对效果"，不改名为最优信息损失（论文 VI-B 末段已有同样的措辞，可沿用）。
3. **差异窗口只有一帧**（D_2 − D_4 = 9.5 m，2 Hz 感知）。这是当前几何与感知帧率决定的，事后扫描能分辨但不能更细；不能声称"jerk 上限对信息时机的敏感度"的连续曲线。
4. **三权重逐值相同不是三个独立复现**：冻结状态相同、灯出现后执行层主导（v4r 干预 88–95% 子步）。它证明机制在执行层×信息时机上成立，不证明策略学到了准备行为。论文里三权重应写成"三个冻结策略下重复观察到同一机制"，不写"三次独立试验"。
5. **单一冻结状态、单一接近速度（22.2 m/s）、单一路线**。边界随速度移动（D_stop 随 v² 增长），本轮没有第二个起始速度。
6. **余量 M_j 是备份制动口径**：负余量 = 该备份失效，不是"所声明控制类中不可行"；本轮未做约束优化核对（论文 VI-D 里 v²/(2B) 的单向判据措辞可以沿用）。
7. **jerk = 4 是机制对照，不是舒适性标准**；jerk 4 分支的全程 I_j 更高（起步段策略油门更冲），不能反过来读成"放宽 jerk 更舒适"。
8. **一条异常**：通行/正常/jerk 4/种子 1 在过线处出现 jerk 10.4 的应急（记忆灯色近线抖动引起的短暂停车目标，v4r 已知近线感知问题）；停车/正常/jerk 4/种子 2 全程能耗 51 Wh 明显低于其他（起步段行为差异，未深究）。两者都要保留在表里。

## 4. 建议的写法

**摘要**（在"A separate timing intervention changes one of nine paired outcomes…"之后加一句）：
"A paired visual-cue timing intervention on three frozen supervised actors shows that a stricter jerk bound requires the cue to be adopted one perception frame (0.5 s, 11 m at 22 m/s) earlier to preserve a smooth stop; later adoption remains recoverable only through emergency braking with higher energy."

**Table II 新增一行**：
| Visual cue timing × jerk bound / Table VI | Same frozen state, detector, memory and executor; three frozen S actors; cue reveal position and jerk bound varied | Smooth stop preserved at 109 m adoption only under the looser bound (3/3); both lost at 75 m; boundary one frame apart | Frozen policies, one approach state, 2 Hz perception; executor-dominated; not the optimal preparation penalty |

**正文 VI-B 之后新增 VI-B′（或替换 Fig. 3 的位置）**：一张两栏图——(a) 三条视觉条件下 jerk 2 与 jerk 4 的速度-距离曲线与首次应急点；(b) 事后扫描：高速段是否平顺 vs 采纳距离，两条 jerk 上限，叠加解析阈值 109.3 / 98.1 m。表格用 `REPORT_v5_timing_zh.md` 第 5 节的精简表（情境 × 视觉 × jerk：采纳时 d、M2|M4、高速段平顺 n/3、首次高速回退 d/v、接近段 I_j、能耗）。

**VII-A/VII-B 的衔接句**："The visual-cue timing study places the model's first boundary in the actual perception–memory–execution chain: with the stricter jerk bound the same cue must be adopted one frame earlier to keep the smooth-stop option, and later adoption is recovered only by emergency braking at higher energy. The passing branches show the mirror cost of preparation: actors that see the lamp early slow down and lose 0.5–2.5 s. These are paired effects under fixed policies and do not estimate the optimal common-preparation penalty of (9), which the reactive executor does not implement."

**VII-C 的限制句**：删掉"No paired camera-cue-retention by jerk-bound intervention … is available"，改成"The paired cue-timing study uses one frozen approach state, a single approach speed and 2 Hz perception; its one-frame boundary separation follows from D_2 − D_4 = 9.5 m and is executor-dominated, so it does not measure learned anticipation."

**Supplementary**：新增一节（建议 L）：冻结状态与来源、延迟推导（D_stop、余量公式、τ 实测 0.5 s、烟测后冻结、种子 0 为开发证据）、接口核对四项、36 条逐分支表、事后扫描表、异常两条、评价接口（2 s 命令 → 4 子步 → 状态/代价，不用旧 critic）。

**一致性检查**（check_v22.py 可加）：分支数 36 / 72；高速段平顺计数 {normal: 6/6, medium: jerk2 0/3, jerk4 3/3, late: 0/6}；首次高速回退位置 {98 m@21.5, 65 m@20.5}；边界 {jerk2 ≤109 失, jerk4 98 保 / 87 失}；违规 0/36。

## 5. 如果还想加强（都在分钟量级，不新增训练）

1. 第二个冻结状态（不同接近速度，如从工况 v0 = 16 m/s 的轨迹在 x ≈ 2740 处冻结，或人为选 v ≈ 18 m/s 的时刻）：预期边界按 D_stop(v) 移动，能把"边界随执行约束与速度移动"从一点变成一条线。
2. 感知帧率 4 Hz 的对照（只改采集频率，不改网络）：能把一帧 0.5 s 的分辨率降到 0.25 s，验证窗口是否仍与 D_2 − D_4 一致。
3. jerk 上限 3 m/s³ 作为第三档：边界预期落在两者之间，能排除"只是两点碰巧"。

这三项都不改变 v5 已冻结的正式表；作为补充扫描标注即可。
