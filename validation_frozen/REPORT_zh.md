# TIV 冻结补充验证报告（99 条正式冻结推理回合）

日期：2026-09-14。仓库：evcRL（`validation_frozen/`）。任务书：`CODEX_TASK.md`。协议锁：`protocol_lock.json`（指纹见其中 `fingerprint`，所有回合记录与 `jobs.csv` 均带该指纹）。

## 首页摘要

| 项目 | 数值 |
|---|---|
| 正式回合 | 99 = R1 54 + R2 9 + R3 36 |
| 完成 / 驾驶失败 / 技术失败 / 尚未运行 | **99 / 0 / 0 / 0**（驾驶失败定义：闯红灯、弯道超速、未停稳、冲出终点；本轮没有） |
| 梯度更新次数 | 0（全部模型 `eval()`、无梯度；推理前后模型状态哈希一致，逐回合记录于 `summary.json`） |
| 旧基线复现 | R1 的 paper_v25 臂 27 条与归档 S 记录逐位相同（轨迹 max diff 0.0，指标完全相等；`baseline_check.json`）：27/27 停稳、0 违规、9/27 override 回合、均值 R −39.423、I_j 88.135 |
| 支持收益的新结果 | 无一项能宣称"候选记忆改动改善闭环舒适性"；R3 时机机制在扩展条件上未触发 |
| 仅支持适用边界的新结果 | R1：组合 profile 在不损失完成率与安全的前提下改变了 override 的分布与位置，但 I_j 均值上升；R2：学习策略相对常量命令的增量小且随条件而异；R3：18 对轨迹全部相同 |
| 全部运行为确定性单进程；整套 99 条重复运行一次得到逐位相同的结果 | 见第 5 节 |

## 1. 预检与协议锁

- 资产（路径与 SHA-256 均在 `protocol_lock.json`）：冻结检测器 `visual/runs/pretrain_v4/encoder.pt`（三个 S 的配置指向同一文件）；S0/S1/S2 = `visual/runs/{v4r,v4r_s1,v4r_s2}/v4_pilot/supervised/final_nets.pt`（训练种子 7/8/9，`extra_dim` 6，`critic_action=applied`）；P-T actor `timing/weights/actor_A.pt`；4 km 路线与车辆参数（`visual/v19_deps/code`）；执行器 `curve_layer.py`（jerk 2 m/s³，DT 0.5 s，与 `env20.J_MAX` 一致）；记忆 `software/EvcRL/src/evcrl/memory.py`（profile 由 `profiles.py` 冻结，`paper_v25`/`guard_v26` 身份哈希在锁内）；相机 `renderer.py`；指标 `study.episode_metrics`。
- 视觉源码身份：`visual/reproduction/reproduce_visual.py check` PASS；仓库工作树在 `validation_frozen/` 之外无改动（`source_diff_nonempty=false`）。
- EvcRL 版本：仓库、`pyproject.toml`、`evcrl.__version__`、wheel 均为 0.0.1；未从索引安装（`release_consistency.json`）。
- R1 唯一处理变量：`paper_v25`（后一步速度×时间步传播、原 15 m 距离冻结）对 `guard_v26`（实际积分位移传播 **加** 融合候选向内跨越 15 m 时的门控）。两臂差异是两项改动的组合效应。未新增滞回储备，未改 8 m 灯色冻结、终点偏置、检测阈值。
- R3 先验使用核查：`timing/runs` 只有 `development`；`timing/reports/decision.json` 记录 `reporting_started=false`。18 个扩展条件此前从未生成，也未参与 actor 或参数选择，因此称"预定扩展条件"；仍是同一路线、同一冻结 actor。
- 评估器：R1/R2 用 `run_pv.py::run_episode`，由 `visual/visual_dev/pipeline.py::evaluate` 派生（`evaluate_derivation.diff`）：删除真值终点 `cmd=0` 分支（current 协议）、显式注入记忆 profile、把实际子步位移传给记忆、逐子步扩展日志；相机、帧存储、感知、adapter、VisionEnv 执行层与物理均为归档对象。R3 用 `run_pt.py`，原样调用 `timing/code/probe.py` 的 `rollout`/`ProbeEnv`/`extra_metrics`，只把输出目录改到本目录，原 `reporting` 门控保持失败状态未改。
- 相机：评估相机基种子 1000，条件 i 外观种子 1000000+i、噪声种子 1000500+i，保留原 condition_id 与 `evalXX` 身份；两臂各自按实际位置渲染。

## 2. 烟测、测速与预算

- 烟测（≤10 min 墙钟）：`R1_S0_c00_paper_v25` 与 `R1_S0_c00_guard_v26` 两条完整回合，11.9 s 与 9.0 s；基线与归档 `dev_00.npz` 逐位相同。两条直接计入正式队列。
- 实测速率（`runtime_budget.json`）：R1/R2 视觉回合均值 6.7 s（5.4–13.8 s），R3 结构化回合均值 0.36 s。正式 99 条实际总耗时约 7.3 min（单进程）。机器：4 vCPU、无 GPU；计费率未知；外部算力 0。
- 恢复机制：每回合原子写 `trace.npz`、`layer.json`、`substeps.json.gz`、`summary.json` 与 `DONE`（含协议指纹），重启只跳过五者齐全且指纹一致的回合；R3 另有 probe.py 的决策边界完整状态快照，可回合内续跑。视觉评估未实现回合内恢复：中断最多损失当前一个回合（≤15 s）。
- 可恢复运行命令：`cd validation_frozen && OMP_NUM_THREADS=1 python3 run_pv.py --select R1,R2 && OMP_NUM_THREADS=1 python3 run_pt.py && python3 metrics.py`

## 3. R1：三个 S × 九条件 × {paper_v25, guard_v26}

主表（`tables.md`、`pairs_R1.csv`、`episodes.csv`；均为全部尝试，两臂 27/27 停稳）：

| 臂 | N | 停稳 | 信号违规 | 弯道违规 | override 回合 | override 子步 | fallback 子步（非蠕行） | 门控拒绝 | I_j 均值 | 时间 s | Wh | R | 峰值 jerk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper_v25 | 27 | 27 | 0 | 0 | 9 | 15 | 96 (28) | 0 | 88.135 | 223.72 | 597.91 | −39.423 | 10.000 |
| guard_v26 | 27 | 27 | 0 | 0 | 9 | 18 | 35 (25) | 7 | 98.817 | 219.98 | 598.09 | −39.130 | 12.159 |

分种子：S0 I_j 86.998→86.624、时间 218.9→221.1、override 回合 4→3；S1 80.026→98.114、217.9→217.7、0→4；S2 97.382→111.715、234.3→221.2、5→2。

27 对差值（guard − paper）：I_j 更低 11 / 更高 16（均值 +10.68，中位数 +4.44）；override 子步更少 8 / 相同 10 / 更多 9；时间更短 19 / 更长 8（均值 −3.74 s）；能量更低 11 / 更高 16（均值 +0.18 Wh）；R 更高 17 / 更低 10（均值 +0.29）。

按任务书顺序解释：
1. 停稳与违规：两臂均 27/27、0 信号违规、0 弯道违规，没有退化。
2. override 与 I_j：override 回合数不变（9 与 9），但分布改变。归档 9 个 override 回合中 8 个在 guard 臂无 override（S0 c0/c2/c5/c8，S2 c3/c5/c6/c8），S2 c7（远处测距修正 + 正命令）仍有；同时新增 8 个 override 回合，其中 7 个是冷电池条件 c3/c4/c6 的信号末端停车（S0、S1 各三个，S2 c4），1 个是 S1 c7 的近线灯色翻转。I_j 均值上升 10.7，峰值 jerk 由 10.0 升到 12.16。所以候选并未改善舒适性，只是把终端 fallback 型事件换成了另一批信号末端事件；fallback 子步从 96 降到 35（蠕行段大幅减少）。
3. 时间与能量：平均行程时间缩短 3.7 s，能量基本不变，R 均值提高 0.29——提高主要来自 S2 c3（paper 臂 386.5 s 的长时停等，guard 臂 274.0 s；ΔR +9.12）；去掉该对后 26 对 R 均值差为 −0.05。

机制（`events_R1.csv`、`guard_events_R1.csv`）：
- 门控只触发 7 次，全部在 t = 151.5–153.0 s、x = 2977–2991 m、车速 5.4–12.8 m/s 的近线段（六个回合：S0 c1/c3/c7，S1 c3/c7，S2 c4/c6）；被拒绝的候选距离比推算值近 0.5–6.2 m。7 次中 2 次（S1 c7、S2 c4）后续 4 个子步内出现 override 与 fallback，5 次没有。门控次数不等于避免的违规次数：本轮两臂都没有违规。
- 门控之外，位移一致传播本身改变了全部 27 条轨迹（如 S0 c0：门控 0 次，行程时间 239.5→274.5 s，I_j 104.1→80.9）。三个 S 是在 legacy 传播下训练的冻结策略，其输入通道在新传播下分布改变，这一点在解释时不能与门控效应分开。
- 余量 m⁻/m^cand/m⁺ 用执行器自身的 `backup_distance`（D_κ(v,a;0)）计算，v/a 为更新后、下一次控制前的值；备份无穷时按类别保留（本轮 7 次门控事件均有限）。
- 校验口径：原轨迹 flag（|a_next−a_prev|>J·DT+1e−7）与数值口径（|j|>J+1e−7）在 63 条回合的全部子步上没有差异（`boundary_discrepancy_steps` 合计 0）。
- 辅助统计：fallback 按 pre-step v<0.5 m/s 拆为蠕行/非蠕行，并按连续子步合并为段数（本轮定义，见 `episodes.csv`）；不与旧 163 或旧 9/27 混用。

结论（对应任务书 6.3 第二行）：R1 只有局部指标改善（fallback 子步、行程时间、8/9 归档 override 回合消失）并伴随退化（I_j 均值与峰值 jerk 上升、新增 8 个信号末端/近线 override 回合）。候选保留为实验结果，不能升级为全面有效执行器；论文中 guard 的"待测"可改为"在本条件范围内做了冻结闭环验证，结果混合"。

## 4. R2：常量 u=+1 与 S0 的同协议配对

九条常量命令回合（current 协议、S0 检测器、paper_v25 记忆、同一执行器、camera_seed 1000）：9/9 停稳、0 违规、5/9 override 回合、I_j 101.008、时间 218.39 s、606.01 Wh、R −39.288。预指定参照 S0/paper_v25（取自 R1）：9/9、4/9、86.998、218.89 s、601.00 Wh、R −39.147。

逐条件（`pairs_R2.csv`）：S0 − 常量 的 I_j 差在 c0 −21.6、c3 −6.8、c4 −20.8、c7 −91.1；c1 +4.4、c6 +4.9、c8 +5.0；c2、c5 两者几乎相同（执行层主导）。R 差：c0 +0.59、c1 +0.28、c4 +0.25、c7 +0.88；c3 −0.13、c6 −0.12、c8 −0.49；c2、c5 为 0。九对均值 ΔI_j −14.0（中位数 −0.03）、ΔR +0.14（中位数 +0.0004）、Δ时间 +0.5 s、ΔWh −5.0。

解释：按原奖励，学习策略相对常量命令的增量为小幅正值，集中在 c0、c4、c7（c1 略正）；c2、c5 两者相同，c3、c6、c8 常量更好。S1、S2 相对同一常量参照的 R 差见附表（共享同一相机与感知，不是独立基线重复）。常量并非最优控制器，本结果不支持"RL 显著优于常量"，也不支持相反结论。论文 Table V 中"Constant u=1, historical / Not aligned"一行可替换为本轮同协议结果。

## 5. R3：18 个扩展条件 × {current-color, color+timing}

两臂各 18/18 停稳、0 闯红灯、0 局部超速、0 物理越界、0 fallback、峰值 jerk 2.0。timing 臂在 2849 个子步看到倒计时、做了 909 次绿灯末端 stop-or-clear 检验，**没有一次改变所选动作**：18 对轨迹逐元素相同，信号窗口 I_j、全行程 I_j、时间、能量、R 差值均为 0（`pairs_R3.csv`，窗口全部完整）。

解释：时机机制的触发条件（绿灯末端、剩余绿灯不足以按当前速度通过且仍可停）在扩展网格（初速 14/18 m/s、偏移 15/45/75 s）上没有出现；开发条件 7 的事件仍是唯一触发案例。这是同一路线、一个冻结 actor 的条件外推结果：它说明机制在这 18 个条件下无害且不起作用，不构成跨路线泛化证据，也不自动增加更有利场景。

## 6. 确定性与重算

- 全部 99 条用同一代码跑了两遍（第二遍在修正常量策略的哈希占位对象后重新锁定协议并重跑）：两遍 63 条视觉回合的指标逐位相同，R3 36 条相同。
- `metrics.py` 从原始 `trace.npz`/`layer.json`/`substeps.json.gz` 独立重算 I_j、时间、Wh、R、override、违规、停稳，并与运行时 `summary.json` 核对（容差 1e−9）；R3 复用 `timing/code/analyze.py::independent_check`。
- 图：`fig_R1_pairs.png`（27 对 ΔI_j/Δ时间/ΔWh/Δoverride）、`fig_R1_target_updates.png`（一对回合的记忆停止线距离与速度轨迹，标出被拒绝候选与 override）。

## 7. 交付文件

`protocol_lock.json`、`preflight.json`、`jobs.csv`（99 行，含指纹与资产哈希）、`runtime_budget.json`、`results/R1|R2/<job_id>/{trace.npz, layer.json, substeps.json.gz, summary.json, DONE}`、`results/R3/{color,timing}_XX{.npz,_layer.json}` 与 `progress.json`、`episodes.csv`、`pairs_R1.csv`、`pairs_R2.csv`、`pairs_R3.csv`、`events_R1.csv`、`guard_events_R1.csv`、`metrics.py`、`metrics_summary.json`、`tables.md`、`baseline_check.py/.json`、`release_consistency.json`、`fig_R1_*.png`、`REPORT_zh.md`、`paper_paragraphs_en.md`、`evaluate_derivation.diff`、`MANIFEST_SHA256.json`、运行日志 `results/run_pv_R1R2.log`、`results/run_pt_R3.log`。
