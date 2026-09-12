# 论文 ↔ 代码逐项核对（长程 20 km 环境感知 TD3）

核对对象：`Manuscript_v19.md` 第 III、VI 节与补充材料附录 G、H；冻结代码 `evsim_v9/`（与 `long_route/code_evsim_v9.tgz` 字节一致，SHA-256 见 `reference/code_evsim_v9.tgz.sha256`）。
所有“核对结果”均为本次实际读代码或实际运行得到，不是转述论文。

## 1. 车辆与能量模型（论文 III-A、附录 G）

| 论文陈述 | 代码位置 | 核对结果 |
|---|---|---|
| m=1928 kg，coast-down 路阻 A+Bv+Cv²（[S4] 2019 Leaf Plus） | `plant.py` `Vehicle`：`m=1928`, `rl_A=30.360 lbf`, `rl_B=0.3201 lbf/mph`, `rl_C=0.0196 lbf/mph²` | 一致；文中 A≈135.05 N、B≈3.1851、C≈0.43626 为换算后取整 |
| 电机 340 N·m / 160 kW，轮半径 0.3234 m，终传动 8.139 | `plant.py` `Vehicle` | 一致；F_max=8556.77 N，基速 18.7 m/s |
| 效率损耗模型 P_loss=P0+k_Fe v+k_Cu F²，锚点 96%@4000 N/15 m/s、90%@421 N/22.22 m/s，P0=300 W | `effcal.py` `_fit()`，`models.py` 用 `VEH.eta = effcal.eta_cal` 替换 `plant.py` 的旧效率图 | 一致；η 裁剪到 [0.05, 0.97] |
| 电池 96s2p×50 Ah，3.65 V 标称 → 35040 Wh；OCV=3.20+0.90s^0.55+0.12s^6；R0(T) Arrhenius；脉冲接受上限 min{…, 90 kW} | `plant.py` `Pack`, `ocv_cell`, `r0_cell`, `p_bat_max_dc` | 一致；`E_BATT_J = PACK.capacity_j` = 35040 Wh |
| 制动分配：v≥2 m/s 时再生上限为不动点 F_ceil；v<2 m/s 再生关闭，牵引仍可用 | `plant.py` `regen_ceiling_mech`（阻尼不动点 40 次）；`models.py` `v1_power` | 一致；低速牵引修复由 `test_regressions.py::test_launch_energy_not_free` 固定 |
| 齿轮效率 0.97，辅助负载 167/0.86≈194 W | `models.py` `ETA_TR=0.97`，`plant.py` `Aux.p_base_dc` | 一致 |
| 80 km/h 稳态约 136 Wh/km | 本次运行 `effcal.py __main__` 表格 | 一致（数量级核对） |

## 2. 奖励、路线、观测与终止（论文 III-A/III-B、式 (2)、附录 G-C）

| 论文陈述 | 代码位置 | 核对结果 |
|---|---|---|
| r_t = −P_b Δt/E0 − λ_T Δt − k_os·min{(v−v_max)+,5}²Δt，E0=100 kJ，λ_T=0.08，k_os=0.02 | `env20.py` `step()`：`r=-p_b*DT/1e5`；`LAM_T=0.08`；`K_OVERSPEED=0.02`，`V_OVER_HARD=5` | 一致（`REWARD_MODE='L'`，`SHAPE_C=0`） |
| 未完成距离收费 15(L−x_T)/1000，违规收费并入 P | `CTG_PER_KM=15`，`OFFROAD_PEN=5`（闯红灯） | 一致；`verify.py` “objective consistency” 实测 Σr = −E/E0 − λt − ctg |
| 20 km、80 km/h；三处 35 km/h 弯道 3450–3550、10950–11050、17950–18050 m；信号 7000、15000 m；周期 90 s、绿灯 30 s，其余不可通行 | `route20.py` `CURVES`, `SIGNALS`, `CYCLE=90`, `GREEN=30`, `YELLOW=4`（计为不可通行） | 一致；`scn/arterial_profile.npy` 由 SUMO netconvert 导出并缓存 |
| 13 维观测 (G7)：v/v_max, a/3.5, d_end/L, v_lim/v_max, d_c/400, v_c/v_max, d_s/1000, φ_s, τ_s, (SOC−0.5)/0.5, (T−283.15)/30, (t_end−t)/T_b, t/T_b；超出 1000 m 时 (φ,τ)=(−1,1) | `env20.py` `obs()`；`SPAT_RANGE=1000`, `PREVIEW=400` | 一致，逐通道核对 |
| 名义时间表 1050 s，实际截止 1312.5 s，截止为终止（不 bootstrap） | `T_BUDGET=1050`，`t_end=1.25*t_budget`；`td3_run.py` 第 252 行 `D[ptr]=float(d_n)`，`done` 包含 `t>=t_end` | 一致 |
| 完赛判定：x≥L−1 且 v≤0.3 | `env20.py` `arrived = x>=LENGTH-1 and v1<=0.3` | 一致 |
| 观测对截止时间是 Markov 的（同 (x,v,t) 不同 t_end 观测不同） | `verify.py` 第 6 项 | 本次实测 PASS |

## 3. 指令映射与原执行层（论文 III-C、式 (3)）

| 论文陈述 | 代码位置 | 核对结果 |
|---|---|---|
| u∈[−1,1]；a_cmd=2.6u (u>0)，3.5u (u≤0) | `td3_run.py` `act(u)=u*(A_HI if u>0 else -A_LO)` | 一致 |
| a_c=clip(a_cmd, a⁻±j_max Δt)，a=max{min{a_c,a_hi,a_safe},a_lo,−v/Δt}，j_max=2，Δt=0.5 | `env20.py` `project()`：a1 jerk 裁剪 → a2 执行器箱 → a3=min(a2,a_safe) → clip(a3, max(A_LO,−v/DT), A_HI) | 代数等价 |
| a_safe 由下游停止线/终点决定；弯道由包络处理（论文 VI 中 8 次运行 `curve=envelope`） | `a_safe()`、`_stop_targets()`；`CURVE_MODE='envelope'` 时弯道进入包络 | 一致（`bench_*.json` 的 `args.curve='envelope'`） |
| 限速在原层为软惩罚，外部硬夹在限速+5 m/s | `a_safe()` 末尾 `min(best,(V_FREE+V_OVER_HARD-v)/DT)`；`CLAMP_LIMIT=False` | 一致 |
| 策略每 2 s 决策，执行每 0.5 s 重算 | `repeat=4`，`DT=0.5`，`env_step()` | 一致 |

## 4. TD3 学习器（论文 III-C、附录 H-F/H-G/H-H）

| 论文陈述 | 代码位置 | 核对结果 |
|---|---|---|
| actor 13–64–64–1（ReLU，tanh 输出）；critic 14–64–64–1 | `td3_run.py` `MLP`, `HID=64` | 一致；归档 `.pt` 的张量形状 (64,13)/(64,14) 已核对 |
| actor lr 3e-5，critic lr 3e-4，batch 256，τ=0.005，actor 延迟 2，目标噪声 0.2/裁剪 0.5 | `alr=3e-5`, `CLR=3e-4`, `BATCH=256`, `TAU=0.005`, `POLICY_DELAY=2`, `target_noise=0.2`, `TARGET_CLIP=0.5` | 一致 |
| 多步目标 y=Σr+γⁿ(1−d)min Q′(o_{t+n},π′+ε)，n≤20，γ=1，终止时缩短 | `nq_` 滑动窗口（第 246–258 行）：满 20 个决策时弹出最早一个；episode 终止时按递减 n 清空 | 一致；行为轨迹回报，无重要性修正（论文已披露） |
| OU 探索 σ=0.2、θ=0.15；warmup 4000 积分步随机动作 N(0,0.6) | `noise='ou'`, `sigma=0.2`, `theta=0.15`, `WARMUP=4000` | 一致 |
| 回放 FIFO 125000 个决策转移 | `BUF=125000`，`ptr=(ptr+1)%BUF` | 一致 |
| 85% 重置用路内探索起点，随机合法速度，时钟按 12–22 m/s 配速；不可行停车状态回退为完整起点 | `mkenv()`：`x0=(L−300)(1−U)`, `v0=U(0,v_lim(x0))`, `t0=x0/pace`, `pace~U(12,22)`；`a_safe()<A_LO` 则 `reset()` | 一致 |
| 训练每次重置在三组电池工况中抽样；信号偏移独立均匀 | `PACK` 三元组；`Route20(rng=rng)` 每个信号独立 `U(0,90)` | 一致 |
| 评估 12 工况：三组电池 × 偏移 {0,22.5,45,67.5}，两信号共享偏移；评估从原点、80 km/h 起 | `evaluate()`：`offsets=[off]*NSIG`；`reset()` 置 `v=V_FREE` | 一致 |
| 500 万步、每 25 万步一个检查点（20 个）；验证偏移 {0,45} 选点，报告偏移 {22.5,67.5} 计分 | `run_12core.sh`：`--steps 5000000 --log 250000`；`analyze.py` `VAL_OFFSETS/REP_OFFSETS` | 一致 |
| 末 8 个检查点均值与样本 SD 为“late” | `long_route/analysis_metrics.py` `run['curve'][-8:]` | 一致；本次重算与 `recomputed_v17.json` 704 个数值全部一致 |
| 进度整形为 0 | `--shape 0.0` | 一致 |

## 5. 结果表格的重算（本次实际运行）

| 论文表 | 生成脚本 | 本次结果 |
|---|---|---|
| Table II（参考相对分解）、Table III（报告分数/late）、S1/S2/S4 | `long_route/analysis_metrics.py` | 与保留的 `recomputed_v17.json` 逐值一致（0 处差异） |
| Table III 各种子选中检查点的 12 工况网格 | `scripts/replay_selected_checkpoints.py` | 8/8 检查点回放网格与训练日志最大 |ΔR| = 0.0 |
| Table S3 屏蔽响应（selected 三列） | `scripts/channel_masks.py` | 8/8 种子的 (电池, SPaT, 弯道) 三元组与论文一致 |
| Table S3 late 列（末 8 检查点均值/SD） | `long_route/data/channel_late8.py` | 见 `results/channel_late8_regenerated.json`（与归档逐值一致） |
| Table F1 网络斜率诊断 | `long_route/data/critic_gradient2.py` | 与归档 `critic_gradient2.json` 共有键逐值一致（新版本多输出 `per_pos_slope`/`positive_count`/`n_positions` 三个键） |
| 附录 J-C：原到达截点下 A/B 平均 I_j 77.861 / 49.083 | `scripts/short_route_frozen_actor.py` | 18 条行程逐工况 I_j 误差 0.0 |

## 6. 与论文陈述不冲突、但会让复现者踩坑的地方

1. **`long_route/data/refs_fixed.json` 的环境指纹（b72133bb…）与冻结代码（04500faf…）不同。** 该文件带 `restamp_note`：后来的 env20.py 增加了训练专用 `K_FRIC/W_EVENT` 旋钮，在默认值下与冻结版逐步奖励一致；1861 个数值与 `evsim_v9/frozen/refs_fixed.json` 完全相同。训练器只读取 `evsim_v9/frozen/refs_fixed.json`，且 8 条归档曲线内嵌的指纹都是 04500faf…，所以不影响复现，但不要把 data/ 下这份复制进 frozen/。
2. **`evaluate_checkpoint.py` 要求运行 JSON 与检查点同目录**（`<tag>_s<seed>.json`），而发布包把权重放在 `weights/`、曲线放在 `data/bench_curves/`，直接调用会 `FileNotFoundError`。`scripts/replay_selected_checkpoints.py` 显式配对。
3. **12k 烟测的 last-3 = −394.262 依赖 `--log 6000`**（只有两个记录点，last-3 是两点均值）。用其它记录间隔时末检查点 R 仍为 −387.723，但 last-3 不同（4000→−391.138，3000→−397.521，2000→−398.149）。
4. **跨库版本重训不是位一致的。** 本环境（torch 2.14.0 / numpy 2.4.6）12k 步与文档一致，但到第一个 25 万步检查点时训练轨迹已与归档分叉（seed 0 归档记录步 250001，本次 250000；12 工况 R −198.07 对 −198.34）。因此“重跑得到论文每个种子的数字”在不同 torch/numpy 上不可期待；论文数字的精确复现路径是检查点回放，重训得到的是同一协议下的新样本。
5. **短程舒适性研究的 `Trainer.load` 硬性要求 torch/numpy 版本与断点内记录完全一致**（`ValueError('runtime mismatch')`），`rollout_layer.py` 也经由它取 actor。除 torch 2.8.0+cpu / numpy 2.3.5 外的环境无法直接运行 Table IV 的评估脚本。`scripts/short_route_frozen_actor.py` 只提取 actor 张量并复现评估，已验证 A 的张量与 SPaT 包 `actor_A.pt` 完全一致。
6. `TIV_comfort_v2/code/test_layer.py` 依赖 scipy（`linprog`），发布 requirements 只列了 numpy/torch。
7. `EVSIM_ROUTE`：`study.py` 在导入时把它设为 `mini`（4 km）；同一进程中不要既导入 `study` 又运行 20 km 代码。所有长程脚本先 `unset EVSIM_ROUTE`。
8. PyPI 上 `pip install torch` 在本机拉的是 cu130 轮子（2.5 GB）；代码全程 CPU，功能不受影响，但冻结包 README 建议的 CPU 索引在受限网络下可能不可达。Python 3.11 可运行（包内记录为 3.12）。
9. `run_batch.py` 拒绝已存在的 `--tag`，重跑必须换 tag；`td3_run.py` 单次评估（12 条完整行程）约 20–30 s，`--log` 太小会使评估时间占比过高。
