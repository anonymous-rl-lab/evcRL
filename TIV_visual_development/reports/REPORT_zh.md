# 视觉编码 Z × 平滑执行层 × TD3：开发阶段报告 v2（按独立审计 v1 修复后重跑）

日期：2026-09-12。v1 报告归档为 `REPORT_v1_zh.md`。本版对应 `TIV_Visual_Experiment_Audit_v1.md` 指出的问题逐条修复并重新运行全部阶段（预训练 → 审计 → 共同适配 → 三臂小规模），所有数字来自本版运行（`runs/pretrain/`、`runs/audit/`、`runs/common/`、`runs/pilot_v2/`）。

## 0. 对审计意见的逐条处理

| 审计条目 | 处理 | 核验位置 |
|---|---|---|
| 3. 检测框标签与 sigmoid 头不兼容（78% 正样本格点不可表示） | 框参数化改为“目标中心所在特征格内的归一化偏移 + 归一化尺寸”（`renderer.encode_box/decode_box`），框先裁剪到画面，中心出画面或裁剪后面积为 0 的目标不打标签；预训练脚本对 train/dev 全量断言四个分量 ∈ [0,1]；新增控制灯检测框 IoU、中心误差与命中率（按尺寸分层） | `runs/pretrain/pretrain_report.json: label_representability, dev_detection`；审计项 `box_labels_in_unit_range` |
| 4. 停车位置灯箱出画面而 `association_valid` 仍为 1；空 ROI 输出仅由偏置决定 | 灯箱改装于停止线远侧 14 m（路口对面常规安装位置），距停止线 2 m 时灯箱在画面内（ROI 299 px）；`association_valid` = 地图关联存在 且 投影 ROI 与画面相交（像素数>0），另记 `map_association`、`roi_pixels`、`visible`、`occluded`；ROI 为空的帧 ROI 头输出被规则强制为 unknown（常量 logits，无梯度），且不参与 ROI 头训练 | 审计项 `empty_roi_outputs_unknown`；`renderer.py` v2 说明 |
| 5. 小目标分层口径偏乐观（bloom 光斑 ≥2.6 px 却按名义灯箱高分层） | 每帧同时记录名义灯箱像素高 `housing_px_h` 与实际渲染光斑直径 `lamp_px_d`；所有灯色指标按两种分层并列报告；bloom 下限作为仿真假设在配置与报告中披露 | `visual_summary` 的 `housing_px_h:*` 与 `lamp_px_d:*` 两组分层 |
| 6. 视觉监督没有训练 Z 的末端投影 | 新增 `signal_z` 头：由 Z 预测最新帧控制灯色（含 unknown），加入视觉监督损失；预训练与审计都检查 `temporal` 投影的监督梯度非零；新增两项独立检查：Z 可解码性（dev 集线性探针）与同状态换图（同位姿/外观/噪声，强制红/绿）的 actor 指令响应 | 审计项 `vision_loss_reaches_z_projection`；`evaluation.json: z_decodability_dev, image_swap` |
| 6. “无增益全部归因执行层”过早 | 结论改写：分别报告“ROI 头识别”“Z 保留灯色”“actor 对图像的依赖”三件事，再讨论执行层主导 | 第 8 节 |
| 7. `executor_independent_of_arm` 直接赋 True | 改为实测：同一状态两个副本对 50 个随机指令输出相同，且 `project` 的代码不引用编码器/策略 | 审计项 `executor_independent_of_arm` |
| 7. 源码身份不含数据/权重哈希；恢复不比较 runtime；`fork_origin` 不在断点内 | `source_identity` 加入池文件、A 臂网络、预训练编码器的 SHA-256；`load()` 比较 torch/numpy 版本；`fork_origin` 进入断点 | `pipeline.py` |
| 7. `run_pilot.sh` 删除已有目录 | 改为拒绝覆盖已存在的 TAG，复用已有 `common.pt`，逐进程检查退出码 | `runs/run_pilot.sh` |
| 7. 缺少权重与固定审计样本 | 每臂导出全部 8 个网络的 `final_nets.pt`，共同起点导出 `common_nets.pt`，预训练编码器 `encoder.pt`，固定审计图像集 `audit_set.npz`，评估轨迹 NPZ 与逐帧视觉预测记录均入库 | `runs/pilot_v2/<arm>/`、`runs/common/`、`runs/pretrain/` |
| （本轮对抗式代码审查新增，3 视角 + 逐条反驳核验）热图焦点损失按全部格点归一化使正样本梯度份额仅 1.6e-5，定位学不到；改为按正样本归一化而不加先验偏置则初始损失 2086、共享特征塌缩 | 热图头偏置按先验 π=0.01 初始化 + 按正样本归一化（RetinaNet/CenterNet 配方）。受控实验（同种子 500 步）：原做法命中率 0.017；仅改归一化 0.008 且灯色头塌缩；先验+正样本归一化 0.788、中心误差 2.29 px；再加梯度裁剪 0.771/8.27 px（不采用） | `runs/pretrain/heat_loss_probe_*.json`、`detector_budget_probe_before_focal_fix_stdout.txt` |
| 标签格（floor(c/4)）与特征格采样中心 (4i,4j) 半格错位 | 格分配改为最近采样中心 floor(c/4+0.5)，格内偏移 (c−4j)/4+0.5；单像素脉冲测试确认响应格与采样中心一致；编码/解码往返误差 ≤3e-6 px | `renderer.encode_box/decode_box`、`model.decode_boxes` |
| 检测评估口径：IoU 分支对 1 px 目标不可达、命中判据实为“中心≤2 px”、(16,inf) 分层为空、0–4 分层吞掉 92% 样本、无误检度量、Z 头指标被 ROI 头占位污染 | 检测指标改为：格一致率、逐帧中心误差（均值/中位数）、≤2 px 命中、阈值召回（得分>0.5 且格正确）、无可见灯帧的误检率；分层 0–1/1–2/2–4/4–8/8–16 px；IoU 仅作辅助字段；Z 头只在最新帧计数 | `pretrain.evaluate_vision`、`pipeline._bin_stats` |
| 遮挡分支把全部类别 box_valid 清零（本路线无影响的潜在缺陷）；评估真值框未裁剪 | 只清灯所在格的框监督；真值框改用裁剪后的框 | `renderer._draw_light/_mark` |
| 事件注意力 att=Σσ(logits) 近似全局平均池化，并把 Z/critic 梯度耦合到热图头（审查测得耦合 25% 量级） | 本轮不改（属交接框架设计，改动会同时改变三臂的 RL→编码器路径）；作为已知限制记录在第 9 节 | — |
| 8. 数字与措辞 | 训练 episode 按“共同适配 2 个 + 分叉后各 20 个”表述；4 帧历史跨度 1.5 s（0.5 s 间隔）；分叉前只训练了 critic（503 次），actor 及其新增列未更新，不称“actor 已完成共同适配”；评估在到达后继续到静止并在 x≥3999 m、v≤0.3 时把指令置零，这是三臂相同的外部终端动作，不是 actor 学会的平顺停车；“轨迹完全一致”改为“汇总性能几乎相同（J/S 最大位置差 0.134 m）”；正式规模预算按各臂实测速率重算 | 第 4、8 节 |

## 1. 代码差异与保留边界

| 项目 | 处理 |
|---|---|
| 冻结源码 | `v19_deps/` 为 TIV_comfort_v1/code 六个文件、scn profile、comfort_v2 `curve_layer.py`（r2）、SPaT 包 `actor_A.pt` 的字节级快照，SHA-256 见 `v19_deps/PROVENANCE_SHA256.json`；未修改 |
| 共同 actor/critic 来源 | 从 `TIV_comfort_v1/runs/development/s7/A/resume.pt` 只提取 actor、q1、q2 六个网络张量；A 的 actor 与 SPaT 包 `actor_A.pt` 逐张量一致 |
| 框架代码 | `visual_z/model.py` 相对交接原件有三处修改（空 ROI 强制 unknown、`signal_z` 头与其损失、`decode_boxes`），其余五个文件逐字一致；`visual_z_framework_original/` 保留原件 |
| 动力学/奖励/截止 | `study.StudyEnv` 原样：L 型奖励、600 s 真实截止、t_budget 480、原动作映射 2.6u/3.5u |
| 执行层 | `SmoothEnv.project` 与 `TIV_comfort_v2/code/rollout_layer.py::CurveEnv.project` 逐行同义；模拟器已知信号时刻的访问权保持不变，三臂相同；每子步记录原始指令、实际动作、是否干预、是否回退 |
| 舒适性正则 | `lambda_c=0`（原 A 配置），三臂相同 |

## 2. 信息流表（camera + map 模式）

| 原下标 | 含义 | 本实现 |
|---|---|---|
| 0–6 | 车速、上一实际加速度、剩余路程、限速、弯道距离/限速、信号地图距离 | 原样进入 actor/critic |
| 7 | 仿真真值绿灯标记 | adapter 固定为 −1；灯色只能经图像 → Z 进入 |
| 8 | 仿真精确倒计时 | adapter 固定为 1（未知），V2X-valid=0 |
| 9–12 | SOC、温度、剩余任务时间、当前时间 | 原样 |
| Z（64） | 编码器输出 | 由最近 4 帧（0.5 s 间隔，跨度 1.5 s，2 Hz）、有效性与帧龄经融合层得到 |
| 元信息（4） | 最新帧有效、帧龄、控制灯关联有效（地图关联且 ROI 与画面相交）、V2X 有效 | 直接拼接 |

actor 输入 81 维，critic 82 维；旧 13 列与 critic 动作列按 `load_legacy_weights` 迁移，新列初始化为零。执行层不读取 Z，也不读取 actor 的任何中间量（审计实测）。

## 3. 相机与数据来源（真实接口的最小接入，v2）

- 输入只有原环境的位姿 x、时刻 t、道路几何与同一绝对时刻的信号相位；灯箱位于停止线远侧 14 m、右侧 3.5 m、离地 4.0 m，停止线本身仍在 3000 m。
- 透视投影（相机高 1.4 m，焦距 144 px，96×160）：车道线、弯道弯折与警示标志、终点停止线与立柱、信号灯杆/灯箱/三灯；亮灯发光半径下限 1.3 px（bloom，仿真假设），450 m 以外或灯箱出画面记为不可见。
- 逐 episode 随机：亮度、天空色板、雾长、传感器噪声、35% 概率树冠遮挡（灯前 90–220 m）。
- 标签：热图中心（灯/标志/终点）、格内偏移 + 尺寸的框回归、控制灯色 5 类、可见性、遮挡、名义灯箱高、光斑直径、真值框像素坐标；`signal_roi` 由地图距离与相机标定投影得到并裁剪到画面。
- 采样率如实为 2 Hz；分辨率为设计候选 384×640 的 1/4。真值不进入 actor（无 HUD/彩条），只进入标签与裁判。
- `capture(force_color=...)` 仅用于同状态换图检查，不用于训练。

## 4. 协议与实测速率（单线程 CPU，4 核）

（同 v1：有效 batch 32、监督批 8 序列、1 次更新/决策、n-step 20、γ=1、warmup 400 子步、回放 6000、学习率 3e-5/3e-4/1e-5、断点每 120 s。）

## 5. 审计（`runs/audit/audit.json`，19/19 通过；对应当前源码/数据身份）

原 16 项全部通过（真值通道屏蔽、加速度通道保留、81 维输入、actor/critic 迁移精确、未来帧/跨 episode 帧拒绝、回放动作为投影前指令、干预记录、三臂梯度路径、目标网络无梯度、断点续接一致）；其中联合臂 TD→编码器梯度在 20 次 critic 更新后为 19.8（分叉时刻为 0，符合零初始化预期），视觉监督→编码器 12.4。
v2 新增或改为实测的 4 项：`executor_independent_of_arm`（同一状态两个副本对 50 个随机指令输出相同，且 `project` 不引用编码器/策略）、`box_labels_in_unit_range`、`empty_roi_outputs_unknown`、`vision_loss_reaches_z_projection`（`temporal` 参数在视觉损失下梯度非零）。

## 6. 视觉预训练 v2（`runs/pretrain/pretrain_report.json`）

池：train 1600 / dev 240 / audit 32 个 4 帧序列（外观 episode 种子分离）；训练 1200 步 × 16 序列，0.171 s/步，共 205 s。
标签可表示性（审计口径复核）：train 2716 个、dev 405 个带框正样本格点，分量 < 0 与 > 1 的数量均为 **0**（v1 审计为 320/410 格点不可表示）。
监督梯度结构：`temporal_Z` 5.35、`signal_head` 8.38、`signal_z_head` 2.28、`backbone_c2` 22.7（v1 中 temporal_Z 为 0）。

dev 集控制灯色（有地图关联的 320 帧；已知类 305、unknown 15）：

| 分层 | n | ROI 头正确率 | 已知类正确率 | 红→绿 | 绿→非绿 | unknown 召回 |
|---|---|---|---|---|---|---|
| 名义灯箱高 <4 px | 294 | 0.935 | 0.968 | 0 | 3 | 0.167（n=12） |
| 名义灯箱高 4–8 px | 16 | 0.938 | 0.938 | 0 | 1 | — |
| 名义灯箱高 8–16 px | 7 | 0.714 | 0.714 | 1 | 1 | — |
| 名义灯箱高 ≥16 px | 3 | 0.667 | — | 0 | 0 | 0.667 |
| 实际光斑直径 0（未渲染亮灯：遮挡/出范围） | 15 | 0.267 | — | 0 | 0 | 0.267 |
| 实际光斑直径 (0,3] px（bloom 下限 2.6 px 起作用） | 303 | 0.964 | 0.964 | 0 | 5 | — |
| 实际光斑直径 3–6 px | 2 | 0.5 | 0.5 | 1 | 0 | — |
| 全部 | 320 | 0.928 | 0.961 | 1 | 5 | 0.267 |

Z 头（由 Z 直接预测最新帧灯色，dev 80 个序列）：正确率 0.925，已知类 0.973，红→绿 0，unknown 召回 0.2。

**如实说明**：(a) 303/320 帧的亮灯是 bloom 下限（直径 2.6 px）画出的光点，名义灯箱高 <1 px；“远距离小灯色正确率 0.96”只描述这套程序化光点渲染；(b) unknown 召回只有 0.27：遮挡帧（灯被树冠挡住）大多仍被判为某种颜色，两个头都如此，这是 v2 仍未解决的缺陷；(c) 控制灯检测框定位在本预算下没有学会：dev 305 个可见灯的解码框平均 IoU 0.009，中心误差 34 px，命中率（IoU>0.3 或中心误差 ≤2 px）12%。灯色识别之所以可用，是因为 ROI 由地图投影给出，不依赖检测器；因此**本实验不能主张“小目标检测已通过”**。`runs/pretrain/detector_budget_probe.json` 记录了在同一编码器上继续预训练 2400 步时命中率的变化，用于区分预算不足与实现错误（见第 8 节）。

## 7. 共同适配与三臂小规模（`runs/common/`、`runs/pilot_v2/`）

（运行后填写。）

## 8. 结论

（运行后填写。）

## 9. 未完成与已知限制

- 无真实相机、无外部 3D 引擎：渲染器是程序化透视场景，类别只有灯/标志/终点，没有行人、车辆、锥桶等类别，也没有碰撞任务。
- 无预训练骨干权重；编码器为交接包原型，随机初始化后在渲染帧上监督预训练。
- 2 Hz、96×160、batch 32、单机 CPU；未测 10 Hz/384×640 的实时预算，未做异步双进程服务。
- 执行层仍使用模拟器真值信号时刻（与交接文档首阶段一致），零违规不能归因于视觉。
- bloom 光斑下限是仿真假设，无相机标定支持；“光斑直径”分层只描述这套程序化渲染。
- 事件注意力近似全局平均池化，且把 Z/critic 的梯度耦合进热图头；检测框尺寸头在 1 px 量级目标上未学到有意义的尺寸；unknown（遮挡）召回偏低。
- 单开发种子、单条路线、小预算。
