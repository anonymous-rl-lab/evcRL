# TIV Visual Z Framework v1

给 Codex 的视觉—TD3 双向学习开发交接包。基于 v19 接口，独立保存，不修改论文及其冻结证据。

先读 `DESIGN_ZH.md`（完整设计），再读 `CODEX_TASK.md`（执行顺序）。`configs/pilot.json` 为候选协议，不是已完成的预注册或实验结果。

| 文件 | 内容 |
|---|---|
| `visual_z/contracts.py` | 观测/转移/相机协议，13+64+4 输入定义 |
| `visual_z/model.py` | 随机初始化多尺度原型、Z、检测监督、原通道屏蔽、旧权重迁移 |
| `visual_z/learner.py` | 三种编码更新模式、目标编码器、TD3梯度、独立视觉监督batch |
| `visual_z/replay.py` | 无损帧与转移索引原型，时间/episode检查 |
| `visual_z/checkpoint.py` | 模型/优化器/RNG与外部全状态保存接口 |
| `tests/test_contracts.py` | 梯度、权重、泄漏、回放及恢复测试 |
| `verification/` | 实际测试结果和核对过的 v19 源码身份 |

Python 3.10+（当前在 Python 3.12 / Torch 2.8.0+cpu 验证）。从本目录运行：

```bash
python -m unittest discover -s tests -v
```

可选指定已有 v19 解压目录，以增加真实旧 actor 与源码检查：

```bash
TIV_BASE=/absolute/path/TIV_v19_Reproducibility python -m unittest discover -s tests -v
```

PowerShell 对应设置：

```powershell
$env:TIV_BASE = 'D:\research\TIV_v19_Reproducibility'
python -m unittest discover -s tests -v
```

本包不含摄像头 renderer、不含视觉训练数据、不含预训练骨干；相机空接口明确抛出 `NotImplementedError`，不会播放假数据并假报闭环已运行。网络和 learner 可以在测试张量上执行，用于验证连接和反传，不代表检测精度、实时性或实际驾驶性能。

保留原执行器时，它的真值信息仍可能防止错误动作。第一阶段评估固定守护器条件下的视觉表示，不能把零违反直接归因于摄像头。camera+map 模式不读取 actor 原精确倒计时；未来 V2X 需独立标识。

生产接入、10 分钟烟测、1 小时小规模和正式规模的测算要求已写在交接文件中。优先复用已有资产；此包不启动新场景训练。
