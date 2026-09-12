"""绕过短程舒适性研究的运行时硬校验：只提取冻结 actor 权重，并复现其九个开发工况的评估。

问题：TIV_comfort_v1/code/study.py 的 Trainer.load 要求 torch/numpy 版本与 resume.pt 内记录
（torch 2.8.0+cpu、numpy 2.3.5）逐字相同，否则抛 ValueError('runtime mismatch')；
TIV_comfort_v2/code/rollout_layer.py 只为取 actor 也走这条路，因此在其他版本上整套短程评估无法运行。
本脚本从完整断点中只取 actor 的六个张量（与 SPaT 实验 weights/actor_A.pt 的做法一致），
校验源码哈希与 SPaT 包内 actor_A.pt 张量一致，然后用原 study.rollout 复现 A/B 两个 actor 的
九个开发工况（原到达截点），与归档 evaluations/step_300000_development.json 逐工况比对。
用法：TIV_BASE=/path/TIV_v19_Reproducibility python scripts/short_route_frozen_actor.py
"""
import json, os, sys
from pathlib import Path
base = os.environ.get('TIV_BASE')
if not base:
    sys.exit('请设置 TIV_BASE 指向已解压的 TIV_v19_Reproducibility 目录')
BASE = Path(base).resolve()
sys.path.insert(0, str(BASE / 'TIV_comfort_v1' / 'code'))
import numpy as np, torch
import study as S   # 导入时设置 EVSIM_ROUTE=mini；本脚本不能与 20 km 代码同进程使用
OUT = Path(__file__).resolve().parents[1] / 'results' / 'short_route'; OUT.mkdir(parents=True, exist_ok=True)
spat = torch.load(BASE / 'TIV_SPaT_Execution_v1' / 'weights' / 'actor_A.pt', map_location='cpu', weights_only=True)['actor']
expected = {'A': 77.86137616330205, 'B': 49.083}   # 论文附录 J-C：原到达截点下 A/B 的平均 I_j
report = {}
for arm in ('A', 'B'):
    ck = BASE / 'TIV_comfort_v1' / 'runs' / 'development' / 's7' / arm / 'resume.pt'
    s = torch.load(ck, map_location='cpu', weights_only=False)   # 只信任本地研究断点（含 RNG 对象）
    assert s['schema'] == 1 and s['code_hash'] == S.source_hash(), '源码/协议哈希不一致'
    actor_sd = {k: v.clone() for k, v in s['networks']['actor'].items()}
    torch.save({'actor': actor_sd, 'origin': str(ck.relative_to(BASE)), 'runtime_of_checkpoint': s['runtime'],
                'used_steps': s['used'], 'lambda_c': s['cfg']['lambda_c']}, OUT / f'actor_{arm}.pt')
    actor = S.MLP(13, True); actor.load_state_dict(actor_sd); actor.eval()
    if arm == 'A':
        same = all(torch.equal(actor_sd[k], spat[k]) for k in spat)
        print(f"A 的 actor 张量与 SPaT 包 weights/actor_A.pt {'完全一致' if same else '不一致'}")
    archived = json.load(open(ck.parent / 'evaluations' / 'step_300000_development.json'))
    rows = []
    for i, cond in enumerate(S.conditions('development')):
        m, _ = S.rollout(actor, *cond); rows.append(m)
        a = archived['rows'][i]
        print(f"{arm} 工况 {i}: I_j {m['Ij']:.6f} (归档 {a['Ij']:.6f}) 时间 {m['time_s']:.1f} 能耗 {m['E_Wh']:.3f} 完赛 {m['arrived']} 违规 {m['violations']}")
    mean_ij = float(np.mean([r['Ij'] for r in rows])); max_err = max(abs(r['Ij'] - a['Ij']) for r, a in zip(rows, archived['rows']))
    report[arm] = dict(mean_Ij=mean_ij, archived_mean_Ij=archived['summary']['completed_mean']['Ij'], max_abs_Ij_error=max_err,
                       checkpoint_runtime=s['runtime'], lambda_c=s['cfg']['lambda_c'], used_steps=s['used'])
    print(f"{arm}: 平均 I_j {mean_ij:.3f}（论文 {expected[arm]}，归档 {archived['summary']['completed_mean']['Ij']:.3f}），逐工况最大误差 {max_err:.2e}\n")
report['runtime_here'] = {'torch': torch.__version__, 'numpy': np.__version__}
json.dump(report, open(OUT / 'frozen_actor_replay.json', 'w'), indent=1)
print('已写入', OUT)
