"""固定策略的观测通道屏蔽（论文 Table S3 “selected” 三列）：环境感知通道的响应测量。

在报告集 6 个工况上，把电池通道替换为暖电池编码、SPaT 通道替换为超出广播范围编码、
弯道通道替换为无预告编码，比较回报变化。这是“屏蔽响应”，不是信息价值，也不是重训消融。
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as P
P.use_long_route()
import env20 as E, route20 as R
from td3_run import MLP, env_step
E.REWARD_MODE = 'L'; E.CURVE_MODE = 'envelope'; E.SHAPE_C = 0.; E.CURVE_ENVELOPE = True
PACK = [(0.85, 288.15), (0.90, 263.15), (0.95, 263.15)]; OFFS = (22.5, 67.5)
WARM = {9: (0.85 - 0.5) / 0.5, 10: (288.15 - 283.15) / 30.0}   # 电池通道：暖电池 0.85/+15°C 的编码
BLIND = {6: 1.0, 7: -1.0, 8: 1.0}                                # SPaT 通道：超出 1000 m 广播范围的编码
NOCURVE = {4: 1.0, 5: 1.0}                                       # 弯道通道：400 m 预告内无弯道的编码
MASKS = {'none': {}, 'battery': WARM, 'spat': BLIND, 'curve': NOCURVE, 'battery+spat': {**WARM, **BLIND}}

def rollout(actor, env, mask, k=4):
    o = env.reset(); Rr = 0.
    while True:
        oo = np.array(o, dtype=np.float32)
        for i, v in mask.items(): oo[i] = v
        with torch.no_grad(): u = float(actor(torch.as_tensor(oo).unsqueeze(0))[0, 0])
        o, r, d, i = env_step(env, u, k); Rr += r
        if d: break
    return Rr, bool(i['arrived'])

def score(actor, mask):
    rows = [rollout(actor, E.Route20(s, T, offsets=[o] * len(R.SIGNALS)), mask) for s, T in PACK for o in OFFS]
    return float(np.mean([r for r, _ in rows])), sum(a for _, a in rows)

results = {r['args']['seed']: r for r in json.load(open(P.REFERENCE / 'bench_results.json'))}
paper_s3 = {0: (-2.26, -2.84, -0.89), 1: (-1.35, -2.12, 0.67), 2: (0.81, -2.22, -1.59), 3: (-1.71, -3.46, 0.47),
            4: (-0.87, 2.48, -0.40), 5: (0.42, -0.51, 3.06), 6: (2.66, 0.70, 0.05), 7: (-4.44, -28.55, -1.43)}
out = {}; ok = True
print(f"{'种子':>4} {'基线R':>8} | {'屏蔽电池':>13} {'屏蔽SPaT':>13} {'屏蔽弯道':>13} {'电池+SPaT':>13} | 论文 Table S3（电池, SPaT, 弯道）")
for seed in sorted(results):
    step = results[seed]['selected_step']
    ck = torch.load(P.REFERENCE / 'selected_checkpoints' / f'ckpt_bench_s{seed}_{step}steps.pt', map_location='cpu', weights_only=True)
    actor = MLP(E.OBS_DIM, 1, True); actor.load_state_dict(ck['actor']); actor.eval()
    base, _ = score(actor, {}); rec = dict(baseline=base)
    line = f"s{seed:<3} {base:8.2f} |"
    for m in list(MASKS)[1:]:
        r, a = score(actor, MASKS[m]); rec[m] = dict(R=r, arrivals=a, delta=r - base)
        line += f" {r-base:+8.2f}({a}/6)"
    out[f's{seed}'] = rec
    deltas = tuple(round(rec[m]['delta'], 2) for m in ('battery', 'spat', 'curve'))
    ok &= deltas == paper_s3[seed]
    print(line + f" | {paper_s3[seed]}  {'一致' if deltas == paper_s3[seed] else '不一致'}", flush=True)
json.dump(out, open(P.RESULTS / 'channel_masks_selected.json', 'w'), indent=1)
print('Table S3 selected 三列', '已复现' if ok else '未复现')
