"""检测器预算探针：在预训练编码器上继续同一监督损失若干步，只观察控制灯检测框命中率/IoU/中心误差随预算的变化。
用途：区分“定位没学会是预算不足”还是“实现错误”。不改变三臂使用的共同起点，不写回 encoder.pt。"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P
from pipeline import pool_observation, labels_from_frames
from pretrain import evaluate_vision
from visual_z.model import VisualEncoder, perception_loss
torch.set_num_threads(1)
pool = torch.load(ROOT / 'runs/pretrain/pool.pt', map_location='cpu', weights_only=False)['sequences']
dev = torch.load(ROOT / 'runs/pretrain/eval_sets.pt', map_location='cpu', weights_only=False)['dev']
enc = VisualEncoder(4); enc.load_state_dict(torch.load(ROOT / 'runs/pretrain/encoder.pt', map_location='cpu', weights_only=False)['encoder'])
opt = torch.optim.Adam(enc.parameters(), lr=3e-4); rng = np.random.default_rng(99); log = []; t0 = time.monotonic()
enc.eval(); s, d, _ = evaluate_vision(enc, dev); log.append(dict(extra_steps=0, detection=d, roi_all=s['all']['roi_head'])); enc.train()
print('额外 0 步：', json.dumps(d['all']), flush=True)
for step in range(1, 2401):
    idx = rng.integers(0, len(pool), 16); seqs = [pool[i] for i in idx]; obs = pool_observation(seqs); lab = labels_from_frames(seqs)
    loss = perception_loss(enc(obs), lab, obs); opt.zero_grad(); loss.backward(); opt.step()
    if step % 600 == 0:
        enc.eval(); s, d, _ = evaluate_vision(enc, dev); enc.train()
        log.append(dict(extra_steps=step, loss=float(loss), detection=d, roi_all=s['all']['roi_head'], wall_s=time.monotonic() - t0))
        print(f'额外 {step} 步：', json.dumps(d['all']), 'ROI acc', round(s['all']['roi_head']['acc'], 3), 'unknown 召回', s['all']['roi_head']['unknown_recall'], flush=True)
json.dump(dict(note='仅诊断预算；不改变三臂共同起点', log=log), open(ROOT / 'runs/pretrain/detector_budget_probe.json', 'w'), indent=1, ensure_ascii=False)
