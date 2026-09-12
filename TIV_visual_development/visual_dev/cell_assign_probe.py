"""格分配规则受控对比：floor vs nearest，各自重新生成标注池（800 训练 / 120 dev 序列），同种子训练 600 步，报告格一致率、中心误差、≤2 px 命中。
用法：RENDER_CELL_ASSIGN=floor python visual_dev/cell_assign_probe.py --seed 1"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P
from pipeline import pool_observation, labels_from_frames
from pretrain import make_sequences, compact, evaluate_vision
from visual_z.model import VisualEncoder, perception_loss
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--steps', type=int, default=600); a = ap.parse_args()
rule = os.environ.get('RENDER_CELL_ASSIGN', 'floor')
train = make_sequences(800, 4242, 'tr'); dev = make_sequences(120, 4342, 'dv'); compact(train); compact(dev)
torch.manual_seed(a.seed); enc = VisualEncoder(4); opt = torch.optim.Adam(enc.parameters(), lr=3e-4); rng = np.random.default_rng(a.seed); t0 = time.monotonic(); log = []
for step in range(1, a.steps + 1):
    seqs = [train[i] for i in rng.integers(0, len(train), 16)]; obs = pool_observation(seqs); lab = labels_from_frames(seqs)
    loss = perception_loss(enc(obs), lab, obs); opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0:
        enc.eval(); s, d, _ = evaluate_vision(enc, dev); enc.train()
        rec = dict(rule=rule, seed=a.seed, step=step, cell_ok=d['all']['cell_ok_rate'], center_err=d['all']['mean_center_err_px'], median_err=d['all']['median_center_err_px'], hit_le2=d['all']['hit_le2px'],
                   recall_thr=d['all']['recall_at_thr'], false_alarm=d['no_visible_light_frames']['false_alarm_rate'], roi_known=s['all']['roi_head']['known_acc'], wall=round(time.monotonic() - t0, 1))
        log.append(rec); print(json.dumps(rec), flush=True)
(ROOT / 'runs/probes').mkdir(parents=True, exist_ok=True)
json.dump(log, open(ROOT / 'runs/probes' / f'cell_assign_probe_{rule}_s{a.seed}.json', 'w'), indent=1)
