"""热图损失变体受控实验：定位“检测框定位学不会”的根因。同种子、同数据、同步数，只改热图损失归一化/先验偏置/梯度裁剪。
变体：all=按全部格点归一化（v2a）；pos=按正样本归一化（v2b）；pos_prior=pos+热图头偏置先验 π=0.01；pos_prior_clip=pos_prior+梯度范数裁剪 10。
用法：python visual_dev/heat_loss_probe.py --variant pos_prior_clip --steps 500
"""
import argparse, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
from torch.nn import functional as F
import pipeline as P
from pipeline import pool_observation, labels_from_frames
from pretrain import evaluate_vision
from visual_z.model import VisualEncoder, decode_boxes
from renderer import encode_box, decode_box
torch.set_num_threads(1)


def heat_focal(output, labels, obs, norm):
    y = labels['heat']; logit = output['heat']; p = logit.sigmoid()
    mask = labels['heat_valid'] * obs.valid[:, :, None, None, None]
    pt = p * y + (1 - p) * (1 - y); alpha = .25 * y + .75 * (1 - y)
    focal = alpha * (1 - pt).square() * F.binary_cross_entropy_with_logits(logit, y, reduction='none')
    denom = mask.expand_as(focal).sum() if norm == 'all' else (y * mask).sum()
    return (focal * mask).sum() / denom.clamp_min(1)


def other_terms(output, labels, obs):
    m = labels['box_valid'] * obs.valid[:, :, None, None, None]
    box = (F.smooth_l1_loss(output['boxes'], labels['boxes'], reduction='none') * m).sum() / m.expand_as(output['boxes']).sum().clamp_min(1)
    ys = labels['signal'].clone(); ys[~obs.valid] = -100; ys[output['roi_empty']] = -100
    sig = F.cross_entropy(output['signal'].reshape(-1, 5), ys.reshape(-1), ignore_index=-100, reduction='sum') / (ys != -100).sum().clamp_min(1)
    yz = labels['signal'][:, -1].clone(); yz[~obs.valid[:, -1]] = -100
    sz = F.cross_entropy(output['signal_z'], yz, ignore_index=-100, reduction='sum') / (yz != -100).sum().clamp_min(1)
    return box + sig + sz


def selftest_decode():
    """解码自检：把真值框编码进 boxes/heat 张量后解码，中心误差应为 0。"""
    box = (83.9, 37.4, 84.9, 39.0); i, j, tgt, clipped = encode_box(box)
    heat = torch.full((1, 1, 8, 24, 40), -9.); heat[0, 0, 0, i, j] = 9.; boxes = torch.zeros(1, 1, 4, 24, 40); boxes[0, 0, :, i, j] = torch.as_tensor(tgt)
    dec, _ = decode_boxes(heat, boxes, 0); d = dec[0, 0].tolist()
    err = math.hypot((d[0] + d[2]) / 2 - (clipped[0] + clipped[2]) / 2, (d[1] + d[3]) / 2 - (clipped[1] + clipped[3]) / 2)
    assert err < 1e-4, err; return err


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True, choices=('all', 'pos', 'pos_prior', 'pos_prior_clip')); ap.add_argument('--steps', type=int, default=500)
    a = ap.parse_args(); v = a.variant
    print('解码自检中心误差', selftest_decode(), flush=True)
    pool = torch.load(ROOT / 'runs/pretrain/pool.pt', map_location='cpu', weights_only=False)['sequences']
    dev = torch.load(ROOT / 'runs/pretrain/eval_sets.pt', map_location='cpu', weights_only=False)['dev']
    torch.manual_seed(4242); enc = VisualEncoder(4)
    if 'prior' in v: torch.nn.init.constant_(enc.heat.bias, -math.log((1 - .01) / .01))
    opt = torch.optim.Adam(enc.parameters(), lr=3e-4); rng = np.random.default_rng(4242); log = []; t0 = time.monotonic()
    for step in range(1, a.steps + 1):
        seqs = [pool[i] for i in rng.integers(0, len(pool), 16)]; obs = pool_observation(seqs); lab = labels_from_frames(seqs)
        out = enc(obs); hf = heat_focal(out, lab, obs, 'all' if v == 'all' else 'pos'); loss = hf + other_terms(out, lab, obs)
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(enc.parameters(), 10. if 'clip' in v else 1e9)); opt.step()
        if step in (1, 50) or step % 250 == 0:
            enc.eval(); s, d, _ = evaluate_vision(enc, dev[:120]); enc.train()
            with torch.no_grad(): p = out['heat'].sigmoid(); ppos = float(p[lab['heat'] > 0].mean()); pneg = float(p[lab['heat'] == 0].mean())
            rec = dict(step=step, loss=float(loss), heat_term=float(hf), grad_norm=gn, roi_acc=s['all']['roi_head']['acc'], roi_known_acc=s['all']['roi_head']['known_acc'],
                       det_hit=d['all']['hit_rate'], det_center_err=float(np.mean([b['mean_center_err_px'] for k, b in d.items() if k != 'all'])) if len(d) > 1 else None,
                       p_pos=ppos, p_neg=pneg, wall_s=time.monotonic() - t0)
            log.append(rec); print(json.dumps(rec), flush=True)
    json.dump(dict(variant=v, log=log), open(ROOT / 'runs/pretrain' / f'heat_loss_probe_{v}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
