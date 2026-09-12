"""事件注意力探针：在已训练编码器上度量 att=Σσ(heat logits) 中“真值灯格”的份额与 att 的非均匀程度。
对应审查发现“event 分支近似全局平均池化、不携带检出格定位信息”。输出 runs/probes/attention_share.json。
用法：python visual_dev/attention_share_probe.py [--encoder runs/pretrain/encoder.pt]"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
from pipeline import pool_observation
from visual_z.model import VisualEncoder
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--encoder', default='runs/pretrain/encoder.pt'); ap.add_argument('--out', default='runs/probes/attention_share.json')
    a = ap.parse_args()
    ck = torch.load(ROOT / a.encoder, map_location='cpu', weights_only=False); enc = VisualEncoder(4); enc.load_state_dict(ck['encoder']); enc.eval()
    dev = torch.load(ROOT / 'runs/pretrain/eval_sets.pt', map_location='cpu', weights_only=False)['dev']
    shares, ratios, ents, gap_rel = [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(dev), 16):
            seqs = dev[s0:s0 + 16]; obs = pool_observation(seqs); b, t, c, h, w = obs.frames.shape
            x = obs.frames.float().reshape(b * t, c, h, w) / 255.
            c2 = enc.c2(enc.stem(x)); c3 = enc.c3(c2); c4 = enc.c4(c3)
            p4 = enc.lat4(c4); p3 = enc.lat3(c3) + torch.nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest')
            p2 = enc.lat2(c2) + torch.nn.functional.interpolate(p3, size=c2.shape[-2:], mode='nearest')
            att = enc.heat(p2).sigmoid().sum(1) + 1e-6                                   # [b*t, 24, 40]
            event = (p2 * att.unsqueeze(1)).sum((2, 3)) / att.sum((1, 2)).unsqueeze(1); gap = p2.mean((2, 3))
            att = att.reshape(b, t, *att.shape[-2:]); event = event.reshape(b, t, -1); gap = gap.reshape(b, t, -1)
            for n, seq in enumerate(seqs):
                for k, fr in enumerate(seq):
                    L = fr['labels']
                    if not L.get('visible') or L.get('light_box_px') is None: continue
                    l, tp, r, bt = L['light_box_px']; cx, cy = (l + r) / 2, (tp + bt) / 2; i, j = int(cy // 4), int(cx // 4)
                    m = att[n, k]; tot = float(m.sum()); shares.append(float(m[i, j]) / tot); ratios.append(float(m.max()) / float(m.mean()))
                    p = (m / tot).flatten(); ents.append(float(-(p * p.log()).sum() / np.log(p.numel())))
                    gap_rel.append(float((event[n, k] - gap[n, k]).norm() / gap[n, k].norm().clamp_min(1e-6)))
    out = dict(n_visible_frames=len(shares), true_cell_att_share_mean=float(np.mean(shares)), true_cell_att_share_max=float(np.max(shares)),
               att_max_over_mean=float(np.mean(ratios)), att_normalized_entropy=float(np.mean(ents)), event_vs_gap_relative_diff=float(np.mean(gap_rel)),
               note='份额=真值灯格 att / 全帧 att 之和；均匀池化时份额=1/960=0.00104、熵=1、event 与 GAP 相对差=0')
    (ROOT / 'runs/probes').mkdir(parents=True, exist_ok=True); json.dump(out, open(ROOT / a.out, 'w'), indent=1, ensure_ascii=False); print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
