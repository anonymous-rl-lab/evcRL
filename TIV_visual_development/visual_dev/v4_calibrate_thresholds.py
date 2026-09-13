"""v4：在 dev 集上按类别校准检测门控阈值（存在性分数 = 校准后的热图峰值）。
输出各阈值下的召回（格命中 ±1 且分数≥阈值）/ 无目标帧误检率，并推荐“误检 ≤ fa_max 下召回最高”的阈值；写入 runs/pretrain_v4/thresholds.json。
用法：python visual_dev/v4_calibrate_thresholds.py [--encoder runs/pretrain_v4/encoder.pt] [--fa-max 0.01]"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P
from pipeline import pool_observation
from visual_z.model import VisualEncoder, decode_detections
from renderer import CLASSES
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--encoder', default='runs/pretrain_v4/encoder.pt'); ap.add_argument('--fa-max', type=float, default=0.01); ap.add_argument('--out', default=None)
    a = ap.parse_args(); enc_path = ROOT / a.encoder
    ck = torch.load(enc_path, map_location='cpu', weights_only=False); enc = VisualEncoder(4); enc.load_state_dict(ck['encoder']); enc.eval()
    dev = torch.load(enc_path.parent / 'eval_sets.pt', map_location='cpu', weights_only=False)['dev']
    pos = {c: [] for c in range(4)}; neg = {c: [] for c in range(4)}; cellok = {c: [] for c in range(4)}; derr = {c: [] for c in range(4)}
    for i in range(0, len(dev), 16):
        chunk = dev[i:i + 16]; obs = pool_observation(chunk)
        with torch.no_grad(): out = enc(obs)
        for c in range(4):
            d = decode_detections(out, c)
            for j, seq in enumerate(chunk):
                for k, f in enumerate(seq):
                    hm = f['labels']['heat'][c]; cells = np.argwhere(hm > 0); sc = float(d['score'][j, k])
                    if len(cells):
                        ti, tj = cells[0]; pi, pj = d['cell'][j, k].tolist(); ok = abs(pi - ti) <= 1 and abs(pj - tj) <= 1
                        pos[c].append(sc); cellok[c].append(ok)
                        if ok: derr[c].append(abs(float(d['dist_m'][j, k]) - float(f['labels']['dist'][0, ti, tj]) * 400))
                    else: neg[c].append(sc)
    thrs = np.round(np.arange(0.05, 0.96, 0.05), 2); rec = {}; table = {}
    for c in range(4):
        p = np.array(pos[c]); n = np.array(neg[c]); co = np.array(cellok[c]); rows = {}
        for t in thrs: rows[float(t)] = dict(recall=float(((p >= t) & co).mean()) if len(p) else None, false_alarm=float((n >= t).mean()) if len(n) else None)
        feasible = [t for t, r in rows.items() if r['false_alarm'] is not None and r['false_alarm'] <= a.fa_max and r['recall'] is not None]
        best = max(feasible, key=lambda t: rows[t]['recall']) if feasible else 0.5
        rec[CLASSES[c]] = best; table[CLASSES[c]] = dict(n_pos=len(p), n_neg=len(n), cell_hit_rate=float(co.mean()) if len(co) else None, dist_mae_when_hit=float(np.mean(derr[c])) if derr[c] else None, rows=rows, recommended=best,
                                                     recall_at_recommended=rows[best]['recall'], false_alarm_at_recommended=rows[best]['false_alarm'])
        print(f"{CLASSES[c]:14s} 正 {len(p):>4} 负 {len(n):>4} 格命中 {co.mean() if len(co) else float('nan'):.3f} 距离MAE {np.mean(derr[c]) if derr[c] else float('nan'):.1f} | 推荐阈值 {best} 召回 {rows[best]['recall']:.3f} 误检 {rows[best]['false_alarm']:.4f} | " + ' '.join(f"{t}:{rows[t]['recall']:.2f}/{rows[t]['false_alarm']:.3f}" for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)), flush=True)
    out = dict(encoder=str(a.encoder), fa_max=a.fa_max, recommended=rec, table=table)
    json.dump(out, open(ROOT / (a.out or (enc_path.parent / 'thresholds.json')), 'w'), indent=1, ensure_ascii=False); print('推荐阈值', rec)


if __name__ == '__main__':
    main()
