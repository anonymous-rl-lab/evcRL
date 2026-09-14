"""v6：让已部署检测器认识“灯箱可见但三灯全灭”（off 类）。原 v4 预训练池没有灭灯帧，灯色头把灭灯判为绿（实测 60% 以上），
无法用“只隐藏灯色”做定理结构的延迟实验。做法：从 pretrain_v4 的池取原序列 + 新生成灭灯序列（light_color 隐藏，标签 signal=off），
以低学习率对整个编码器混合微调，验证：(1) 灭灯 dev：检出率与 off 正确率；(2) 原 dev 的 v4 检测/距离/灯色指标不明显退化；输出 runs/pretrain_v4_off/encoder.pt。
用法：python visual_dev/v6_finetune_off.py [--steps 800 --lr 5e-5 --n-off 1200]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
torch.set_num_threads(2)
import pipeline as P
from pipeline import pool_observation, labels_from_frames, STACK
from pretrain import make_sequences, compact, evaluate_vision_v4, evaluate_vision, roi_head_by_distance
from visual_z.model import VisualEncoder, perception_loss, decode_detections
from renderer import CLASSES


def off_metrics(enc, seqs, thr=0.4):
    """灭灯序列：灯在画面内（labels visible=1）的最新帧上，检出率（score≥thr）与灯色头判 off 的比例。"""
    n = det = off = 0; by = {}
    for i in range(0, len(seqs), 16):
        chunk = seqs[i:i + 16]; obs = pool_observation(chunk)
        with torch.no_grad(): out = enc(obs); pr = out['signal'][:, -1].softmax(-1); d = decode_detections(out, CLASSES.index('traffic_light'))
        for j, seq in enumerate(chunk):
            L = seq[-1]['labels']
            if not L.get('visible'): continue
            n += 1; ok = float(d['score'][j, -1]) >= thr; det += ok; o = int(pr[j].argmax()) == 3; off += o
            b = int(min(L['light_distance_m'] // 50, 4)); by.setdefault(b, [0, 0, 0]); by[b][0] += 1; by[b][1] += ok; by[b][2] += o
    return dict(n=n, det_rate=det / max(n, 1), off_rate=off / max(n, 1), by_50m={f'{k*50}-{k*50+50}': dict(n=v[0], det=v[1] / v[0], off=v[2] / v[0]) for k, v in sorted(by.items())})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=800); ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--n-off', type=int, default=1200); ap.add_argument('--batch', type=int, default=16); ap.add_argument('--seed', type=int, default=606); ap.add_argument('--out', default='pretrain_v4_off'); ap.add_argument('--init', default='runs/pretrain_v4/encoder.pt', help='起点权重（可从上一轮微调继续）'); ap.add_argument('--off-frac', type=float, default=0.5, help='每批灭灯序列占比')
    a = ap.parse_args(); OUT = ROOT / 'runs' / a.out; OUT.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    base = torch.load(ROOT / a.init, map_location='cpu', weights_only=False)
    pool = torch.load(ROOT / 'runs/pretrain_v4/pool.pt', map_location='cpu', weights_only=False)['sequences']; es = torch.load(ROOT / 'runs/pretrain_v4/eval_sets.pt', map_location='cpu', weights_only=False); dev = es['dev']
    off_tr = make_sequences(a.n_off, a.seed, 'of', 'v4', hide=('light_color',)); off_dev = make_sequences(240, a.seed + 100, 'od', 'v4', hide=('light_color',)); compact(off_tr); compact(off_dev)
    print(f'池：原 {len(pool)} 序列，灭灯 {len(off_tr)}（含灯帧 {sum(s[-1]["labels"].get("visible", 0) for s in off_tr)}），灭灯 dev {len(off_dev)}，生成 {time.monotonic() - t0:.0f}s', flush=True)
    enc = VisualEncoder(STACK); enc.load_state_dict(base['encoder']); enc.eval()
    before = dict(off=off_metrics(enc, off_dev), v4=evaluate_vision_v4(enc, dev)); print('微调前 灭灯 dev:', json.dumps(before['off'], ensure_ascii=False)[:300], flush=True)
    opt = torch.optim.Adam(enc.parameters(), lr=a.lr); rng = np.random.default_rng(a.seed); enc.train()
    for step in range(1, a.steps + 1):
        n_off = int(round(a.batch * a.off_frac)); i1 = rng.integers(0, len(pool), a.batch - n_off); i2 = rng.integers(0, len(off_tr), n_off); seqs = [pool[i] for i in i1] + [off_tr[i] for i in i2]
        obs = pool_observation(seqs); labels = labels_from_frames(seqs); loss = perception_loss(enc(obs), labels, obs); opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0: enc.eval(); m = off_metrics(enc, off_dev[:120]); enc.train(); print(f'step {step} loss {float(loss):.3f} 灭灯 检出 {m["det_rate"]:.2f} off 率 {m["off_rate"]:.2f}', flush=True)
    enc.eval(); after = dict(off=off_metrics(enc, off_dev), v4=evaluate_vision_v4(enc, dev)); summ, det, _ = evaluate_vision(enc, dev); byd = roi_head_by_distance(enc, dev)
    def brief(v): return {c: dict(recall=round(v[c]['recall'], 3), fa=round(v[c]['false_alarm'], 4), dist_mae=round(v[c]['dist_mae'], 1)) for c in ('traffic_light', 'curve_sign', 'end_marker', 'release_sign')} | dict(light_color={k: round(x, 3) for k, x in v['light_color'].items() if isinstance(x, float)})
    print('微调后 灭灯 dev:', json.dumps(after['off'], ensure_ascii=False)); print('原 dev v4 指标 微调前:', json.dumps(brief(before['v4']), ensure_ascii=False)); print('原 dev v4 指标 微调后:', json.dumps(brief(after['v4']), ensure_ascii=False))
    torch.save(dict(encoder=enc.state_dict(), steps=a.steps, base=a.init, finetune=dict(lr=a.lr, n_off=a.n_off, seed=a.seed, batch=a.batch, off_frac=a.off_frac), coverage='v4+off'), OUT / 'encoder.pt')
    json.dump(dict(before=before, after=after, dev_signal_metrics=summ['all'], dev_roi_head_by_distance=byd, args=vars(a), wall_s=time.monotonic() - t0), open(OUT / 'finetune_report.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('写入', OUT / 'encoder.pt', f'{time.monotonic() - t0:.0f}s')


if __name__ == '__main__':
    main()
