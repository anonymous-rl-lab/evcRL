"""离线视觉标注池的生成与编码器监督预训练（三臂共同起点）。

池：按 episode（外观种子）划分 train / dev / audit；每个样本是同一 episode 内连续 4 个 0.5 s 子步的帧序列
（匀速行驶，位姿由环境几何推进），标签来自渲染器真值（灯色、灯箱/标志/终点框、可见性）。
60% 样本从信号可见区（x∈[2200,3000]）起步，其余均匀覆盖全程，避免小目标样本过少。
指标：按灯箱像素高 <4 / 4–8 / 8–16 / ≥16 px 分层的灯色正确率、红→绿误判、绿→非绿、unknown 率。
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P
from pipeline import S, SceneCamera, STACK, pool_observation, labels_from_frames
from visual_z.model import VisualEncoder, perception_loss
OUT = ROOT / 'runs' / 'pretrain'


def make_sequences(n, seed, tag):
    cam = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=seed); rng = np.random.default_rng(seed + 1)
    seqs = []
    for i in range(n):
        eid = f'{tag}{i:05d}'; cam.new_episode(eid)
        x = float(rng.uniform(2200., 3000.)) if rng.random() < .6 else float(rng.uniform(0., 3990.))
        v = float(rng.uniform(4., 22.)); t = float(rng.uniform(0., 600.)); off = float(rng.uniform(0., 90.))
        seq = []
        for k in range(STACK):
            seq.append(cam.capture(episode_id=eid, sim_time=t + k * S.E.DT, pose=dict(x=min(x + v * k * S.E.DT, 3999.), v=v, offsets=[off])))
        seqs.append(seq)
    return seqs


def compact(seqs):
    """ROI→bool(H,W)，热图/有效位→uint8（取值仅 0/1，无损）；框回归保持 float32。"""
    for seq in seqs:
        for f in seq:
            f['roi'] = (f['roi'][0] if f['roi'].ndim == 3 else f['roi']) > 0
            L = f['labels']; L['heat'] = L['heat'].astype(np.uint8); L['heat_valid'] = L['heat_valid'].astype(np.uint8); L['box_valid'] = L['box_valid'].astype(np.uint8)


def evaluate_signal(encoder, seqs, batch=16):
    rows = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]; obs = pool_observation(chunk)
        with torch.no_grad(): pred = encoder(obs)['signal'].argmax(-1).numpy()
        for j, seq in enumerate(chunk):
            for k, f in enumerate(seq):
                if f['meta']['association_valid']:
                    rows.append(dict(pred=int(pred[j, k]), truth=int(f['labels']['signal']), px=float(f['labels']['light_px_h']), d=f['labels']['light_distance_m']))
    return P.visual_summary(rows), rows


def main(n_train=1600, n_dev=240, n_audit=32, steps=1200, batch=16, seed=4242):
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    train = make_sequences(n_train, seed, 'tr'); dev = make_sequences(n_dev, seed + 100, 'dv'); audit = make_sequences(n_audit, seed + 200, 'au')
    gen_s = time.monotonic() - t0
    for seqs in (train, dev, audit): compact(seqs)   # 无损压缩存储 dtype，降低三臂并行时的内存占用
    torch.save(dict(sequences=train, dev=dev, audit=audit, seed=seed, hw=(P.H, P.W)), OUT / 'pool.pt')
    torch.manual_seed(seed); enc = VisualEncoder(STACK); opt = torch.optim.Adam(enc.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed); log = []; t1 = time.monotonic()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train), batch); seqs = [train[i] for i in idx]
        obs = pool_observation(seqs); labels = labels_from_frames(seqs)
        loss = perception_loss(enc(obs), labels, obs); opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            enc.eval(); summ, _ = evaluate_signal(enc, dev[:120]); enc.train()
            log.append(dict(step=step, loss=float(loss), dev=summ, wall_s=time.monotonic() - t1))
            print(f"预训练 step {step} loss {float(loss):.4f} | dev 灯色: " + ' '.join(f"{k}:acc={v['acc']:.2f}(n={v['n']})" for k, v in summ.items()), flush=True)
    enc.eval(); summ, rows = evaluate_signal(enc, dev)
    torch.save(dict(encoder=enc.state_dict(), steps=steps, batch=batch, seed=seed, dev_metrics=summ, pool_seed=seed), OUT / 'encoder.pt')
    report = dict(pool=dict(train=n_train, dev=n_dev, audit=n_audit, frames_each=STACK, hw=[P.H, P.W], generation_wall_s=gen_s,
                            frame_bytes=int(train[0][0]['rgb'].nbytes)), training=dict(steps=steps, batch_sequences=batch, frames_per_step=batch * STACK,
                            wall_s=time.monotonic() - t1, seconds_per_step=(time.monotonic() - t1) / steps), dev_signal_metrics=summ, log=log,
                  note='随机初始化编码器的监督预训练，非预训练 MobileNetV3；dev 与 train 使用不同外观 episode 种子')
    json.dump(report, open(OUT / 'pretrain_report.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(dict(dev=summ, wall=report['training']['wall_s']), ensure_ascii=False))


if __name__ == '__main__':
    main()
