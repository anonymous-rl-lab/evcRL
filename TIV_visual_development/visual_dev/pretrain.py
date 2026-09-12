"""离线视觉标注池的生成与编码器监督预训练（三臂共同起点），v2。

池：按 episode（外观种子）划分 train / dev / audit；每个样本是同一 episode 内连续 4 个 0.5 s 子步的帧序列。
60% 样本从信号可见区（x∈[2200,3000]）起步，其余均匀覆盖全程。
v2 指标：灯色按“名义灯箱像素高”和“实际渲染光斑直径”两种分层，ROI 头与 Z 头分别报告；
检测框：类别 0（控制灯）热图最大格解码框与真值框的 IoU、中心误差、命中率（按尺寸分层）；
标签可表示性断言：全部框分量在 [0,1]。
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P
from pipeline import S, SceneCamera, STACK, pool_observation, labels_from_frames
from visual_z.model import VisualEncoder, perception_loss, decode_boxes
OUT = ROOT / 'runs' / 'pretrain'


def compact(seqs):
    """ROI→bool(H,W)，热图/有效位→uint8（取值仅 0/1，无损）；框回归保持 float32。"""
    for seq in seqs:
        for f in seq:
            f['roi'] = (f['roi'][0] if f['roi'].ndim == 3 else f['roi']) > 0
            L = f['labels']; L['heat'] = L['heat'].astype(np.uint8); L['heat_valid'] = L['heat_valid'].astype(np.uint8); L['box_valid'] = L['box_valid'].astype(np.uint8)


def make_sequences(n, seed, tag):
    cam = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=seed); rng = np.random.default_rng(seed + 1)
    seqs = []
    for i in range(n):
        eid = f'{tag}{i:05d}'; cam.new_episode(eid)
        x = float(rng.uniform(2200., 3000.)) if rng.random() < .6 else float(rng.uniform(0., 3990.))
        v = float(rng.uniform(4., 22.)); t = float(rng.uniform(0., 600.)); off = float(rng.uniform(0., 90.))
        seqs.append([cam.capture(episode_id=eid, sim_time=t + k * S.E.DT, pose=dict(x=min(x + v * k * S.E.DT, 3999.), v=v, offsets=[off])) for k in range(STACK)])
    return seqs


def label_representability(seqs):
    n = neg = over = 0; lo, hi = np.inf, -np.inf
    for seq in seqs:
        for f in seq:
            m = f['labels']['box_valid'][0] > 0; v = f['labels']['boxes'][:, m]; n += v.shape[1]
            if v.size: neg += int((v < 0).sum()); over += int((v > 1).sum()); lo = min(lo, float(v.min())); hi = max(hi, float(v.max()))
    return dict(positive_box_cells=n, negative_components=neg, over_one_components=over, min_target=lo, max_target=hi)


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0])); iy = max(0., min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


def evaluate_vision(encoder, seqs, batch=16, score_thr=0.5):
    """灯色分层指标 + 控制灯检测指标（v2c 口径）。
    检测：对每帧取类别 0 热图最大格解码框。可见灯帧报告：格一致率 cell_ok（最大格 == 真值格）、逐帧中心误差、
    中心误差 ≤2 px 命中率、阈值召回（最大格得分 > score_thr 且格正确）；无可见灯的帧报告误检率（最大格得分 > score_thr）。
    IoU 只作辅助（1 px 量级目标下 IoU 分支不可达，不作定位能力度量）。分层按名义灯箱高 0–1/1–2/2–4/4–8/8–16 px。"""
    rows = []; det = []; neg = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]; obs = pool_observation(chunk)
        with torch.no_grad():
            out = encoder(obs); pred = out['signal'].argmax(-1).numpy(); predz = out['signal_z'].argmax(-1).numpy()
            boxes, score = decode_boxes(out['heat'], out['boxes'], cls=0)
            hh, ww = out['heat'].shape[-2:]; idx = out['heat'][:, :, 0].reshape(out['heat'].shape[0], out['heat'].shape[1], -1).argmax(-1)
        for j, seq in enumerate(chunk):
            for k, f in enumerate(seq):
                L, M = f['labels'], f['meta']
                if M['map_association']:
                    rows.append(dict(pred=int(pred[j, k]), pred_z=int(predz[j]) if k == STACK - 1 else None, truth=int(L['signal']), px=float(L['housing_px_h']),
                                     lamp_px_d=float(L['lamp_px_d']), d=L['light_distance_m'], roi_nonempty=int(M['roi_pixels'] > 0), visible=int(L['visible']), occluded=int(L['occluded'])))
                sc = float(score[j, k])
                if L['visible'] and L['light_box_px'] is not None:
                    pb = [float(x) for x in boxes[j, k]]; tb = L['light_box_px']; cell_true = np.argwhere(L['heat'][0] > 0)
                    ti, tj = (int(cell_true[0][0]), int(cell_true[0][1])) if len(cell_true) else (-1, -1)
                    pi, pj = int(idx[j, k]) // ww, int(idx[j, k]) % ww
                    cerr = float(np.hypot((pb[0] + pb[2]) / 2 - (tb[0] + tb[2]) / 2, (pb[1] + pb[3]) / 2 - (tb[1] + tb[3]) / 2))
                    det.append(dict(px=float(L['housing_px_h']), iou=iou(pb, tb), center_err_px=cerr, cell_ok=int((pi, pj) == (ti, tj)), hit_le2px=int(cerr <= 2.),
                                    recall_thr=int(sc > score_thr and (pi, pj) == (ti, tj)), score=sc))
                else:
                    neg.append(dict(score=sc, false_alarm=int(sc > score_thr), map_association=int(M['map_association'])))
    summ = P.visual_summary(rows)
    def agg(d):
        return dict(n=len(d), cell_ok_rate=float(np.mean([r['cell_ok'] for r in d])), mean_center_err_px=float(np.mean([r['center_err_px'] for r in d])),
                    median_center_err_px=float(np.median([r['center_err_px'] for r in d])), hit_le2px=float(np.mean([r['hit_le2px'] for r in d])),
                    recall_at_thr=float(np.mean([r['recall_thr'] for r in d])), mean_score=float(np.mean([r['score'] for r in d])), aux_mean_iou=float(np.mean([r['iou'] for r in d])),
                    sequences=None)
    detection = {}
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]:
        d = [r for r in det if lo <= r['px'] < hi]
        if d: detection[f'{lo:g}-{hi if hi < 1e9 else "inf"}px'] = agg(d)
    detection['all'] = agg(det) if det else dict(n=0)
    detection['all']['hit_rate'] = detection['all'].get('hit_le2px')   # 兼容旧字段名：命中率 = 中心误差 ≤2 px
    detection['no_visible_light_frames'] = dict(n=len(neg), false_alarm_rate=float(np.mean([r['false_alarm'] for r in neg])) if neg else None,
                                                mean_score=float(np.mean([r['score'] for r in neg])) if neg else None,
                                                n_with_map_association=int(sum(r['map_association'] for r in neg)), score_threshold=score_thr)
    return summ, detection, rows


def z_head_on_latest(encoder, seqs, batch=16):
    rows = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]; obs = pool_observation(chunk)
        with torch.no_grad(): predz = encoder(obs)['signal_z'].argmax(-1).numpy()
        for j, seq in enumerate(chunk):
            f = seq[-1]
            if f['meta']['map_association']: rows.append(dict(pred=int(predz[j]), pred_z=int(predz[j]), truth=int(f['labels']['signal']), px=float(f['labels']['housing_px_h']), lamp_px_d=float(f['labels']['lamp_px_d'])))
    return P.visual_summary(rows)['all']['roi_head'] if rows else None


def main(n_train=1600, n_dev=240, n_audit=32, steps=1200, batch=16, seed=4242):
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    train = make_sequences(n_train, seed, 'tr'); dev = make_sequences(n_dev, seed + 100, 'dv'); audit = make_sequences(n_audit, seed + 200, 'au')
    gen_s = time.monotonic() - t0
    rep = {k: label_representability(v) for k, v in (('train', train), ('dev', dev))}
    assert all(r['negative_components'] == 0 and r['over_one_components'] == 0 for r in rep.values()), '框标签超出 [0,1]'
    for seqs in (train, dev, audit): compact(seqs)
    torch.save(dict(sequences=train, seed=seed, hw=(P.H, P.W)), OUT / 'pool.pt')
    torch.save(dict(dev=dev, audit=audit, seed=seed), OUT / 'eval_sets.pt')
    np.savez_compressed(OUT / 'audit_set.npz', rgb=np.stack([[f['rgb'] for f in s] for s in audit]), roi=np.stack([[f['roi'] for f in s] for s in audit]),
                        signal=np.array([[f['labels']['signal'] for f in s] for s in audit]), map_association=np.array([[f['meta']['map_association'] for f in s] for s in audit]))
    torch.manual_seed(seed); enc = VisualEncoder(STACK); opt = torch.optim.Adam(enc.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed); log = []; t1 = time.monotonic()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train), batch); seqs = [train[i] for i in idx]
        obs = pool_observation(seqs); labels = labels_from_frames(seqs)
        loss = perception_loss(enc(obs), labels, obs); opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == 1:
            enc.eval(); summ, det, _ = evaluate_vision(enc, dev[:120]); zh = z_head_on_latest(enc, dev[:120]); enc.train()
            log.append(dict(step=step, loss=float(loss), dev_all=summ['all'], z_head_latest=zh, detection=det['all'], wall_s=time.monotonic() - t1))
            print(f"预训练 step {step} loss {float(loss):.4f} | ROI头 acc {summ['all']['roi_head']['acc']:.3f} unknown召回 {summ['all']['roi_head']['unknown_recall']} | Z头(最新帧) acc {zh['acc'] if zh else None:.3f} | 检测 格一致 {det['all'].get('cell_ok_rate')} ≤2px {det['all'].get('hit_le2px')} 误检 {det['no_visible_light_frames']['false_alarm_rate']}", flush=True)
    enc.eval(); summ, det, rows = evaluate_vision(enc, dev); zh = z_head_on_latest(enc, dev)
    torch.save(dict(encoder=enc.state_dict(), steps=steps, batch=batch, seed=seed, dev_metrics=summ, pool_seed=seed), OUT / 'encoder.pt')
    grads = {}
    obs = pool_observation(dev[:4]); labels = labels_from_frames(dev[:4]); enc.zero_grad(); perception_loss(enc(obs), labels, obs).backward()
    for name, mod in (('temporal_Z', enc.temporal), ('signal_head', enc.signal), ('signal_z_head', enc.signal_z), ('backbone_c2', enc.c2)):
        grads[name] = sum(float(p.grad.square().sum()) for p in mod.parameters() if p.grad is not None) ** .5
    report = dict(pool=dict(train=n_train, dev=n_dev, audit=n_audit, frames_each=STACK, hw=[P.H, P.W], generation_wall_s=gen_s, frame_bytes=int(train[0][0]['rgb'].nbytes)),
                  label_representability=rep, training=dict(steps=steps, batch_sequences=batch, frames_per_step=batch * STACK, wall_s=time.monotonic() - t1, seconds_per_step=(time.monotonic() - t1) / steps),
                  dev_signal_metrics=summ, dev_z_head_latest_frame=zh, dev_detection=det, supervised_gradient_norms=grads, log=log,
                  note='随机初始化编码器的监督预训练，非预训练 MobileNetV3；dev 与 train 使用不同外观 episode 种子；分层同时给出名义灯箱高与实际光斑直径')
    json.dump(report, open(OUT / 'pretrain_report.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(dict(all=summ['all'], z_head=zh, detection=det['all'], grads=grads, wall=report['training']['wall_s']), ensure_ascii=False))


if __name__ == '__main__':
    main()
