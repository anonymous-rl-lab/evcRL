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
from visual_z.contracts import VisualObservation
from visual_z.model import VisualEncoder, perception_loss, decode_boxes
OUT = ROOT / 'runs' / 'pretrain'


def compact(seqs):
    """ROI→bool(H,W)，热图/有效位→uint8（取值仅 0/1，无损）；框回归保持 float32。"""
    for seq in seqs:
        for f in seq:
            f['roi'] = (f['roi'][0] if f['roi'].ndim == 3 else f['roi']) > 0
            L = f['labels']; L['heat'] = L['heat'].astype(np.uint8); L['heat_valid'] = L['heat_valid'].astype(np.uint8); L['box_valid'] = L['box_valid'].astype(np.uint8)


def make_sequences(n, seed, tag, coverage='v2', world=None):
    """生成 n 个 4 帧序列。coverage='v2'：原采样（60% x∈[2200,3000]，v∈[4,22]，相位均匀）。
    coverage='v3'：分层覆盖——30% 原区间；25% 近线 x∈[2900,3000]、v∈[0,12]；10% 停在线上 x∈[2994,3000]、v∈[0,2]；
    20% 全路线；15% 相位定向（x∈[2500,3000]，相位落在绿末/黄/红初 [26,38) s），使近距离与黄灯不再是覆盖缺口。"""
    world = world or ('v4' if coverage in ('v4', 'v4b') else 'v2')
    cam = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=seed, world=world); rng = np.random.default_rng(seed + 1)
    seqs = []; xs = S.R.SIGNALS[0]; ca, cb, _ = S.R.CURVES[0]
    for i in range(n):
        eid = f'{tag}{i:05d}'; cam.new_episode(eid)
        t = float(rng.uniform(0., 600.)); off = float(rng.uniform(0., 90.))
        if coverage in ('v4', 'v4b'):   # 无地图世界：按道路事件分区覆盖；v4b：停在线上分区扩到线后 3 m（灯箱在线远侧 14 m，停车等待帧 d∈[11,20] m），并加近线低速接近分区
            u = rng.random()
            if coverage == 'v4b' and u < .08: x = float(rng.uniform(xs - 6., xs + 3.)); v = float(rng.uniform(0., 2.))       # v4b：停在线上/略过线等待
            elif coverage == 'v4b' and u < .13: x = float(rng.uniform(xs - 30., xs - 2.)); v = float(rng.uniform(0., 6.))   # v4b：近线低速接近
            elif u < .20: x = float(rng.uniform(ca - 400. - 330., ca - 400.)); v = float(rng.uniform(4., 22.))      # 警示牌可见区
            elif u < .30: x = float(rng.uniform(ca - 400., cb + 30.)); v = float(rng.uniform(4., 14.))            # 牌后到弯道/解除牌
            elif u < .55: x = float(rng.uniform(xs - 210., xs)); v = float(rng.uniform(0., 20.))                    # 灯 200 m 可见区
            elif u < .63: x = float(rng.uniform(xs - 6., xs)); v = float(rng.uniform(0., 2.))                       # 停在线上
            elif u < .73: x = float(rng.uniform(xs - 210., xs)); v = float(rng.uniform(0., 18.)); off = float((rng.uniform(26., 38.) - t) % 90.)   # 相位定向（绿末/黄/红初）
            elif u < .85: x = float(rng.uniform(S.R.LENGTH - 410., S.R.LENGTH - 10.)); v = float(rng.uniform(0., 20.))   # 终点标志区
            else: x = float(rng.uniform(0., 3990.)); v = float(rng.uniform(4., 22.))                                # 全路线
        elif coverage == 'v2':
            x = float(rng.uniform(2200., 3000.)) if rng.random() < .6 else float(rng.uniform(0., 3990.)); v = float(rng.uniform(4., 22.))
        else:
            u = rng.random()
            if u < .30: x = float(rng.uniform(2200., 3000.)); v = float(rng.uniform(4., 22.))
            elif u < .55: x = float(rng.uniform(xs - 100., xs)); v = float(rng.uniform(0., 12.))
            elif u < .65: x = float(rng.uniform(xs - 6., xs)); v = float(rng.uniform(0., 2.))
            elif u < .85: x = float(rng.uniform(0., 3990.)); v = float(rng.uniform(4., 22.))
            else:
                x = float(rng.uniform(2500., 3000.)); v = float(rng.uniform(0., 18.))
                off = float((rng.uniform(26., 38.) - t) % 90.)   # 使 (t+off)%90 落在 [26,38)
        seqs.append([cam.capture(episode_id=eid, sim_time=t + k * S.E.DT, pose=dict(x=min(x + v * k * S.E.DT, 3999.), v=v, offsets=[off])) for k in range(STACK)])
    return seqs


def roi_head_by_distance(encoder, seqs, batch=16, bins=((0, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 450))):
    """v3：ROI 头按“车到停止线距离”分层的已知类正确率与黄→绿误判率（执行层感知可靠性的直接口径）。"""
    recs = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]; obs = pool_observation(chunk)
        with torch.no_grad(): pr = encoder(obs)['signal'].softmax(-1).numpy()
        for j, seq in enumerate(chunk):
            for k, f in enumerate(seq):
                L = f['labels']
                if f['meta']['map_association'] and L.get('visible'): recs.append(dict(d=float(L['light_distance_m']) - 14., truth=int(L['signal']), pred=int(pr[j, k].argmax()), p_green=float(pr[j, k, 2])))
    out = {}
    for lo, hi in bins:
        b = [r for r in recs if lo <= r['d'] < hi]; known = [r for r in b if r['truth'] < 4]; y = [r for r in b if r['truth'] == 1]; g = [r for r in b if r['truth'] == 2]; rd = [r for r in b if r['truth'] == 0]
        out[f'{lo}-{hi}m'] = dict(n=len(b), known_acc=float(np.mean([r['pred'] == r['truth'] for r in known])) if known else None,
                                  green_recall=float(np.mean([r['pred'] == 2 for r in g])) if g else None, green_p_median=float(np.median([r['p_green'] for r in g])) if g else None,
                                  yellow_to_green=float(np.mean([r['pred'] == 2 for r in y])) if y else None, red_to_green=float(np.mean([r['pred'] == 2 for r in rd])) if rd else None,
                                  n_yellow=len(y), n_green=len(g), n_red=len(rd))
    return out


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


def evaluate_vision_v4(encoder, seqs, batch=16, score_thr=0.5, bins=((0, 50), (50, 100), (100, 200), (200, 330), (330, 400))):
    """v4 口径：每类目标（灯/警示牌/终点/解除牌）按真值距离分层的检测召回（得分>thr 且格在真值格 ±1 内）与距离估计误差；
    无该类目标的帧上的误检率；灯色的运行时路径：由预测框构造 ROI 再过灯色头的正确率（与教师 ROI 对比）。"""
    from visual_z.model import decode_detections, predicted_roi
    from renderer import CLASSES
    per = {c: [] for c in range(4)}; neg = {c: [] for c in range(4)}; color = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]; obs = pool_observation(chunk)
        with torch.no_grad(): out = encoder(obs)
        dets = {c: decode_detections(out, c) for c in range(4)}
        # 预测 ROI 灯色：用最新帧的灯检测框构造 ROI，再前向一次
        roi2 = obs.signal_roi.clone()
        for j in range(len(chunk)):
            for k in range(STACK):
                roi2[j, k, 0] = predicted_roi(dets[0]['box'][j, k], P.H, P.W) if float(dets[0]['score'][j, k]) >= score_thr else torch.zeros(P.H, P.W)
        obs2 = VisualObservation(obs.legacy, obs.frames, obs.valid, obs.age_s, roi2, obs.association_valid, obs.v2x_valid)
        with torch.no_grad(): out2 = encoder(obs2)
        for j, seq in enumerate(chunk):
            for k, f in enumerate(seq):
                L = f['labels']; hm = L['heat']
                for c in range(4):
                    cells = np.argwhere(hm[c] > 0)
                    if len(cells):
                        ti, tj = cells[0]; pi, pj = dets[c]['cell'][j, k].tolist(); ok = float(dets[c]['score'][j, k]) >= score_thr and abs(pi - ti) <= 1 and abs(pj - tj) <= 1
                        d_true = float(L['dist'][0, ti, tj]) * 400.; d_pred = float(dets[c]['dist_m'][j, k])
                        per[c].append(dict(d=d_true, hit=int(ok), derr=abs(d_pred - d_true) if ok else None, score=float(dets[c]['score'][j, k])))
                    else: neg[c].append(int(float(dets[c]['score'][j, k]) >= score_thr))
                if L.get('visible') and f['meta']['map_association']:
                    color.append(dict(d=float(L['light_distance_m']), truth=int(L['signal']), teacher=int(out['signal'][j, k].argmax()), runtime=int(out2['signal'][j, k].argmax()),
                                      detected=int(float(dets[0]['score'][j, k]) >= score_thr), roi_iou=float((roi2[j, k, 0] * obs.signal_roi[j, k, 0]).sum() / ((roi2[j, k, 0] + obs.signal_roi[j, k, 0]).clamp(max=1).sum().clamp_min(1)))))
    rep = {}
    for c in range(4):
        rows = per[c]; rep[CLASSES[c]] = dict(n=len(rows), recall=float(np.mean([r['hit'] for r in rows])) if rows else None, false_alarm=float(np.mean(neg[c])) if neg[c] else None,
            dist_mae=float(np.mean([r['derr'] for r in rows if r['derr'] is not None])) if any(r['derr'] is not None for r in rows) else None,
            by_distance={f'{lo}-{hi}': dict(n=len(b), recall=float(np.mean([r['hit'] for r in b])) if b else None, dist_mae=float(np.mean([r['derr'] for r in b if r['derr'] is not None])) if any(r['derr'] is not None for r in b) else None,
                                            dist_rel=float(np.mean([r['derr'] / max(r['d'], 1.) for r in b if r['derr'] is not None])) if any(r['derr'] is not None for r in b) else None)
                         for lo, hi in bins for b in [[r for r in rows if lo <= r['d'] < hi]]})
    known = [r for r in color if r['truth'] < 4]
    rep['light_color'] = dict(n=len(color), teacher_roi_known_acc=float(np.mean([r['teacher'] == r['truth'] for r in known])) if known else None,
                              runtime_roi_known_acc=float(np.mean([r['runtime'] == r['truth'] for r in known])) if known else None,
                              runtime_detected_rate=float(np.mean([r['detected'] for r in color])) if color else None, mean_roi_iou=float(np.mean([r['roi_iou'] for r in color])) if color else None,
                              by_distance={f'{lo}-{hi}': dict(n=len(b), runtime_known_acc=float(np.mean([r['runtime'] == r['truth'] for r in b if r['truth'] < 4])) if any(r['truth'] < 4 for r in b) else None,
                                                             red_to_green=int(sum(r['truth'] == 0 and r['runtime'] == 2 for r in b)), yellow_to_green=int(sum(r['truth'] == 1 and r['runtime'] == 2 for r in b)))
                                           for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 150), (150, 215)) for b in [[r for r in color if lo <= r['d'] - 14. < hi]]})
    return rep


def z_head_on_latest(encoder, seqs, batch=16):
    rows = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]; obs = pool_observation(chunk)
        with torch.no_grad(): predz = encoder(obs)['signal_z'].argmax(-1).numpy()
        for j, seq in enumerate(chunk):
            f = seq[-1]
            if f['meta']['map_association']: rows.append(dict(pred=int(predz[j]), pred_z=int(predz[j]), truth=int(f['labels']['signal']), px=float(f['labels']['housing_px_h']), lamp_px_d=float(f['labels']['lamp_px_d'])))
    return P.visual_summary(rows)['all']['roi_head'] if rows else None


def main(n_train=1600, n_dev=240, n_audit=32, steps=1200, batch=16, seed=4242, coverage='v2'):
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    train = make_sequences(n_train, seed, 'tr', coverage); dev = make_sequences(n_dev, seed + 100, 'dv', coverage); audit = make_sequences(n_audit, seed + 200, 'au', coverage)
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
    enc.eval(); summ, det, rows = evaluate_vision(enc, dev); zh = z_head_on_latest(enc, dev); byd = roi_head_by_distance(enc, dev)
    print('ROI 头按距离分层（dev）:', json.dumps(byd, ensure_ascii=False), flush=True)
    v4rep = evaluate_vision_v4(enc, dev) if coverage in ('v4', 'v4b') else None
    if v4rep: print('v4 检测/距离/运行时灯色（dev）:', json.dumps(v4rep, ensure_ascii=False), flush=True)
    torch.save(dict(encoder=enc.state_dict(), steps=steps, batch=batch, seed=seed, dev_metrics=summ, pool_seed=seed, coverage=coverage), OUT / 'encoder.pt')
    grads = {}
    obs = pool_observation(dev[:4]); labels = labels_from_frames(dev[:4]); enc.zero_grad(); perception_loss(enc(obs), labels, obs).backward()
    for name, mod in (('temporal_Z', enc.temporal), ('signal_head', enc.signal), ('signal_z_head', enc.signal_z), ('backbone_c2', enc.c2)):
        grads[name] = sum(float(p.grad.square().sum()) for p in mod.parameters() if p.grad is not None) ** .5
    report = dict(pool=dict(train=n_train, dev=n_dev, audit=n_audit, frames_each=STACK, hw=[P.H, P.W], generation_wall_s=gen_s, frame_bytes=int(train[0][0]['rgb'].nbytes)),
                  label_representability=rep, training=dict(steps=steps, batch_sequences=batch, frames_per_step=batch * STACK, wall_s=time.monotonic() - t1, seconds_per_step=(time.monotonic() - t1) / steps),
                  dev_signal_metrics=summ, dev_z_head_latest_frame=zh, dev_detection=det, dev_roi_head_by_distance=byd, dev_v4=v4rep, coverage=coverage, supervised_gradient_norms=grads, log=log,
                  note='随机初始化编码器的监督预训练，非预训练 MobileNetV3；dev 与 train 使用不同外观 episode 种子；分层同时给出名义灯箱高与实际光斑直径')
    json.dump(report, open(OUT / 'pretrain_report.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(dict(all=summ['all'], z_head=zh, detection=det['all'], grads=grads, wall=report['training']['wall_s']), ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--coverage', default='v2', choices=('v2', 'v3', 'v4', 'v4b')); ap.add_argument('--out', default=None); ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--n-train', type=int, default=1600); ap.add_argument('--n-dev', type=int, default=240)
    a = ap.parse_args()
    if a.out: OUT = ROOT / 'runs' / a.out
    main(n_train=a.n_train, n_dev=a.n_dev, steps=a.steps, coverage=a.coverage)
