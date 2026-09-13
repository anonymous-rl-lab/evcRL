"""v4 回归测试：车停在停止线上（或略过线）等待时，灯箱（线远侧 14 m）必须仍可见、可检出、灯色可更新，否则绿灯后永不起步（种子 0 冻结臂工况 0/4 的死锁）。
三层检查：(1) 渲染标签（硬判据）；(2) 编码器近线检测+灯色（信息性：当前编码器在 d_line∈(-1,1) m 对绿灯检测不可靠，见报告）；
(3) 用冻结臂已训练网络在工况 0/4 闭环重评估（硬判据：--must-arrive 指定的工况须到达，默认 4）。
用法：python visual_dev/v4_test_stopline_visibility.py [--arm frozen] [--tag v4_pilot]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P, study as S
from pipeline import Perception, evaluate, FrameStore, STACK
from renderer import SceneCamera, LIGHT_AHEAD
from vision_state import VisionMemory
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='frozen'); ap.add_argument('--tag', default='v4_pilot'); ap.add_argument('--config', default='v4_cpu.json'); ap.add_argument('--conditions', default='0,4'); ap.add_argument('--must-arrive', default='4')
    a = ap.parse_args(); cfg = json.load(open(ROOT / 'configs' / a.config)); ok = True
    xs = S.R.SIGNALS[0]
    # (1) 渲染标签：线前 0.3 m、线后 1.6 m、线后 8 m 时灯箱可见；灯箱后方（线后 14.5 m）不可见；v2 世界不受影响
    cam = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=5, world='v4'); cam.new_episode('t')
    for dx, want in ((-10., 1), (-0.3, 1), (1.6, 1), (14.5, 0)):   # 线后 8 m 时灯箱已在画面上缘之外（物理合理），不作要求
        f = cam.capture(episode_id='t', sim_time=10., pose=dict(x=xs + dx, v=0., offsets=[0.]), force_color='green'); L = f['labels']
        good = int(L['visible']) == want and (want == 0 or (L['color_truth'] == 'green' and abs(L['light_distance_m'] - (LIGHT_AHEAD - dx)) < 1e-6 and f['meta']['map_association'] == 1 and f['meta']['roi_pixels'] > 0))
        print(f"渲染 线{'前' if dx < 0 else '后'} {abs(dx):.1f} m: visible={L['visible']} color={L['color_truth']} light_m={L['light_distance_m']} roi_px={f['meta']['roi_pixels']} {'通过' if good else '失败'}"); ok &= good
    cam2 = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=5, world='v2'); cam2.new_episode('t')
    f2 = cam2.capture(episode_id='t', sim_time=10., pose=dict(x=xs - 0.3, v=0., offsets=[0.]), force_color='green'); print('v2 世界线前 0.3 m visible =', f2['labels']['visible'], '(v2 逻辑未改)')
    # (2) 检测与灯色：静止在线前 0.3 m / 线后 1.6 m，STACK 帧后编码器应检出灯并给出绿色
    thr = cfg['det_thr']['traffic_light'] if isinstance(cfg['det_thr'], dict) else cfg['det_thr']
    per = Perception(ROOT / cfg['pretrained_encoder'], 'roi_head', det_thr=thr)
    for dx in (-0.3, 1.6):
        for color in ('green', 'red'):
            store = FrameStore(); hist = []; cam = SceneCamera(S.R.CURVES, S.R.SIGNALS, S.R.LENGTH, seed=11, noise_seed=3, world='v4'); cam.new_episode('e'); mem = VisionMemory(det_thr=cfg['det_thr'], hold_s=cfg.get('hold_s', 3.))
            for k in range(STACK + 4):
                f = cam.capture(episode_id='e', sim_time=k * S.E.DT, pose=dict(x=xs + dx, v=0., offsets=[0.]), force_color=color); fid = f"e_{k:05d}"; store.put(fid, f); hist.append(fid); del hist[:-STACK]
                dets, probs = per.observe(store, hist, 'e', k * S.E.DT); mem.update(dets, probs, 0., S.E.DT if k else 0.)
            lt = dets['traffic_light']; ph = None if probs is None else P.PHASES[int(np.argmax(probs))]
            good = lt['score'] >= thr and ph == color and mem.sig['phase'] == color
            print(f"检测(信息性) 线{'前' if dx < 0 else '后'} {abs(dx):.1f} m 真值{color}: score={lt['score']:.2f} dist={lt['dist_m']:.1f} 灯色={ph} 记忆相位={mem.sig['phase']} d_line={mem.sig['d_line']} 目标={mem.executor_targets(xs + dx)} {'一致' if good else '不一致'}")
    # (3) 闭环：已训练臂在原先死锁的工况上应到达
    out = ROOT / 'runs' / cfg['runs_subdir'] / a.tag / a.arm
    if (out / 'final_nets.pt').exists():
        from reevaluate import learner_from_final_nets
        learner, ck = learner_from_final_nets(out / 'final_nets.pt'); conds = S.conditions('development'); ids = [int(i) for i in a.conditions.split(',')]
        mk = dict(det_thr=cfg['det_thr'], hold_s=cfg.get('hold_s', 3.)); t0 = time.time()
        rows, _, _ = evaluate(learner, [conds[i] for i in ids], world='v4', perception=per, memory_kw=mk, camera_seed=1000)
        for i, r in zip(ids, rows):
            must = i in [int(j) for j in a.must_arrive.split(',') if j]
            print(f"闭环 {a.arm} 工况 {i}: 到达={r['arrived']} 静止={r['settled']} 违规={r['violations']} 时间={r['time_s']} 停车时长={r['stopped_time_s']} 终点x={r['end_x']:.1f} {'[硬判据]' if must else '[信息性]'} ({time.time() - t0:.0f}s)")
            if must: ok &= bool(r['settled'] and r['violations'] == 0)
            else: ok &= r['violations'] == 0
    print('总结:', '全部通过' if ok else '有失败'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
