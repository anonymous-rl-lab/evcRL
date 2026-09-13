"""v4 闭环诊断：常量策略（执行层驱动）在指定工况上跑一遍，把每子步的记忆快照、四类检测与执行层目标压缩成文本日志，便于定位卡死/误检位置。
用法：python visual_dev/v4_diag_closed_loop.py --config v4b_cpu.json --encoder runs/pretrain_v4b/encoder.pt --conditions 1,3 --out runs/probes/diag.txt"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import torch; torch.set_num_threads(1)
from pipeline import Perception, evaluate, S   # 先导入 pipeline（它负责把 v19_deps 加入路径），再取 study
from v4_probe_closed_loop import ConstantLearner


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='v4b_cpu.json'); ap.add_argument('--encoder', default=None); ap.add_argument('--conditions', default='0'); ap.add_argument('--out', default='runs/probes/diag.txt'); ap.add_argument('--det-thr', default=None); ap.add_argument('--u', type=float, default=1.0)
    a = ap.parse_args(); cfg = json.load(open(ROOT / 'configs' / a.config)); mk = {k: cfg[k] for k in ('det_thr', 'hold_s') if k in cfg}
    if a.det_thr: mk['det_thr'] = json.loads(a.det_thr)
    per = Perception(ROOT / (a.encoder or cfg['pretrained_encoder']), 'roi_head', det_thr=mk['det_thr']['traffic_light']); conds = S.conditions('development')
    f = open(ROOT / a.out, 'w')
    sel = [int(c) for c in a.conditions.split(',')]
    rows_all, _, traces_all = evaluate(ConstantLearner(a.u), conds, world='v4', perception=per, memory_kw=mk, camera_seed=1000)   # 跑全部 9 工况以保持与探针相同的逐工况相机种子
    for ci in sel:
        r = rows_all[ci]; traces = [traces_all[ci]]; print(f"工况 {ci}: 到达 {r['arrived']} 违规 {r['violations']} 弯道超速 {r['offroad_substeps']} end_x {r['end_x']:.1f} 时间 {r['time_s']} 停车时长 {r['stopped_time_s']}", file=f, flush=True)
        for k, L in enumerate(traces[0][1]):
            m = L.get('memory', {}); det = L.get('perceived_detail', {}).get('dets', {}); g = lambda d, kk: (round(d.get(kk, 0), 2) if isinstance(d.get(kk), float) else d.get(kk))
            c = m.get('curve', {}); s = m.get('sig', {}); e = m.get('end', {})
            print(k, 'x', round(L.get('x', 0)), 'v', round(L.get('v', 0), 1), 'vlim', round(L.get('v_limit', 0), 1), '| 弯 d', g(c, 'd_est'), 'ann', c.get('announced'), 'act', c.get('active'), 'rel', g(c, 'release_d'), '| 灯 d', g(s, 'd_line'), s.get('phase'), 'age', g(s, 'age'), '| 终 d', g(e, 'd_est'),
                  '| 检 牌', (g(det.get('curve_sign', {}), 'score'), round(det.get('curve_sign', {}).get('dist_m', 0))), '灯', (g(det.get('traffic_light', {}), 'score'), round(det.get('traffic_light', {}).get('dist_m', 0))), '终', (g(det.get('end_marker', {}), 'score'), round(det.get('end_marker', {}).get('dist_m', 0))), '解', (g(det.get('release_sign', {}), 'score'), round(det.get('release_sign', {}).get('dist_m', 0))),
                  '| 感知', L.get('perceived'), '真绿', L.get('truth_green'), '目标', [(round(t[0]), t[1]) for t in L.get('targets', [])], 'fb', L.get('fallback'), file=f)
    f.close()


if __name__ == '__main__':
    main()
