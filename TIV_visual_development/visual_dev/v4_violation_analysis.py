"""v4r 规模运行的失败工况归因：对每个 runs/<sub>/v4_pilot/<arm> 的评估轨迹，
(1) 闯红灯：找真值相位由绿转非绿（黄灯起始）的子步，给出此刻车到停止线距离 d、车速 v、以 3.5 m/s² 的最短停车距离 v²/7，判断“黄灯起始时是否物理可停”；
(2) 未到达：给出停车位置、停车时记忆相位/真值相位、感知与真值不一致的子步数。
用法：python visual_dev/v4_violation_analysis.py --runs v4r v4r_s1 v4r_s2 [--out reports/_v4_failures.json]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import pipeline  # noqa: 路径
from study import TRACE_COLS as C
XS = 3000.


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', default=['v4r', 'v4r_s1', 'v4r_s2']); ap.add_argument('--tag', default='v4_pilot'); ap.add_argument('--out', default='reports/_v4_failures.json')
    a = ap.parse_args(); out = []
    for runs in a.runs:
        for arm in ('frozen', 'supervised', 'joint', 'joint_head'):
            p = ROOT / 'runs' / runs / a.tag / arm / 'evaluation.json'
            if not p.exists(): continue
            ev = json.load(open(p))
            for r in ev['rows']:
                if r['violations'] == 0 and r['arrived']: continue
                i = r['condition_id']; z = np.load(p.parent / 'evaluation_traces' / f'dev_{i:02d}.npz'); V = z['values']; log = json.load(open(p.parent / 'evaluation_traces' / f'dev_{i:02d}_layer.json'))
                rec = dict(runs=runs, arm=arm, condition=i, arrived=r['arrived'], violations=r['violations'], time_s=r['time_s'], end_x=round(r['end_x'], 1))
                rc = np.where(V[:, C.index('red_crossing')] > 0)[0]
                if len(rc):
                    k = int(rc[0]); g = V[:, C.index('signal_green')]; v_cross = float(V[k, C.index('v0')])
                    # 时序约定（审计整改）：signal_green[j] 是第 j 子步结束时刻 t1 的真值相位；相位由绿转非绿发生在子步 j 内。
                    # 事件时刻的车辆状态取子步 j 的起点 (x0, v0)（最晚的“仍是绿灯”的已知状态，保守：真实起始略晚于此）。只在过线前 40 子步（20 s）内找切换，否则不属于同一相位事件。
                    onset = [j for j in range(max(1, k - 40), k + 1) if g[j - 1] > 0.5 and g[j] < 0.5]; j = onset[-1] if onset else None
                    if v_cross < 1.0: rec.update(kind='红灯蠕行过线', crossing_speed=round(v_cross, 2), mem_phase_at_cross=log[k].get('memory', {}).get('sig', {}).get('phase'), mem_hold_at_cross=log[k].get('memory', {}).get('sig', {}).get('hold'))
                    elif j is not None:
                        d = XS - V[j, C.index('x0')]; v = V[j, C.index('v0')]
                        rec.update(kind='闯红灯（黄灯起始后）', yellow_onset_substep=j, d_line_at_onset=round(float(d), 1), v_at_onset=round(float(v), 1), min_stop_dist=round(float(v * v / 7.), 1), stoppable_ideal=bool(v * v / 7. <= d),
                                   note='v²/7 为忽略反应延迟与 jerk 的理想最短距离：d < v²/7 说明即使立即最大制动也不够；反之不保证 jerk 约束下可停',
                                   mem_phase_at_onset=log[j].get('memory', {}).get('sig', {}).get('phase'), mem_d_line_at_onset=None if log[j].get('memory', {}).get('sig', {}).get('d_line') is None else round(log[j]['memory']['sig']['d_line'], 1),
                                   substeps_onset_to_cross=k - j, crossing_speed=round(v_cross, 1))
                    else: rec.update(kind='闯红灯（20 s 内无绿→非绿切换）', crossing_speed=round(v_cross, 1), mem_phase_at_cross=log[k].get('memory', {}).get('sig', {}).get('phase'))
                else:
                    st = [k for k, l in enumerate(log) if l.get('v', 1) < 0.05]
                    if st:
                        k = st[0]; L = log[k]; sg = L.get('memory', {}).get('sig', {}); tail = log[k:]
                        mism = sum(1 for l in tail if l.get('truth_green') is True and l.get('perceived') != 'green')
                        rec.update(kind='未到达', stop_x=round(L['x'], 1), stop_rel_line=round(L['x'] - XS, 1), stopped_substeps=len(tail), mem_phase_at_stop=sg.get('phase'), mem_d_line_at_stop=None if sg.get('d_line') is None else round(sg['d_line'], 1),
                                   perceived_nongreen_while_truth_green=int(mism), end_target=[round(t[0], 1) for t in log[-1].get('targets', []) if t[1] == 0.0])
                    else: rec.update(kind='未到达', note='无停车')
                out.append(rec); print(rec)
    json.dump(out, open(ROOT / a.out, 'w'), ensure_ascii=False, indent=1); print('失败工况数', len(out))


if __name__ == '__main__':
    main()
