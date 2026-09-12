"""v4 闭环探针：无地图世界里，用“执行层驱动”的常量策略（u=+1，名义加速由执行层裁剪）在 9 个开发工况上比较
oracle 记忆（真值检测喂记忆，检验记忆规则与执行层）与视觉记忆（v4 编码器检测喂记忆），并做“驾驶是否依赖视觉”消融：
隐藏警示牌 / 隐藏灯色 / 隐藏解除牌。指标：到达、闯红灯、弯道超速子步与罚分、回退、I_j、时间、能耗。
用法：python visual_dev/v4_probe_closed_loop.py [--encoder runs/pretrain_v4/encoder.pt] [--u 1.0] [--oracle-only]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P, study as S
from pipeline import Perception, evaluate, STACK
from visual_z.model import VisualEncoder, InputAdapter
torch.set_num_threads(1)


class ConstantLearner:
    def __init__(self, u): self.encoder = VisualEncoder(STACK).eval(); self.adapter = InputAdapter('vision_memory'); self.u = float(u)
    def actor(self, st): return torch.full((st.shape[0], 1), self.u)


def run(name, learner, perception, hide=(), memory_kw=None):
    t0 = time.time(); rows, vis, traces = evaluate(learner, S.conditions('development'), world='v4', perception=perception, hide=hide, memory_kw=memory_kw)
    sm = S.summarize(rows); cm = sm['completed_mean'] or {}
    r = dict(arrived=int(sum(r['arrived'] for r in rows)), settled=int(sum(r['settled'] for r in rows)), violations=int(sm['violations']), offroad=int(sum(r['offroad_substeps'] for r in rows)),
             curve_pen=float(sum(r['curve_excess_penalty'] for r in rows)), fallback=int(sum(r['fallback_substeps'] for r in rows)), max_jerk=float(sm['max_jerk']),
             Ij=cm.get('Ij'), time_s=cm.get('time_s'), E_Wh=cm.get('E_Wh'), R=cm.get('R'),
             per_condition=[dict(id=r['condition_id'], arrived=r['arrived'], violations=r['violations'], offroad=r['offroad_substeps'], fallback=r['fallback_substeps'], Ij=round(r['Ij'], 1), time_s=r['time_s'], E_Wh=round(r['E_Wh']), R=round(r['R'], 2)) for r in rows])
    print(f"[{name}] 到达 {r['arrived']}/9 静止 {r['settled']} 闯红灯 {r['violations']} 弯道超速子步 {r['offroad']} 弯道罚分 {r['curve_pen']:.1f} 回退 {r['fallback']} 最坏jerk {r['max_jerk']:.2f} | I_j {r['Ij'] if r['Ij'] is None else round(r['Ij'], 2)} 时间 {r['time_s'] if r['time_s'] is None else round(r['time_s'], 1)} 能耗 {r['E_Wh'] if r['E_Wh'] is None else round(r['E_Wh'], 1)} R {r['R'] if r['R'] is None else round(r['R'], 3)} ({time.time() - t0:.0f}s)", flush=True)
    print('   逐工况 (到达,闯红灯,弯道超速,回退,I_j,时间):', [(c['arrived'], c['violations'], c['offroad'], c['fallback'], c['Ij'], c['time_s']) for c in r['per_condition']], flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--encoder', default='runs/pretrain_v4/encoder.pt'); ap.add_argument('--u', type=float, default=1.0); ap.add_argument('--oracle-only', action='store_true'); ap.add_argument('--out', default='runs/probes/v4_probe_closed_loop.json')
    a = ap.parse_args(); L = ConstantLearner(a.u); res = {}
    res['oracle'] = run('oracle 记忆', L, Perception(source='oracle'))
    res['oracle_hide_curve_sign'] = run('oracle 记忆 / 隐藏警示牌', L, Perception(source='oracle'), hide=('curve_sign',))
    res['oracle_hide_light_color'] = run('oracle 记忆 / 灯色灭', L, Perception(source='oracle'), hide=('light_color',))
    if not a.oracle_only:
        per = Perception(ROOT / a.encoder, 'roi_head')
        res['vision'] = run('视觉记忆', L, per)
        res['vision_hide_curve_sign'] = run('视觉记忆 / 隐藏警示牌', L, Perception(ROOT / a.encoder, 'roi_head'), hide=('curve_sign',))
        res['vision_hide_light_color'] = run('视觉记忆 / 灯色灭', L, Perception(ROOT / a.encoder, 'roi_head'), hide=('light_color',))
        res['vision_hide_release_sign'] = run('视觉记忆 / 隐藏解除牌', L, Perception(ROOT / a.encoder, 'roi_head'), hide=('release_sign',))
    json.dump(res, open(ROOT / a.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
