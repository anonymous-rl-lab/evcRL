"""用已导出的三臂网络（final_nets.pt）重新做闭环评估、Z 可解码性与同状态换图检查，不重新训练。
v2e 用途：修正评估随机数（每工况独立相机外观种子、噪声流分离；换图检查两色同外观同噪声）后的公平比较。
旧的 evaluation.json / evaluation_traces / image_swap_rows.json 改名为 *_v2d_shared_camera* 保留。
用法：python visual_dev/reevaluate.py --tag pilot_v2 [--arm joint] [--swap-only]"""
import argparse, json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'visual_dev'))
import numpy as np, torch
import pipeline as P
from pipeline import S, STACK, VisualTD3, Config, evaluate, visual_summary, z_decodability, image_swap_sensitivity
torch.set_num_threads(1)
RUNS = ROOT / 'runs'


def learner_from_final_nets(path):
    ck = torch.load(path, map_location='cpu', weights_only=False); cfg = ck['cfg']
    learner = VisualTD3(Config(mode=cfg['mode'], information_mode='camera_map', stack=STACK, actor_lr=cfg['actor_lr'], critic_lr=cfg['critic_lr'],
                               encoder_lr=cfg['encoder_lr'], vision_weight=cfg['vision_weight'], lambda_c=cfg['lambda_c'], tau=cfg['tau'],
                               policy_delay=cfg['policy_delay'], target_noise=cfg['target_noise'], target_clip=cfg['target_clip']))
    for k, v in ck['nets'].items(): learner.named_nets()[k].load_state_dict(v)
    for m in learner.named_nets().values(): m.eval()
    return learner, ck


def archive(path, suffix='_v2d_shared_camera'):
    p = Path(path)
    if not p.exists(): return
    q = p.with_name(p.stem + suffix + p.suffix) if p.is_file() else p.with_name(p.name + suffix)
    if q.exists(): (shutil.rmtree(q) if q.is_dir() else q.unlink())
    p.rename(q)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='pilot_v2'); ap.add_argument('--arm', default=None); ap.add_argument('--swap-only', action='store_true'); ap.add_argument('--swap-update', action='store_true'); ap.add_argument('--runs', default='', help='runs 子目录，如 v3'); ap.add_argument('--pretrain', default='pretrain')
    a = ap.parse_args(); arms = [a.arm] if a.arm else ['frozen', 'supervised', 'joint']
    global RUNS; RUNS = ROOT / 'runs' / a.runs
    es = torch.load(ROOT / 'runs' / a.pretrain / 'eval_sets.pt', map_location='cpu', weights_only=False)
    for arm in arms:
        out = RUNS / a.tag / arm; learner, ck = learner_from_final_nets(out / 'final_nets.pt'); old = json.load(open(out / 'evaluation.json'))
        t0 = time.monotonic(); sw = image_swap_sensitivity(learner)
        print(f"[{arm}] 换图 |Δu| {sw['mean_abs_du']:.4f} Δu(绿−红) {sw['mean_du_green_minus_red']:+.4f} | 投影后 |Δa| {sw['mean_abs_da']:.4f} Δa(绿−红) {sw['mean_da_green_minus_red']:+.4f} Δr(绿−红) {sw['mean_dr_green_minus_red']:+.4f} 真值绿灯占比 {sw['truth_green_frac']:.2f} ({time.monotonic() - t0:.0f}s)", flush=True)
        if a.swap_only: json.dump(sw['rows'], open(out / 'image_swap_rows_smoke.json', 'w'), indent=1); continue
        if a.swap_update:   # 只更新换图检查字段（闭环评估不变）
            old['image_swap'] = {k: v for k, v in sw.items() if k != 'rows'}; json.dump(sw['rows'], open(out / 'image_swap_rows.json', 'w'), indent=1)
            json.dump(old, open(out / 'evaluation.json', 'w'), indent=1, ensure_ascii=False); continue
        te = time.monotonic(); rows, visual, traces = evaluate(learner, S.conditions('development'))
        ev = dict(rows=rows, summary=S.summarize(rows), settled=sum(r['settled'] for r in rows), fallback=sum(r['fallback_substeps'] for r in rows),
                  intervened=sum(r['intervened_substeps'] for r in rows), visual=visual_summary(visual), z=old['z'], eval_wall_s=time.monotonic() - te,
                  arm=arm, substeps_trained=old['substeps_trained'], updates=old['updates'],
                  evaluation_version='v2e: 每工况独立相机外观种子 + 噪声流分离；换图两色同外观同噪声并记录投影后动作',
                  final_nets_identity=ck.get('identity'))
        for name in ('evaluation.json', 'evaluation_traces', 'image_swap_rows.json'): archive(out / name)
        for r, (tr, log) in zip(rows, traces):
            S.save_trace(out / 'evaluation_traces' / f"dev_{r['condition_id']:02d}.npz", tr); json.dump(log, open(out / 'evaluation_traces' / f"dev_{r['condition_id']:02d}_layer.json", 'w'))
        ev['z_decodability_dev'] = z_decodability(learner.encoder, es['dev'])
        ev['image_swap'] = {k: v for k, v in sw.items() if k != 'rows'}; json.dump(sw['rows'], open(out / 'image_swap_rows.json', 'w'), indent=1)
        json.dump(ev, open(out / 'evaluation.json', 'w'), indent=1, ensure_ascii=False)
        cm = ev['summary']['completed_mean']
        print(f"[{arm}] 评估：完赛(静止) {ev['settled']}/9 违规 {ev['summary']['violations']} 平均I_j {cm['Ij']:.2f} 时间 {cm['time_s']:.1f} 能耗 {cm['E_Wh']:.1f} R {cm['R']:.3f} 干预 {ev['intervened']} | Z探针 {ev['z_decodability_dev']['probe_test_acc']:.3f} ({time.monotonic() - te:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
