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
    learner = VisualTD3(Config(mode=cfg['mode'], information_mode=('vision_memory' if cfg.get('world') == 'v4' else 'camera_map'), stack=STACK, actor_lr=cfg['actor_lr'], critic_lr=cfg['critic_lr'],
                               encoder_lr=cfg['encoder_lr'], vision_weight=cfg['vision_weight'], lambda_c=cfg['lambda_c'], tau=cfg['tau'],
                               policy_delay=cfg['policy_delay'], target_noise=cfg['target_noise'], target_clip=cfg['target_clip'],
                               critic_action=cfg.get('critic_action', 'command'), actor_clip_ste=bool(cfg.get('actor_clip_ste', False)), extra_dim=int(cfg.get('extra_dim', 6))))
    for k, v in ck['nets'].items(): learner.named_nets()[k].load_state_dict(v)
    for m in learner.named_nets().values(): m.eval()
    return learner, ck


def archive(path, suffix='_prev'):
    p = Path(path)
    if not p.exists(): return
    q = p.with_name(p.stem + suffix + p.suffix) if p.is_file() else p.with_name(p.name + suffix)
    if q.exists(): (shutil.rmtree(q) if q.is_dir() else q.unlink())
    p.rename(q)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='pilot_v2'); ap.add_argument('--arm', default=None); ap.add_argument('--swap-only', action='store_true'); ap.add_argument('--swap-update', action='store_true'); ap.add_argument('--runs', default='', help='runs 子目录，如 v3'); ap.add_argument('--pretrain', default='pretrain')
    a = ap.parse_args(); arms = [a.arm] if a.arm else (['frozen', 'supervised', 'joint', 'joint_head'] if a.runs == 'v4' else ['frozen', 'supervised', 'joint'])
    global RUNS; RUNS = ROOT / 'runs' / a.runs
    es = torch.load(ROOT / 'runs' / a.pretrain / 'eval_sets.pt', map_location='cpu', weights_only=False)
    for arm in arms:
        out = RUNS / a.tag / arm; learner, ck = learner_from_final_nets(out / 'final_nets.pt'); old = json.load(open(out / 'evaluation.json'))
        if ck['cfg'].get('world') == 'v4': sw = dict(rows=[], note='v4 无换图探针，用遮蔽消融')
        else: sw = image_swap_sensitivity(learner)
        if sw['rows']: print(f"[{arm}] 换图 |Δu| {sw['mean_abs_du']:.4f} Δu(绿−红) {sw['mean_du_green_minus_red']:+.4f} | 投影后 |Δa| {sw['mean_abs_da']:.4f} Δa(绿−红) {sw['mean_da_green_minus_red']:+.4f} Δr(绿−红) {sw['mean_dr_green_minus_red']:+.4f} 真值绿灯占比 {sw['truth_green_frac']:.2f} ({time.monotonic() - t0:.0f}s)", flush=True)
        if a.swap_only: json.dump(sw['rows'], open(out / 'image_swap_rows_smoke.json', 'w'), indent=1); continue
        if a.swap_update:   # 只更新换图检查字段（闭环评估不变）
            old['image_swap'] = {k: v for k, v in sw.items() if k != 'rows'}; json.dump(sw['rows'], open(out / 'image_swap_rows.json', 'w'), indent=1)
            json.dump(old, open(out / 'evaluation.json', 'w'), indent=1, ensure_ascii=False); continue
        cfg = ck['cfg']; v4 = cfg.get('world') == 'v4'
        if v4:
            per = P.Perception(ROOT / cfg['pretrained_encoder'], 'roi_head', det_thr=(cfg['det_thr'].get('traffic_light', .5) if isinstance(cfg.get('det_thr'), dict) else cfg.get('det_thr', .5)))
            mk = dict(det_thr=cfg.get('det_thr', .5), hold_s=cfg.get('hold_s', 3.))
        te = time.monotonic(); rows, visual, traces = evaluate(learner, S.conditions('development'), world=('v4' if v4 else 'v2'), perception=(per if v4 else None), memory_kw=(mk if v4 else None))
        ev = dict(rows=rows, summary=S.summarize(rows), settled=sum(r['settled'] for r in rows), fallback=sum(r['fallback_substeps'] for r in rows),
                  intervened=sum(r['intervened_substeps'] for r in rows), visual=visual_summary(visual), z=old['z'], eval_wall_s=time.monotonic() - te,
                  arm=arm, substeps_trained=old['substeps_trained'], updates=old['updates'],
                  evaluation_version='v2e: 每工况独立相机外观种子 + 噪声流分离；换图两色同外观同噪声并记录投影后动作',
                  final_nets_identity=ck.get('identity'))
        for name in ('evaluation.json', 'evaluation_traces', 'image_swap_rows.json'): archive(out / name)
        for r, (tr, log) in zip(rows, traces):
            S.save_trace(out / 'evaluation_traces' / f"dev_{r['condition_id']:02d}.npz", tr); json.dump(log, open(out / 'evaluation_traces' / f"dev_{r['condition_id']:02d}_layer.json", 'w'))
        ev['z_decodability_dev'] = z_decodability(learner.encoder, es['dev'])
        ev['image_swap'] = {k: v for k, v in sw.items() if k != 'rows'} if sw['rows'] else None
        if sw['rows']: json.dump(sw['rows'], open(out / 'image_swap_rows.json', 'w'), indent=1)
        if v4:
            ev.update(arrived=int(sum(r['arrived'] for r in rows)), offroad=int(sum(r.get('offroad_substeps', 0) for r in rows)), curve_pen=float(sum(r.get('curve_excess_penalty', 0.) for r in rows)), overshoot=int(sum(r.get('overshoot', False) for r in rows)), ablation={})
            for name, hide in (('hide_curve_sign', ('curve_sign',)), ('hide_light_color', ('light_color',)), ('hide_end_marker', ('end_marker',))):
                rr, _, _ = evaluate(learner, S.conditions('development'), world='v4', perception=per, memory_kw=mk, hide=hide); sm = S.summarize(rr)
                ev['ablation'][name] = dict(arrived=int(sum(r['arrived'] for r in rr)), violations=int(sm['violations']), offroad=int(sum(r['offroad_substeps'] for r in rr)), curve_pen=float(sum(r['curve_excess_penalty'] for r in rr)), fallback=int(sum(r['fallback_substeps'] for r in rr)), completed_mean=sm['completed_mean'])
        json.dump(ev, open(out / 'evaluation.json', 'w'), indent=1, ensure_ascii=False)
        cm = ev['summary']['completed_mean'] or {}
        print(f"[{arm}] 评估：完赛(静止) {ev['settled']}/9 到达 {ev.get('arrived','-')} 违规 {ev['summary']['violations']} 弯道超速 {ev.get('offroad','-')} 平均I_j {cm.get('Ij',float('nan')):.2f} 时间 {cm.get('time_s',float('nan')):.1f} 能耗 {cm.get('E_Wh',float('nan')):.1f} R {cm.get('R',float('nan')):.3f} 干预 {ev['intervened']} | Z探针 {ev['z_decodability_dev']['probe_test_acc']:.3f} | 消融 {ev.get('ablation') and {k:(v['arrived'],v['violations'],v['offroad']) for k,v in ev['ablation'].items()}} ({time.monotonic() - te:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
