"""把论文八个（按验证集选出的）长程检查点重新放入冻结模拟器评估。

核对：每个 12 工况评估网格与归档训练日志逐工况一致（误差 <= 1e-6），
报告集（偏移 22.5/67.5 s 的 6 个工况）均值与 reference/bench_results.json 一致（对应论文 Table III）。
注意：发布包把权重放在 weights/、曲线放在 data/bench_curves/，与 evsim_v9/evaluate_checkpoint.py
期望的“运行 JSON 与检查点同目录”不符，直接调用该脚本会报 FileNotFoundError；本脚本显式配对。
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as P
P.use_long_route()
import env20 as E, td3_run as T
from refresh_refs import env_fingerprint

REP = {22.5, 67.5}; VAL = {0., 45.}
results = {r['args']['seed']: r for r in json.load(open(P.REFERENCE / 'bench_results.json'))}
out_dir = P.RESULTS / 'replay'; out_dir.mkdir(parents=True, exist_ok=True)
rows = []
for seed in sorted(results):
    step = results[seed]['selected_step']
    ck_path = P.REFERENCE / 'selected_checkpoints' / f'ckpt_bench_s{seed}_{step}steps.pt'
    run = json.load(open(P.REFERENCE / 'bench_curves' / f'bench_s{seed}.json'))
    ck = torch.load(ck_path, map_location='cpu', weights_only=True)
    assert ck['obs_dim'] == E.OBS_DIM and ck['step'] == step
    assert run['refs']['environment_sha256'] == env_fingerprint(), '冻结环境指纹与生成曲线时的环境不一致'
    args = ck['args']
    E.REWARD_MODE = args['reward']; E.CURVE_MODE = args['curve']; E.SHAPE_C = args['shape']
    E.CURVE_ENVELOPE = bool(args.get('curve_env') or args['curve'] == 'envelope')
    E.CLAMP_LIMIT = args['clamp']; E.K_CURVE = args['kcurve']
    actor = T.MLP(E.OBS_DIM, 1, True); actor.load_state_dict(ck['actor']); actor.eval()
    grid = T.evaluate(actor, args['repeat'])
    archived = next(c for c in run['curve'] if c['step'] == step)['grid']
    err = max(abs(a['R'] - b['R']) for a, b in zip(grid, archived))
    rep = float(np.mean([g['R'] for g in grid if float(g['offset']) in REP]))
    val = float(np.mean([g['R'] for g in grid if float(g['offset']) in VAL]))
    arr = sum(bool(g['arr']) for g in grid if float(g['offset']) in REP)
    row = dict(seed=seed, step=step, grid_max_abs_error=err, reporting_R=rep, archived_reporting_R=results[seed]['reporting_R'],
               reporting_arrivals=arr, validation_R=val, twelve_grid_R=float(np.mean([g['R'] for g in grid])))
    rows.append(row)
    json.dump(dict(row, grid=grid), open(out_dir / f'ckpt_bench_s{seed}_{step}steps.eval.json', 'w'), indent=1)
    print(f"seed {seed} step {step:>8}：网格最大|ΔR| = {err:.2e}；报告集 R {rep:.6f}（归档 {results[seed]['reporting_R']:.6f}）；完赛 {arr}/6", flush=True)
ok = all(r['grid_max_abs_error'] <= 1e-6 and abs(r['reporting_R'] - r['archived_reporting_R']) <= 1e-6 for r in rows)
json.dump(dict(passed=ok, rows=rows, torch=torch.__version__, numpy=np.__version__), open(out_dir / 'replay_summary.json', 'w'), indent=1)
print('检查点回放', '通过：八个选定检查点的评估网格与归档记录完全一致' if ok else '失败：存在不一致，请检查环境指纹与依赖版本')
sys.exit(0 if ok else 1)
