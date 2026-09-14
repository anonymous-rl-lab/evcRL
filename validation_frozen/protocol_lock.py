"""Preflight: bind assets, code identities, protocol fields and the 99-job list; write protocol_lock.json, preflight.json, jobs.csv."""
import csv, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / 'software' / 'EvcRL' / 'src'))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy, torch, scipy, evcrl
    from evcrl.profiles import profile_identity, PROFILES, CONTRACT_VERSION
    git = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', 'diff', 'HEAD', '--', ':!validation_frozen'], cwd=REPO, capture_output=True, text=True).stdout
    vis = REPO / 'visual'
    sources = {p: sha(REPO / p) for p in ['visual/visual_dev/pipeline.py', 'visual/visual_dev/vision_state.py', 'visual/visual_dev/renderer.py', 'visual/visual_dev/reevaluate.py',
                                          'visual/v19_deps/code/curve_layer.py', 'visual/v19_deps/code/study.py', 'visual/v19_deps/code/env20.py', 'visual/v19_deps/code/route20.py',
                                          'visual/v19_deps/code/models.py', 'visual/v19_deps/code/plant.py', 'visual/v19_deps/code/effcal.py', 'visual/v19_deps/code/scn/arterial_mini_profile.npy',
                                          'visual/visual_z/model.py', 'visual/visual_z/learner.py', 'visual/visual_z/contracts.py', 'visual/visual_z/replay.py', 'visual/visual_z/checkpoint.py', 'visual/visual_z/__init__.py',
                                          'software/EvcRL/src/evcrl/memory.py', 'software/EvcRL/src/evcrl/profiles.py', 'software/EvcRL/src/evcrl/execution.py', 'software/EvcRL/src/evcrl/_version.py',
                                          'timing/code/probe.py', 'timing/code/analyze.py', 'timing/deps/study.py', 'timing/deps/curve_layer.py', 'timing/deps/env20.py',
                                          'validation_frozen/run_pv.py', 'validation_frozen/run_pt.py']}
    # visual source identity through the repository-bound wrapper
    chk = subprocess.run([sys.executable, str(vis / 'reproduction' / 'reproduce_visual.py'), 'check', '--root', str(vis)], capture_output=True, text=True)
    identity = json.loads(chk.stdout) if chk.returncode == 0 else dict(status='FAIL', stderr=chk.stderr[-500:])
    sys.path.insert(0, str(REPO / 'timing' / 'code')); import probe as PT
    pt_fp = PT.source_hash(); freeze = json.loads((REPO / 'timing/audit/freeze.json').read_text())
    ck = {f'S{k}': str(vis / 'runs' / r / 'v4_pilot' / 'supervised' / 'final_nets.pt') for k, r in enumerate(['v4r', 'v4r_s1', 'v4r_s2'])}
    cfgs = {}
    for k, p in ck.items():
        c = torch.load(p, map_location='cpu', weights_only=False)['cfg']
        cfgs[k] = dict(seed=c['seed'], camera_seed_training=c['camera_seed'], pretrained_encoder=c['pretrained_encoder'], det_thr=c['det_thr'], hold_s=c.get('hold_s', 3.), extra_dim=int(c.get('extra_dim', 6)),
                       critic_action=c.get('critic_action'), assumed_green_remaining=c.get('assumed_green_remaining', 30.), lambda_c=c['lambda_c'], actor_output_scale=c.get('actor_output_scale'))
    detector = vis / 'runs/pretrain_v4/encoder.pt'
    wheel = REPO / 'software/EvcRL/dist/evcrl-0.0.1-py3-none-any.whl'
    reporting_prior = dict(timing_runs=sorted(p.name for p in (REPO / 'timing/runs').iterdir()), reporting_dir_exists=(REPO / 'timing/runs/reporting').exists(),
                           decision=json.loads((REPO / 'timing/reports/decision.json').read_text())['reporting_started'],
                           statement='Reporting conditions never generated and not used for actor/parameter selection (timing/reports/decision.json: reporting_started=false; only runs/development exists). They are prespecified extension conditions of the same 4 km route, one frozen actor.')
    lock = dict(
        task='TIV frozen supplementary validation (99 formal frozen-inference episodes: R1 54, R2 9, R3 36)', date='2026-09-14', new_training_steps=0, new_model_selection=False, parameter_search=False,
        source_commit=git('rev-parse', 'HEAD'), source_branch=git('rev-parse', '--abbrev-ref', 'HEAD'), source_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(), source_diff_nonempty=bool(diff.strip()),
        installed_package_version=evcrl.__version__, installed_module_path=str(Path(evcrl.__file__).resolve()), package_sha256=dict(wheel=sha(wheel), memory_py=sources['software/EvcRL/src/evcrl/memory.py']),
        simulator_sha256=dict(study_py=sources['visual/v19_deps/code/study.py'], env20_py=sources['visual/v19_deps/code/env20.py'], models_py=sources['visual/v19_deps/code/models.py'], plant_py=sources['visual/v19_deps/code/plant.py'], effcal_py=sources['visual/v19_deps/code/effcal.py'], route20_py=sources['visual/v19_deps/code/route20.py']),
        executor_sha256=dict(curve_layer_py=sources['visual/v19_deps/code/curve_layer.py'], pipeline_VisionEnv=sources['visual/visual_dev/pipeline.py'], jerk_limit=2.0, dt=0.5, a_comfort=1.5, assumed_green_remaining=30.0),
        memory_sha256=dict(archived_vision_state_py=sources['visual/visual_dev/vision_state.py'], injected_evcrl_memory_py=sources['software/EvcRL/src/evcrl/memory.py'], profiles_py=sources['software/EvcRL/src/evcrl/profiles.py'],
                           contract_version=CONTRACT_VERSION, profile_identity={n: profile_identity(n) for n in ('paper_v25', 'odometry_v26', 'guard_v26')}, rules={n: PROFILES[n]['memory'] for n in ('paper_v25', 'guard_v26')},
                           treatment='paper_v25 (legacy propagation v*dt, no guard) versus guard_v26 (displacement-consistent propagation AND inward 15 m crossing guard): a combined change, not an isolated guard effect'),
        camera_sha256=sources['visual/visual_dev/renderer.py'], metric_source_sha256=dict(study_episode_metrics=sources['visual/v19_deps/code/study.py'], recompute_script='validation_frozen/metrics.py (added after runs; hashed in manifest)'),
        detector_sha256=sha(detector), detector_path=str(detector), S0_checkpoint_sha256=sha(ck['S0']), S1_checkpoint_sha256=sha(ck['S1']), S2_checkpoint_sha256=sha(ck['S2']), checkpoint_paths=ck, checkpoint_configs=cfgs,
        PT_actor_A_sha256=sha(REPO / 'timing/weights/actor_A.pt'), visual_source_identity_check=identity, all_sources_sha256=sources,
        policy_input_schema=dict(legacy=13, z=64, meta=4, extra=6, total=87, note='13 memory-generated legacy channels, 64-dim Z from the 4-frame encoder, 4 metadata, 6 memory extras; extra_dim 6 for all three archived checkpoints'),
        image_and_noise_seed_scheme=dict(camera_base_seed=1000, appearance_seed='1000*1000 + condition_id', noise_seed='1000*1000 + 500 + condition_id', training_camera_seeds=[77, 78, 79], note='evaluation seeds only; per-condition streams retain the original condition_id and evalXX identity'),
        termination_contract_per_protocol=dict(R1_R2=dict(protocol='current', terminal_true_position_cmd0_branch=False, settle='arrived and v<=0.05 and |a|<=0.05', red_crossing='terminal', deadline_s=600, overshoot='x>4100 m terminal', t_budget=480, reward='archived L reward'),
                                              R3=dict(protocol='archived P-T', terminal_true_position_cmd0_branch=True, settle='arrived and v<=1e-6 and |a|<=1e-6', red_crossing='terminal', deadline_s=600, executor_information_range_m=1000, actor_input='13-dim with SPaT in both arms')),
        runtime_versions=dict(python=sys.version.split()[0], numpy=numpy.__version__, torch=torch.__version__, scipy=scipy.__version__, evcrl=evcrl.__version__),
        hardware=dict(cpu_count=os.cpu_count(), platform=platform.platform(), machine=platform.machine(), gpu='none used', threads='torch/OMP single thread per process'),
        floating_comparison_tolerances=dict(baseline_reproduction='exact (0.0) on R, time, I_j, energy, violation counts, override counts and full trace against the archived S records; any nonzero difference is reported as non-reproduction with its magnitude',
                                             pair_deltas='exact differences of recomputed metrics', trajectory_identity='1e-9 absolute on trace columns', override_flag='abs(a_next-a_prev)>J*dt+1e-7 (archived trace flag); numeric exceedance abs(j)>J+1e-7 reported separately'),
        reporting_prior_usage=reporting_prior,
        authorized_runtime_budget=dict(smoke_wall_s=600, pilot_cumulative_wall_s=3600, formal='estimated from measured per-episode rates; run within the current machine only', price='unknown (no billing rate available for this container)', external_compute=0),
        R3=dict(pt_source_hash=pt_fp, pt_freeze_hash=freeze['source_hash'], pt_freeze_matches=freeze['source_hash'] == pt_fp, conditions='conditions("reporting"): 3 packs x speeds 14/18 x offsets 15/45/75, pack->speed->offset', entry='validation_frozen/run_pt.py (archived probe.py reporting stage left gated)'))
    body = json.dumps(lock, sort_keys=True, default=str); lock['fingerprint'] = hashlib.sha256(body.encode()).hexdigest()
    (HERE / 'protocol_lock.json').write_text(json.dumps(lock, indent=1, default=str) + '\n')
    rows = list(csv.DictReader(open(HERE / 'jobs_99.csv'))); assert len(rows) == 99 and len({r['job_id'] for r in rows}) == 99
    for r in rows:
        r['fingerprint'] = lock['fingerprint']; r['policy_asset_sha256'] = (sha(ck[r['policy']]) if r['policy'] in ck else (sha(ck['S0']) + ' (S0 encoder/adapter path; constant command)' if r['policy'] == 'constant_u_plus_1' else lock['PT_actor_A_sha256']))
        r['detector_sha256'] = sha(detector) if r['experiment'] in ('R1', 'R2') else ''
        r['profile_identity'] = profile_identity(r['profile']) if r['profile'] else ''
    with open(HERE / 'jobs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    pre = dict(status='PASS' if identity.get('status') == 'PASS' and lock['R3']['pt_freeze_matches'] and not lock['source_diff_nonempty'] else 'CHECK',
               visual_identity=identity.get('status'), pt_freeze_matches=lock['R3']['pt_freeze_matches'], working_tree_clean_outside_validation=not lock['source_diff_nonempty'],
               assets_present={k: Path(v).exists() for k, v in ck.items()} | dict(detector=detector.exists(), actor_A=(REPO / 'timing/weights/actor_A.pt').exists()),
               checkpoint_configs=cfgs, reporting_prior_usage=reporting_prior, jobs=99, fingerprint=lock['fingerprint'])
    (HERE / 'preflight.json').write_text(json.dumps(pre, indent=1, default=str) + '\n'); print(json.dumps({k: pre[k] for k in ('status', 'visual_identity', 'pt_freeze_matches', 'working_tree_clean_outside_validation', 'fingerprint')}, indent=1))


if __name__ == '__main__': main()
