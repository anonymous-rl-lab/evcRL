"""Software contract of EvcRL: Gymnasium API, action semantics, termination, isolation, snapshots, identities."""
import copy
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import evcrl
from evcrl import EcoDriveEnv, VisionMemory, Executor, OracleDetector

U = lambda x: np.array([x], np.float32)


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, float) and math.isnan(a): assert math.isnan(b)
    else: assert a == b


def run(env, commands, seed=0, options=None):
    env.reset(seed=seed, options=options); out = []
    for u in commands:
        o, r, t, tr, info = env.step(U(u)); out.append((r, t, tr, info['time_s'], info['energy_Wh'], tuple(info['applied_a'])))
        if t or tr: break
    return out


# ------------------------------------------------------------------ import hygiene
def test_import_is_light_and_side_effect_free():
    assert 'torch' not in sys.modules and 'scipy' not in sys.modules and 'matplotlib' not in sys.modules
    import evcrl.backend as B, evcrl.camera as C
    before = (B.J_MAX, B.CURVE_MODE, B.T_BUDGET, C.CELL_ASSIGN)
    e = EcoDriveEnv(jerk=4.); e.reset(seed=0); e.step(U(.3))
    f = EcoDriveEnv(mode='structured', route_length=20000.); f.reset(seed=0); f.step(U(.3))
    assert (B.J_MAX, B.CURVE_MODE, B.T_BUDGET, C.CELL_ASSIGN) == before   # module constants untouched by instances
    assert e.vehicle.cfg.J_MAX == 4. and f.vehicle.cfg.J_MAX == 2. and e.vehicle.cfg.CURVE_MODE == 'envelope' and f.vehicle.cfg.CURVE_MODE == 'penalty'


# ------------------------------------------------------------------ gymnasium contract
@pytest.mark.parametrize('name', ['EvcRL-Camera-v0', 'EvcRL-Structured-v0', 'EvcRL-StructuredMini-v0'])
def test_registered_gym_contract(name):
    e = gym.make(name); check_env(e.unwrapped, skip_render_check=False); e.close()
    e = gym.make(name, render_mode='rgb_array'); e.reset(seed=0); frame = e.render(); assert frame.shape == (96, 160, 3) and frame.dtype == np.uint8; e.close()


def test_gym_make_forwards_kwargs():
    e = gym.make('EvcRL-Camera-v0', profile='guard_v26', jerk=3., detector=OracleDetector(seed=1)).unwrapped
    assert e.profile == 'guard_v26' and e.jerk == 3. and e.detector_kind == 'oracle'
    _, info = e.reset(seed=0); assert info['detector_kind'] == 'oracle' and info['profile_identity'] == evcrl.profile_identity('guard_v26')


def test_action_parsing_and_command_mapping():
    e = EcoDriveEnv(); e.reset(seed=0)
    _, _, _, _, info = e.step(0.5)                       # scalar accepted
    assert info['command_u'] == .5 and info['command_a'] == pytest.approx(1.3) and not info['command_clipped']
    _, _, _, _, info = e.step([-0.5]); assert info['command_a'] == pytest.approx(-1.75)
    _, _, _, _, info = e.step(U(1.5)); assert info['command_u'] == 1. and info['command_clipped']
    for bad in (np.array([np.nan]), np.array([0.1, 0.2]), 'x'):
        with pytest.raises((ValueError, TypeError)): e.step(bad)
    strict = EcoDriveEnv(clip_actions=False); strict.reset(seed=0)
    with pytest.raises(ValueError): strict.step(U(1.5))
    for a in (2.6, -3.5, 0., 1.3, -0.7): assert evcrl.normalized_to_command(evcrl.command_to_normalized(a)) == pytest.approx(a)


def test_command_holding_and_execution_record():
    e = EcoDriveEnv(); e.reset(seed=1); o, r, _, _, info = e.step(U(.5))
    assert len(info['substeps']) == 4 and info['elapsed_dt'] == 2. and len(info['applied_a']) == 4
    assert all(row['command_u'] == .5 and row['command_a'] == pytest.approx(1.3) for row in info['substeps'])
    assert info['applied_a'] == [row['applied_a'] for row in info['substeps']]
    assert len(set(info['applied_a'])) > 1                                        # the jerk bound makes the applied accelerations differ from the held command
    assert abs(r - sum(sum(row['reward_components'].values()) for row in info['substeps'])) < 1e-12
    for row in info['substeps']:
        assert row['v1'] == pytest.approx(max(row['v0'] + .5 * row['applied_a'], 0.)) and row['jerk_exceeded'] is False
    assert 'x' not in o and 'truth' not in info and 'labels' not in o and 'offsets' not in info


def test_reward_matches_archived_ledger():
    e = EcoDriveEnv(mode='structured', route_length=4000.); out = run(e, np.linspace(-1, 1, 60), seed=3)
    b = e.vehicle; expected = -b.e_batt / 1e5 - b.cfg.LAM_T * b.t - b.e_overspeed_pen - b.terminal_pen
    assert e.total_reward == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------------------ termination
def test_intrinsic_deadline_is_terminal_and_external_cap_is_truncation():
    for kw, expected in [({'task_deadline_s': .5}, (True, False)), ({'task_deadline_s': None, 'max_episode_seconds': .5}, (False, True))]:
        e = EcoDriveEnv(**kw); e.reset(seed=0); _, r, t, u, info = e.step(U(0.)); assert (t, u) == expected
        assert len(info['substeps']) == 1 and info['termination_reason'] == ('task_deadline' if t else None)
        assert (info['substeps'][0]['reward_components']['events_and_other'] < -1.) == t   # terminal cost-to-go only for the task deadline
        with pytest.raises(RuntimeError): e.step(U(0.))
    with pytest.raises(RuntimeError): EcoDriveEnv().step(U(0.))


def test_blind_camera_env_runs_the_red_and_is_judged():
    e = EcoDriveEnv(); out = run(e, [1.] * 400, seed=0, options=dict(signal_offset=40.))
    assert out[-1][1] and e.last_substeps[-1]['reason'] == 'feasible'
    assert e.vehicle.n_violation == 1


@pytest.mark.parametrize('offset', [0., 40., 70.])
def test_oracle_episode_settles_without_violation(offset):
    e = EcoDriveEnv(detector=OracleDetector(seed=0)); e.reset(seed=0, options=dict(signal_offset=offset)); n = fb = 0
    while True:
        _, _, t, tr, info = e.step(U(.5)); n += len(info['substeps']); fb += info['fallback_substeps']
        if t or tr: break
    assert t and not tr and info['termination_reason'] == 'settled' and info['time_s'] < 600.
    assert e.vehicle.n_violation == 0 and e.vehicle.n_offroad == 0
    assert e.route.length - 1. <= e.vehicle.x <= e.route.length + 100. and e.vehicle.v <= .05
    assert fb <= 5 and all(q['fallback'] is False or q['v0'] < .1 for q in e.last_substeps)   # archived rules: fallback only while creeping to a standstill
    assert info['memory']['end']['d_est'] is not None and info['memory']['sig']['seen'] is False


def test_detector_output_validation():
    for bad in (lambda rgb: None, lambda rgb: ({'lamp': {'score': 1., 'dist_m': 3.}}, None), lambda rgb: ({}, [1., 0.]), lambda rgb: ({'end_marker': {'score': float('nan'), 'dist_m': 1.}}, None)):
        e = EcoDriveEnv(detector=bad)
        with pytest.raises(ValueError): e.reset(seed=0)
    with pytest.raises(TypeError): EcoDriveEnv(detector=3)


# ------------------------------------------------------------------ determinism, isolation, snapshots
def test_seed_determinism_and_independence():
    cmds = np.sin(np.arange(40)); ea, eb, ec = (EcoDriveEnv(detector=OracleDetector()) for _ in range(3))
    a = run(ea, cmds, seed=7); b = run(eb, cmds, seed=7); c = run(ec, cmds, seed=8)
    equal(a, b); assert ea.vehicle.offsets == eb.vehicle.offsets and np.array_equal(ea.rgb, eb.rgb)
    assert ea.vehicle.offsets != ec.vehicle.offsets and not np.array_equal(ea.rgb, ec.rgb)   # signal offsets and camera appearance follow the seed
    a2 = run(EcoDriveEnv(detector=OracleDetector()), [.5] * 400, seed=7, options=dict(signal_offset=40.)); c2 = run(EcoDriveEnv(detector=OracleDetector()), [.5] * 400, seed=8, options=dict(signal_offset=70.))
    assert a2[-1][3] != c2[-1][3]                                                              # different signal timing changes the episode
    e = EcoDriveEnv(); e.reset(seed=7, options=dict(camera_seed=3)); f1 = e.rgb.copy(); e.reset(seed=99, options=dict(camera_seed=3)); assert np.array_equal(f1, e.rgb)


def test_multiple_instances_in_one_process_are_isolated():
    a = EcoDriveEnv(jerk=2); b = EcoDriveEnv(jerk=4, profile='odometry_v26'); ref = EcoDriveEnv(jerk=2)
    for e in [a, b, ref]: e.reset(seed=41)
    for _ in range(7):
        got = a.step(U(.6)); b.step(U(-.6)); expected = ref.step(U(.6)); equal(got, expected)


def test_snapshot_round_trip_mid_episode():
    e = EcoDriveEnv(detector=OracleDetector(seed=2)); e.reset(seed=5)
    for _ in range(60): e.step(U(.5))
    saved = e.get_state(); assert saved['profile_identity'] == e.profile_identity and saved['schema'] == 2
    expected = [e.step(U(.4)) for _ in range(6)]
    e.set_state(saved); actual = [e.step(U(.4)) for _ in range(6)]; equal(actual, expected)
    fresh = EcoDriveEnv(detector=OracleDetector(seed=99)); fresh.set_state(saved); equal([fresh.step(U(.4)) for _ in range(6)], expected)
    other = EcoDriveEnv(detector=OracleDetector(), profile='odometry_v26')
    with pytest.raises(ValueError): other.set_state(saved)
    with pytest.raises(ValueError): EcoDriveEnv().set_state(saved)                     # detector attachment differs
    tampered = copy.deepcopy(saved); tampered['source_identity'] = '0' * 64
    with pytest.raises(ValueError): e.set_state(tampered)
    with pytest.warns(UserWarning): e.set_state(tampered, strict_source=False)


def test_structured_snapshot_and_pickle():
    e = EcoDriveEnv(mode='structured', route_length=20000.); e.reset(seed=0)
    for _ in range(10): e.step(U(.7))
    s = e.get_state(); assert s['camera'] is None
    expected = [e.step(U(-.2)) for _ in range(3)]; e.set_state(s); equal([e.step(U(-.2)) for _ in range(3)], expected)
    clone = pickle.loads(pickle.dumps(e)); equal([clone.step(U(.1)) for _ in range(3)], [e.step(U(.1)) for _ in range(3)])


# ------------------------------------------------------------------ frozen identities
def test_profile_identities_are_pinned():
    expected = json.loads((Path(__file__).parent / 'expected_identities.json').read_text())
    assert evcrl.CONTRACT_VERSION == expected['contract']
    for name, h in expected['profiles'].items(): assert evcrl.profile_identity(name) == h, name
    assert evcrl.get_profile('paper_v25')['memory']['propagation'] == 'legacy' and evcrl.get_profile('paper_v25')['memory']['guard'] is False
    assert len({evcrl.profile_identity(n) for n in evcrl.PROFILES}) == len(evcrl.PROFILES)
    p = evcrl.get_profile('paper_v25'); p['memory']['margin_abs'] = 7.   # copies cannot mutate the registry
    assert evcrl.PROFILES['paper_v25']['memory']['margin_abs'] == 6.


def test_profile_changes_the_memory_rules_only():
    a = EcoDriveEnv(profile='paper_v25'); b = EcoDriveEnv(profile='odometry_v26'); c = EcoDriveEnv(profile='guard_v26')
    for e in (a, b, c): e.reset(seed=0)
    assert (a.memory.propagation, b.memory.propagation, c.memory.propagation) == ('legacy', 'consistent', 'consistent') and c.memory.guard and not b.memory.guard
    assert a.identity != b.identity != c.identity


# ------------------------------------------------------------------ memory / executor unit checks (archived rules)
def test_guard_is_prospective_and_color_update_remains_live():
    opts = dict(init_votes=1, propagation='consistent')
    old = VisionMemory(**opts); new = VisionMemory(**opts, guard=True)
    for m in [old, new]:
        m.sig.update(d_line=18., seen=True, phase='green', age=0., conf=1.)
        m.update({'traffic_light': {'score': 1., 'dist_m': 16.}}, [1., 0., 0., 0., 0.], 5., .5, odom_ds=0.)
    assert old.sig['d_line'] < 15 and new.sig['d_line'] == 18.
    assert old.sig['phase'] == new.sig['phase'] == 'red'
    assert new.update_diagnostic['range_guard']


def test_odometry_removes_endpoint_propagation_difference():
    a = VisionMemory(); b = VisionMemory(propagation='consistent')
    for m in [a, b]: m.sig.update(d_line=10., seen=True, age=0.)
    a.update({}, None, .15, .5, odom_ds=.05); b.update({}, None, .15, .5, odom_ds=.05)
    assert abs(a.sig['d_line'] - 9.925) < 1e-12 and abs(b.sig['d_line'] - 9.95) < 1e-12
    with pytest.raises(ValueError): b.update({}, None, .15, .5)


@pytest.mark.parametrize('j', [2., 3., 4.])
def test_release_inverse_and_interval(j):
    c = Executor(j)
    for v in np.linspace(0, 8, 41):
        b = c.brake_bound(v); loss = .5 * sum(max(b - i * j * .5, 0) for i in range(5)); assert loss <= v + 1e-10
        if b < 3.5 - 1e-6:
            b2 = b + 1e-5; assert .5 * sum(max(b2 - i * j * .5, 0) for i in range(5)) > v
    bounds, reason = c.interval(16., 0., [(100., 0.)], vmax=80 / 3.6); assert bounds is not None
    for u in np.linspace(*bounds, 17):
        vn = 16 + .5 * u; assert .25 * (16 + vn) + c.backup_distance(vn, u) <= 100 + 1e-6
    with pytest.raises(ValueError): Executor(2.5)
    with pytest.raises(ValueError): EcoDriveEnv(jerk=5.)


def test_route_profile_matches_exported_grid():
    r4, r20 = evcrl.Route(4000.), evcrl.Route(20000.)
    assert r4.profile_limit(0.) == 22.22 and r4.profile_limit(1500.) == 9.72 and r4.profile_limit(4000.) == 0.
    assert r20.profile_limit(11000.) == 9.72 and r20.profile_limit(19995.) == 22.22 and r20.profile_limit(20000.) == 0.
    with pytest.raises(ValueError): evcrl.Route(5000.)
    with pytest.raises(ValueError): EcoDriveEnv(mode='camera', route_length=20000.)
