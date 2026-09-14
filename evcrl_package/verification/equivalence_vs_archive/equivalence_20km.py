"""20 km structured route: evcrl Backend vs archived env20.Route20 (EVSIM_ROUTE unset; evsim_v9 sources with the exported 20 km profile)."""
import os, sys, json
os.environ.pop('EVSIM_ROUTE', None)
sys.path.insert(0, '/home/user/evcrl/evsim_v9'); sys.path.insert(0, sys.argv[1])
import numpy as np, env20 as E, evcrl

def run(seed, offsets, soc, T, v0, n=800):
    orig = E.Route20(soc, T, offsets=offsets); orig.reset(); orig.v = v0; orig._was_moving = v0 >= .1
    env = evcrl.EcoDriveEnv(mode='structured', route_length=20000.); env.reset(seed=seed, options=dict(soc=soc, temperature=T, speed=v0, signal_offsets=offsets))
    rng = np.random.default_rng(seed); A = []; B = []; OA = []; OB = []
    for u in np.clip(rng.normal(0.3, 0.5, n), -1, 1):
        cmd = evcrl.normalized_to_command(u); done = False
        for _ in range(4):
            o, r, d, info = orig.step(cmd); A.append((orig.v, orig.a, r)); OA.append(o)
            if d: done = True; break
        o2, r2, t, tr, info2 = env.step(np.array([u]))
        for q in info2['substeps']: B.append((q['v1'], q['applied_a'], q['reward']))
        OB.append(o2)
        if done or t or tr:
            print('  ended orig done=%s t=%.1f x=%.1f | pkg reason=%s t=%.1f x=%.1f' % (done, orig.t, orig.x, info2['termination_reason'], info2['time_s'], env.vehicle.x)); break
    A = np.array(A); B = np.array(B); OA = np.array(OA)[3::4]; OB = np.array(OB); m = min(len(OA), len(OB))
    d1 = float(np.max(np.abs(A - B))) if A.shape == B.shape else float('inf'); d2 = float(np.max(np.abs(OA[:m] - OB[:m])))
    print('  substeps %d/%d  max|diff| v,a,r = %.3e   obs = %.3e   final x %.6f vs %.6f e_batt %.6f vs %.6f' % (len(A), len(B), d1, d2, orig.x, env.vehicle.x, orig.e_batt, env.vehicle.e_batt))
    return A.shape == B.shape and d1 == 0. and d2 == 0. and orig.x == env.vehicle.x and orig.e_batt == env.vehicle.e_batt

ok = all([run(0, [10., 50.], .85, 288.15, 22.22), run(1, [70., 20.], .6, 263.15, 16.), run(2, [0., 0.], .95, 298.15, 0.)])
print(json.dumps({'structured_20km': ok})); sys.exit(0 if ok else 1)
