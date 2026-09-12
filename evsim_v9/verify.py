"""Pre-flight verification. run_all.sh refuses to start unless this exits 0.

Checks, in order:
  1. shaping invariance  -- total potential-based shaping over an episode must be the
     CONSTANT c*L for every policy (Ng et al. 1999 requires Phi(absorbing)=0).
     This is the check that was missing when the v4 package shipped.
  2. objective consistency -- the step rewards of a trajectory sum to its reported R_L.
  3. action sensitivity A1 -- fraction of on-policy states where +-0.2 on the action
     does not change the executed acceleration.
  4. shaping constant    -- c must equal the plant's c* = argmin g(v)/v, not a tuned value.
"""
import sys, numpy as np, copy
import env20 as E, route20 as R, models as M

FAIL = []
def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok: FAIL.append(name)

E.REWARD_MODE = "L"; E.CURVE_MODE = "envelope"
LAM, E0, L = E.LAM_T, 1e5, R.LENGTH

print(f"\nverify.py  route L = {L:.0f} m\n")

# ---- 4. the constant, from the plant, before anything else
vg = np.linspace(2, 22.222, 4000)
g = np.array([M.v1_power(v, 0., 0., .85, 288.15)[0] / E0 + LAM for v in vg])
i0 = int(np.argmin(g / vg)); V_STAR, C_STAR = float(vg[i0]), float((g / vg)[i0])
C_USE = 0.00844
check("shaping constant", abs(C_USE - C_STAR) < 5e-5,
      f"c = {C_USE:.5f}, plant c* = g(v*)/v* = {C_STAR:.5f} at v* = {V_STAR:.2f} m/s "
      f"({100*(C_USE/C_STAR-1):+.1f}%)")

# ---- 1. shaping invariance
def run(pi, shape):
    E.SHAPE_C = shape
    env = E.Route20(0.85, 288.15, offsets=[0.0]*len(R.SIGNALS)); env.reset(); Rr = 0.0
    while True:
        o, r, d, i = env.step(pi(env)); Rr += r
        if d: break
    return Rr, env.x, i["arrived"], env
tgts = [("finisher", 18.0), ("slow", 9.0), ("quitter", 4.0), ("stander", None)]
tot, ok1 = [], True
for nm, tv in tgts:
    pi = (lambda e: -3.5) if tv is None else (lambda e, t=tv: float(np.clip((t-e.v)/E.DT, -3.5, 2.6)))
    Ru, xu, au, _ = run(pi, 0.0)
    Rs, xs, _, _ = run(pi, C_USE)
    tot.append(Rs - Ru)
    print(f"        {nm:<9} x_T={xu:8.0f}  arr={int(au)}  R={Ru:9.3f}  shaping={Rs-Ru:.9f}")
ok1 = max(tot) - min(tot) < 1e-9 and abs(tot[0] - C_USE*L) < 1e-6
check("shaping invariance", ok1,
      f"total shaping is {'constant' if max(tot)-min(tot)<1e-9 else 'POLICY-DEPENDENT'} "
      f"= {np.mean(tot):.6f}, must equal c*L = {C_USE*L:.6f}")

# ---- 2. objective consistency
E.SHAPE_C = 0.0
pi = lambda e: float(np.clip((17.0-e.v)/E.DT, -3.5, 2.6))
Rr, x, arr, env = run(pi, 0.0)
true = -env.e_batt/E0 - LAM*env.t - (0.0 if arr else E.CTG_PER_KM*max(L-env.x, 0.0)/1000.0)
check("objective consistency", abs(Rr - true) < 1e-6,
      f"sum of step rewards {Rr:.6f} vs -E/E0 - lam t - ctg = {true:.6f}")

# ---- 3. action sensitivity
E.SHAPE_C = C_USE
rng = np.random.default_rng(0); dead = n = 0
for _ in range(3):
    env = E.Route20(0.85, 288.15, offsets=[float(rng.uniform(0, 90))]*len(R.SIGNALS)); env.reset()
    while True:
        u = float(np.clip(np.clip((17.0-env.v)/E.DT/2.6, -1, 1) + rng.normal(0, 0.15), -1, 1))
        a = u*(E.A_HI if u > 0 else -E.A_LO)
        u2 = float(np.clip(u+0.2, -1, 1)); a2 = u2*(E.A_HI if u2 > 0 else -E.A_LO)
        g1 = copy.deepcopy(env); g2 = copy.deepcopy(env); g1.step(a); g2.step(a2)
        dead += abs(g1.v - g2.v) < 1e-9; n += 1
        o, r, d, i = env.step(a)
        if d: break
E.SHAPE_C = 0.0
check("action sensitivity A1", dead/n < 0.5, f"{100*dead/n:.1f}% dead over {n} on-policy states (must be < 50%)")

# ---- 5. the arrival threshold and the bootstrap hole
pace = L / E.T_BUDGET
crit = pace / 1.25
print(f"\n  [INFO] episode cut-off is t_end = t + 1.25 (L-x)/pace, pace = {pace:.2f} m/s.")
print(f"         Arriving therefore requires an average of {crit:.2f} m/s FROM ANY START -- the criterion")
print(f"         is scale-free, so exploring starts near the finish are no easier than a cold start.")
print(f"         A policy that settles below {crit:.2f} m/s completes NOTHING and its replay buffer")
print(f"         holds no evidence of what a finished trip is worth.  This is a learning limitation;")
print(f"         priming is not a demonstrated long-run repair; see README.")
E.SHAPE_C = 0.0
env = E.Route20(0.85, 288.15, offsets=[0.0]*len(R.SIGNALS)); env.reset()
import numpy as _np
for vt, want in ((14.0, False), (18.0, True)):
    e = E.Route20(0.85, 288.15, offsets=[0.0]*len(R.SIGNALS)); e.reset()
    while True:
        o, r, d, i = e.step(float(_np.clip((vt - e.v)/E.DT, -3.5, 2.6)))
        if d: break
    got = bool(i["arrived"])
    check(f"constant {vt:.0f} m/s cruise {'arrives' if want else 'does NOT arrive'}", got == want,
          f"x_T = {e.x:.0f} m, t = {e.t:.0f} s, arrived = {got} (threshold {crit:.2f} m/s)")

# ---- 6. Markov property of the observation w.r.t. the episode deadline
def _at(x0, v0, t0, explore):
    e = E.Route20(0.85, 288.15, offsets=[0.0]*len(R.SIGNALS)); e.reset()
    if explore: e.reset_from(x0, v0, t0)
    e.x, e.v, e.t = 10000.0, 18.0, 600.0
    return _np.asarray(e.obs()), e.t_end
import numpy as _np
oA, tA = _at(0.0, 0.0, 0.0, False)
oB, tB = _at(8000.0, 18.0, 550.0, True)
check("observation is Markov in the deadline", not _np.allclose(oA, oB),
      f"same (x,v,t) with deadlines {tA:.0f} s and {tB:.0f} s must give different observations "
      f"(clock channel {oA[11]:.4f} vs {oB[11]:.4f})")

print()
if FAIL:
    print(f"VERIFICATION FAILED: {', '.join(FAIL)}\nDo not train. Fix the environment first.\n"); sys.exit(1)
print("LEGACY VERIFICATION PASSED. Also run test_regressions.py; this is not a safety certificate.\n"); sys.exit(0)
