"""20 km arterial DP, done right: Lagrangian time, signal PHASE as state.

WHY THE PREVIOUS DP WAS WRONG. It carried elapsed time as a state, so every
cell's crossing time had to be rounded to a bin. The optimiser found speeds
whose crossing time rounded DOWN (50 m at 19.4 m/s is 2.577 s, charged as
2.50 s) and chattered between two adjacent levels for 277 of 400 cells to
harvest that. Its "1050 s" trip integrated to 1091 s. A reference that is 4 %
faster than physics is not a reference.

THE FIX
  time     leaves the state. Each transition costs  e + lambda * dt  with dt the
           EXACT crossing time. Nothing is rounded, so there is nothing to game.
           Sweeping lambda traces the energy-time front; each lambda gives one
           trajectory whose time and energy are integrated exactly on recovery.
  signals  need absolute time only modulo the 90 s cycle. So the state carries
           the clock PHASE  theta = t mod CYCLE  in 0.05 s bins (1800 of them,
           versus 4600 time bins before). Rounding now touches only WHICH phase
           the vehicle reaches a signal at - +-0.025 s per cell - and not the
           cost, so it buys the optimiser nothing.
  waiting  an explicit self-transition at zero speed at a signal boundary,
           circular in phase, costing (aux + lambda) per second.
  decel    capped at 2.0 m/s^2 (not the actuator's 3.5) so that a tracker
           under the 2 m/s^3 jerk limit can follow the solution. The reference
           has to be a FEASIBLE policy, not a bound.
  curves   speed capped one cell before the entry (approach margin), same
           reason.

The DP's own numbers are never the headline. The recovered trajectory is
tracked through env20 - continuous time, jerk limit, envelope, true SOC - and
THAT number is the reference. The DP is a policy generator.
"""
import sys
import time

import numpy as np

import env20 as E
import models as M
import route20 as R

CELL = 50.0
N_CELL = int(round(R.LENGTH / CELL))
N_LVL = 33
LEVELS = np.linspace(0.0, R.V_FREE, N_LVL)       # level 14 = 9.72 = curve cap
DPH = 0.05                                       # s, phase bin
N_PH = int(round(R.CYCLE / DPH))                 # 1800
A_ACC, A_DEC = E.A_HI, 2.0
INF = np.float32(np.inf)
WAIT = -2                                        # NJ marker for a wait step
SIG_MARGIN = 6.0                                 # s of green still left when
                                                 # crossing. The tracker lags
                                                 # the DP by ~5 s; without this
                                                 # the DP catches greens by 1 s
                                                 # that the tracked policy then
                                                 # misses, and stops on red.


def masks():
    vmax = np.full(N_CELL + 1, R.V_FREE)
    for a, b, vc in R.CURVES:
        i0, i1 = int(round(a / CELL)), int(round(b / CELL))
        vmax[max(i0 - 1, 0):i1 + 1] = vc
    lvl_max = np.searchsorted(LEVELS, vmax + 1e-9, side="right") - 1
    sig = {int(round(x / CELL)): k for k, x in enumerate(R.SIGNALS)}
    return lvl_max, sig


def tables(soc, T):
    v = LEVELS
    vm = 0.5 * (v[:, None] + v[None, :])
    a = (v[None, :] ** 2 - v[:, None] ** 2) / (2.0 * CELL)
    ok = (a <= A_ACC + 1e-9) & (a >= -A_DEC - 1e-9) & (vm > 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(ok, CELL / np.maximum(vm, 1e-9), np.inf)
    e = np.zeros_like(vm); f = np.zeros_like(vm)
    for j in range(N_LVL):
        for k in range(N_LVL):
            if ok[j, k]:
                p_b, p_f, _ = M.v1_power(vm[j, k], a[j, k], 0.0, soc, T)
                e[j, k] = p_b * dt[j, k]; f[j, k] = p_f * dt[j, k]
    return e, f, dt, ok


def solve(lam, soc, T, offsets, tabs=None):
    """One lambda -> one trajectory. Returns exact (T, E, F) and the path."""
    lvl_max, sig = masks()
    e, f, dt, ok = tabs if tabs is not None else tables(soc, T)
    cost = (e + lam * dt).astype(np.float32)
    dph = np.where(ok, np.round(dt / DPH), 0).astype(np.int64)
    w_cost = np.float32((M.AUX.p_base_dc + lam) * DPH)
    ar = np.arange(N_PH)

    V = np.full((N_CELL + 1, N_LVL, N_PH), INF, np.float32)
    NJ = np.full((N_CELL + 1, N_LVL, N_PH), -1, np.int16)
    NP = np.full((N_CELL + 1, N_LVL, N_PH), -1, np.int16)
    V[0, N_LVL - 1, 0] = 0.0

    for i in range(N_CELL):
        green = None
        if i in sig:
            k_s = sig[i]
            green = np.array([R.signal_green(R.SIGNALS[k_s], p * DPH, offsets[k_s]) and
                              R.signal_green(R.SIGNALS[k_s], p * DPH + SIG_MARGIN, offsets[k_s])
                              for p in range(N_PH)])
            # wait at rest, circular running relaxation, two passes
            w = V[i, 0]
            for p in range(2 * N_PH):
                cur, prv = p % N_PH, (p - 1) % N_PH
                cand = w[prv] + w_cost
                if cand < w[cur]:
                    w[cur] = cand; NJ[i, 0, cur] = WAIT; NP[i, 0, cur] = prv
        for j in range(int(lvl_max[i]) + 1):
            src = V[i, j]
            if not np.isfinite(src).any():
                continue
            if green is not None:
                src = np.where(green, src, INF)
            for k in range(int(lvl_max[i + 1]) + 1):
                if not ok[j, k]:
                    continue
                d = int(dph[j, k])
                cand = np.roll(src, d) + cost[j, k]
                tgt = V[i + 1, k]
                m = cand < tgt
                if m.any():
                    tgt[m] = cand[m]
                    NJ[i + 1, k][m] = j
                    NP[i + 1, k][m] = ((ar - d) % N_PH)[m]

    p_end = int(np.argmin(V[N_CELL, 0]))
    if not np.isfinite(V[N_CELL, 0, p_end]):
        return None
    # recover the path exactly
    i, j, p = N_CELL, 0, p_end
    xs, vs = [R.LENGTH], [0.0]
    T_ex = E_ex = F_ex = 0.0; waits = 0.0
    while i > 0:
        jp, pp = int(NJ[i, j, p]), int(NP[i, j, p])
        if jp == WAIT:
            T_ex += DPH; E_ex += M.AUX.p_base_dc * DPH; waits += DPH
            p = pp
            continue
        if jp < 0:
            return None
        T_ex += dt[jp, j]; E_ex += e[jp, j]; F_ex += f[jp, j]
        i -= 1; j, p = jp, pp
        xs.append(i * CELL); vs.append(float(LEVELS[j]))
    return dict(lam=lam, T=T_ex, E_Wh=E_ex / 3600.0, F_Wh=F_ex / 3600.0,
                wait=waits, xs=np.array(xs[::-1]), vs=np.array(vs[::-1]))


def track(xs, vs, soc, T, offsets):
    """The DP trajectory through env20: continuous time, jerk limit, envelope,
    true SOC. This is the reference number."""
    env = E.Route20(soc, T, offsets=list(offsets))
    def pi(e, o):
        # target read a short distance AHEAD (1 m + one step of travel): a DP
        # path with an interior zero (a wait realised as a stop-and-go cell)
        # otherwise dead-locks the tracker at v = 0 with target 0.
        v_t = float(np.interp(e.x, xs, vs))
        if v_t < 0.5 and e.x < R.LENGTH - 5.0:
            v_t = 0.5
        for ca, cb, cvc in R.CURVES:
            if cb > e.x - 1e-9:
                d = max(ca - E.MARGIN - e.x, 0.0)
                v_t = min(v_t, np.sqrt(max(cvc ** 2 + 2.0 * A_DEC * d, 0.0)))
        return (v_t - e.v) / E.DT
    return env.rollout(pi)


if __name__ == "__main__":
    soc = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
    Tc = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    off = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    T = Tc + 273.15
    print("=" * 78)
    print(f"LAGRANGIAN DP  20 km arterial   {N_CELL} x {CELL:.0f} m, {N_LVL} levels, "
          f"phase bin {DPH} s, decel cap {A_DEC} m/s^2")
    print(f"  SOC {soc}, T {Tc:+.0f} C, signal offset {off:.0f} s, budget "
          f"{E.T_BUDGET:.0f} s")
    print("=" * 78)
    tabs = tables(soc, T)
    print(f"\n  {'lambda':>8}{'DP T':>9}{'DP E Wh':>10}{'DP fric':>9}{'wait s':>8} | "
          f"{'trk T':>8}{'trk E':>9}{'Wh/km':>8}{'trk fric':>9}{'stops':>6}{'off':>4}")
    rows = []
    t0 = time.time()
    for lam in (8000., 8400., 8700., 9000., 9300., 9600., 10000.):
        r = solve(lam, soc, T, (off, off), tabs)
        if r is None:
            print(f"  {lam:>8.1f}   infeasible"); continue
        tr = track(r["xs"], r["vs"], soc, T, (off, off))
        r["trk"] = tr; rows.append(r)
        print(f"  {lam:>8.1f}{r['T']:>9.1f}{r['E_Wh']:>10.1f}{r['F_Wh']:>9.1f}{r['wait']:>8.1f} | "
              f"{tr['t']:>8.1f}{tr['E_Wh']:>9.1f}{tr['Wh_km']:>8.2f}{tr['fric_Wh']:>9.1f}"
              f"{tr['stops']:>6}{tr['off']:>4}", flush=True)
    print(f"  ({time.time()-t0:.0f} s for the sweep)")

    feas = [r for r in rows if r["trk"]["arrived"] and r["trk"]["off"] == 0
            and r["trk"]["t"] <= E.T_BUDGET + 1e-6]
    if feas:
        b = min(feas, key=lambda r: r["trk"]["E_Wh"])
        print(f"\n  REFERENCE (tracked DP, within {E.T_BUDGET:.0f} s): "
              f"{b['trk']['E_Wh']:.1f} Wh = {b['trk']['Wh_km']:.2f} Wh/km at "
              f"{b['trk']['t']:.1f} s, friction {b['trk']['fric_Wh']:.1f} Wh, "
              f"{b['trk']['stops']} stop(s), R {b['trk']['R']:.3f}   [lambda {b['lam']}]")
        d = np.abs(b["trk"]["t"] - b["T"])
        print(f"  DP-vs-tracked time gap {d:.1f} s, energy gap "
              f"{100*(b['trk']['E_Wh']-b['E_Wh'])/b['E_Wh']:+.2f} %  "
              f"(was 42 s / +0.5 % with time-as-state)")
        np.save("out/dp20L_ref.npy", np.array([b["xs"], b["vs"]]))
    print(f"\n  hand-policy family, same conditions:")
    print(f"  {'a_b':>6}{'R':>10}{'trip s':>9}{'E Wh':>9}{'Wh/km':>8}{'fric':>7}{'vs ref':>8}")
    for ab in (0.3, 0.4, 0.5, 0.8, 1.2):
        env = E.Route20(soc, T, offsets=[off, off])
        h = env.rollout(E.make_pi_gentle(ab))
        gap = 100 * (h["E_Wh"] - b["trk"]["E_Wh"]) / b["trk"]["E_Wh"] if feas else float("nan")
        okk = h["arrived"] and h["off"] == 0 and h["t"] <= E.T_BUDGET + 1e-6
        print(f"  {ab:>6.1f}{h['R']:>10.3f}{h['t']:>9.1f}{h['E_Wh']:>9.1f}{h['Wh_km']:>8.2f}"
              f"{h['fric_Wh']:>7.1f}{gap:>+8.2f}%" + ("" if okk else "  (over budget)"))
