"""Information-matched DP: the optimum a controller can reach when it learns the
signal phase only inside SPAT_RANGE (400 m), exactly as the TD3 observation does.

Backward DP over 50 m cells with cost  e + lambda*dt  (same tables, same
Lagrangian time price as dp20L). Outside a SPaT zone the state is the speed
level only; inside a zone [x_sig - SPAT_RANGE, x_sig] it is (level, phase).
Signal offsets are independent and uniform on [0, CYCLE), so at the zone entry
the value seen from upstream is the MEAN over phase of the in-zone value:
upstream decisions cannot depend on the phase, which is the information
constraint. Downstream of a signal the phase of that signal is irrelevant, so
the state drops back to level only.

The recovered policy is non-anticipative by construction. It is then rolled
out per offset and tracked through env20 exactly like the full-information
DP, so the three references (human / info-matched DP / full-info DP) are
measured by the same integrator.
"""
import json, sys, time
import numpy as np
import env20 as E, models as M, route20 as R
import dp20L as D

CELL, N_CELL, N_LVL, LEVELS, DPH, N_PH = D.CELL, D.N_CELL, D.N_LVL, D.LEVELS, D.DPH, D.N_PH
INF = np.float32(np.inf)
ZONE_CELLS = int(round(E.SPAT_RANGE / CELL))


def solve(lam, soc, T, tabs=None):
    lvl_max, sig = D.masks()
    e, f, dt, ok = tabs if tabs is not None else D.tables(soc, T)
    cost = (e + lam * dt).astype(np.float32)
    dph = np.where(ok, np.round(dt / DPH), 0).astype(np.int64)
    w_cost = np.float32((M.AUX.p_base_dc + lam) * DPH)
    sig_cells = sorted(sig)
    zone_of = {}                                   # cell boundary -> signal index, if inside a zone
    for c in sig_cells:
        for i in range(c - ZONE_CELLS, c + 1):
            zone_of[i] = sig[c]
    # J_out[i][j] outside zones; J_in[i][j][p] inside zones (dict by boundary)
    J_out = np.full((N_CELL + 1, N_LVL), INF, np.float32)
    J_in = {}
    K_out = np.full((N_CELL + 1, N_LVL), -1, np.int16)          # argmin next level
    K_in = {}
    J_out[N_CELL, 0] = 0.0
    for i in range(N_CELL - 1, -1, -1):
        in_zone = i in zone_of
        is_sig = i in sig
        nxt_in = (i + 1) in zone_of and not is_sig             # next boundary carries the same phase
        if in_zone:
            Ji = np.full((N_LVL, N_PH), INF, np.float32); Ki = np.full((N_LVL, N_PH), -1, np.int16)
            k_s = zone_of[i]
            green = None
            if is_sig:
                green = np.array([R.signal_green(0.0, p * DPH, 0.0) and R.signal_green(0.0, p * DPH + D.SIG_MARGIN, 0.0)
                                  for p in range(N_PH)])
            for j in range(int(lvl_max[i]) + 1):
                for k in range(int(lvl_max[i + 1]) + 1):
                    if not ok[j, k]: continue
                    d = int(dph[j, k])
                    if nxt_in:
                        cand = cost[j, k] + np.roll(J_in[i + 1][k], -d)     # phase advances by d
                    else:
                        cand = np.full(N_PH, cost[j, k] + J_out[i + 1, k], np.float32)
                    if green is not None:
                        cand = np.where(green, cand, INF)
                    m = cand < Ji[j]
                    Ji[j][m] = cand[m]; Ki[j][m] = k
            if is_sig:                                            # waiting at rest, circular relaxation
                w = Ji[0]; kw = Ki[0]
                for _ in range(2):
                    for p in range(N_PH - 1, -1, -1):
                        nxt = (p + 1) % N_PH
                        cand = w[nxt] + w_cost
                        if cand < w[p]:
                            w[p] = cand; kw[p] = D.WAIT
            J_in[i] = Ji; K_in[i] = Ki
            if i == min(c for c in zone_of if zone_of[c] == k_s):   # zone entry: upstream sees the phase-mean
                with np.errstate(invalid="ignore"):
                    J_out[i] = np.where(np.isfinite(Ji).all(1), Ji.mean(1), INF)
        else:
            for j in range(int(lvl_max[i]) + 1):
                best, bk = INF, -1
                for k in range(int(lvl_max[i + 1]) + 1):
                    if not ok[j, k]: continue
                    c = cost[j, k] + J_out[i + 1, k]
                    if c < best: best, bk = c, k
                J_out[i, j] = best; K_out[i, j] = bk
    return dict(J_out=J_out, J_in=J_in, K_out=K_out, K_in=K_in, zone_of=zone_of, sig=sig, dph=dph, dt=dt, e=e, lam=lam,
                expected=float(J_out[0, N_LVL - 1]))


def rollout(sol, offsets):
    """Follow the non-anticipative policy for one realisation of the offsets. Returns the (x, v) trajectory."""
    zone_of, sig = sol["zone_of"], sol["sig"]
    i, j, t = 0, N_LVL - 1, 0.0
    xs, vs = [0.0], [float(LEVELS[j])]
    while i < N_CELL:
        if i in zone_of:
            k_s = zone_of[i]
            p = int(round(((t + offsets[k_s]) % R.CYCLE) / DPH)) % N_PH
            k = int(sol["K_in"][i][j, p])
            if k == D.WAIT:
                t += DPH; continue
        else:
            k = int(sol["K_out"][i, j])
        if k < 0:
            return None
        t += float(sol["dt"][j, k]); i += 1; j = k
        xs.append(i * CELL); vs.append(float(LEVELS[j]))
    return np.array(xs), np.array(vs)


if __name__ == "__main__":
    PACK = [(0.85, 288.15), (0.90, 263.15), (0.95, 263.15)]
    OFFS = [0., 22.5, 45., 67.5]
    NS = len(R.SIGNALS)
    refs = json.load(open("frozen/refs_mini.json" if R.MINI else "frozen/refs.json"))
    lams = (4000., 5000., 6000., 7000., 8000.) if R.MINI else (7000., 8000., 8400., 8700., 9000., 9300., 9600., 10000.)
    out = {}
    print(f"info-matched DP ({'mini' if R.MINI else '20 km'}), SPaT zone {E.SPAT_RANGE:.0f} m = {ZONE_CELLS} cells")
    for soc, T in PACK:
        tabs = D.tables(soc, T); t0 = time.time(); best = None
        for lam in lams:
            sol = solve(lam, soc, T, tabs); rows = []
            for off in OFFS:
                tr = rollout(sol, [off] * NS)
                if tr is None: rows = None; break
                r = D.track(tr[0], tr[1], soc, T, [off] * NS); r["lam"] = lam; rows.append(r)
            if rows is None: continue
            mR = float(np.mean([r["R"] for r in rows]))
            arr = all(r["arrived"] and r["off"] == 0 for r in rows)
            print(f"  soc {soc} T {T-273.15:+.0f}  lam {lam:6.0f}  mean R {mR:8.3f}  t {[round(r['t']) for r in rows]}  stops {[r['stops'] for r in rows]}  ok {arr}", flush=True)
            if arr and (best is None or mR > best[0]): best = (mR, lam, rows)
        mR, lam, rows = best
        for off, r in zip(OFFS, rows):
            k = f"{soc}_{T}_{off}"; out[k] = dict(R=r["R"], t=r["t"], E_Wh=r["E_Wh"], fric_Wh=r["fric_Wh"], reg_Wh=r["reg_Wh"], stops=r["stops"], lam=lam)
            g = refs["grid"][k]
            print(f"    off {off:4.1f}: info-DP R {r['R']:8.3f} t {r['t']:5.0f} stops {r['stops']} | full DP {g['ref']['R']:8.3f} | expert {g['humans']['expert']:8.3f} normal {g['humans']['normal']:8.3f}")
        print(f"  ({time.time()-t0:.0f} s)")
    m = float(np.mean([v["R"] for v in out.values()]))
    print(f"\nMEAN 12-grid: info-matched DP {m:.3f} | full DP {refs['R_dp']:.3f} | expert {refs['R_expert']:.3f} | normal {refs['R_normal']:.3f}")
    print(f"information gap (full - info) = {refs['R_dp'] - m:+.3f}; share of normal->fullDP gap that is information: {100*(refs['R_dp']-m)/(refs['R_dp']-refs['R_normal']):.1f}%")
    refs["R_dp_info"] = m; refs["grid_info"] = out
    json.dump(refs, open("frozen/refs_mini.json" if R.MINI else "frozen/refs.json", "w"), indent=1)
