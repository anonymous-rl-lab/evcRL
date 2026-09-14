"""Where does the 20 km trip's energy actually go, event by event?

Splits every trip into curve windows, signal windows and free cruise, and
reports battery energy, friction-brake energy and time in each, for the TD3
controller, the full-information DP, the information-matched DP and the three
drivers. This is what says whether the learner loses at the curves, at the
signals, or on the open road."""
import json, sys
import numpy as np
import torch
import env20 as E, route20 as R, dp20L as D, dp20I as I, td3_run as T

E.REWARD_MODE = "L"; E.CURVE_ENVELOPE = True
LAM = E.LAM_T; NS = len(R.SIGNALS)
refs = json.load(open("frozen/refs_v3.json"))
PACK = [(0.85, 288.15), (0.90, 263.15), (0.95, 263.15)]
OFFS = [0., 22.5, 45., 67.5]
CURVE_W = [(a - 600.0, b + 300.0) for a, b, _ in R.CURVES]
SIG_W = [(x - E.SPAT_RANGE, x + 300.0) for x in R.SIGNALS]


def label(x):
    for i, (a, b) in enumerate(CURVE_W):
        if a <= x < b: return "curve"
    for i, (a, b) in enumerate(SIG_W):
        if a <= x < b: return "signal"
    return "cruise"


def trace(env, pi):
    o = env.reset()
    acc = {k: dict(E=0.0, F=0.0, t=0.0, d=0.0, vs=[]) for k in ("curve", "signal", "cruise")}
    while True:
        eb, ef, x0, v0 = env.e_batt, env.e_fric, env.x, env.v
        o, r, d, i = env.step(pi(env, o))
        k = label(0.5 * (x0 + env.x))
        acc[k]["E"] += env.e_batt - eb; acc[k]["F"] += env.e_fric - ef
        acc[k]["t"] += E.DT; acc[k]["d"] += env.x - x0; acc[k]["vs"].append(0.5 * (v0 + env.v))
        if d: break
    for k in acc:
        acc[k]["E"] /= 3600.0; acc[k]["F"] /= 3600.0
        acc[k]["v"] = float(np.mean(acc[k]["vs"])) if acc[k]["vs"] else 0.0
        acc[k].pop("vs")
    return acc, dict(t=env.t, E=env.e_batt / 3600.0, F=env.e_fric / 3600.0, stops=env.n_stop, arr=i["arrived"])


def dp_pi(xs, vs):
    def pi(e, o):
        v_t = float(np.interp(e.x, xs, vs))
        if v_t < 0.5 and e.x < R.LENGTH - 5.0: v_t = 0.5
        for ca, cb, cvc in R.CURVES:
            if cb > e.x - 1e-9:
                dd = max(ca - E.MARGIN - e.x, 0.0)
                v_t = min(v_t, np.sqrt(max(cvc ** 2 + 2.0 * D.A_DEC * dd, 0.0)))
        return (v_t - e.v) / E.DT
    return pi


def td3_pi(actor):
    def pi(e, o):
        with torch.no_grad():
            return T.act(float(actor(torch.as_tensor(o).unsqueeze(0))[0, 0]))
    return pi


ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "out/ckpt_N0_s0_2500k.pt")
actor = T.MLP(12, 1, True); actor.load_state_dict(ck["actor"]); actor.eval()

pols = {"TD3": None, "full DP": None, "info DP": None}
pols.update({n: None for n in E.HUMANS})
agg = {k: {s: dict(E=0., F=0., t=0., v=0.) for s in ("curve", "signal", "cruise")} for k in pols}
tot = {k: dict(t=0., E=0., F=0., stops=0., arr=0.) for k in pols}
warm_cold = {k: {"warm": dict(E=0., F=0.), "cold": dict(E=0., F=0.)} for k in pols}
n = 0
for soc, Tk in PACK:
    tabs = D.tables(soc, Tk)
    isol = I.solve(refs["grid"][f"{soc}_{Tk}_0.0"]["info"]["lam"], soc, Tk, tabs)
    for off in OFFS:
        k = f"{soc}_{Tk}_{off}"; n += 1
        s = D.solve(refs["grid"][k]["ref"]["lam"], soc, Tk, [off] * NS, tabs)
        xs, vs = I.rollout(isol, [off] * NS)
        runs = {"TD3": td3_pi(actor), "full DP": dp_pi(s["xs"], s["vs"]), "info DP": dp_pi(xs, vs)}
        runs.update({nm: E.make_pi_human(*p) for nm, p in E.HUMANS.items()})
        for nm, pi in runs.items():
            a, t = trace(E.Route20(soc, Tk, offsets=[off] * NS), pi)
            for sg in a:
                for f in ("E", "F", "t", "v"): agg[nm][sg][f] += a[sg][f]
            for f in ("t", "E", "F", "stops"): tot[nm][f] += t[f]
            tot[nm]["arr"] += float(t["arr"])
            w = "warm" if Tk > 280 else "cold"
            warm_cold[nm][w]["E"] += a["curve"]["E"]; warm_cold[nm][w]["F"] += a["curve"]["F"]

print(f"20 km, mean over {n} conditions. Windows: curve = entry-600 m to exit+300 m (3 of them);"
      f" signal = stop line-1000 m to +300 m (2); cruise = the remaining {R.LENGTH/1000 - 3*0.9 - 2*1.3:.1f} km.\n")
print(f"{'policy':11s} | {'R_L':>8} {'trip s':>7} {'E Wh':>7} {'fric':>6} {'stops':>5} | "
      f"{'CURVES E':>9}{'fric':>6}{'v m/s':>6} | {'SIGNALS E':>10}{'fric':>6}{'v m/s':>6} | {'CRUISE E':>9}{'v m/s':>6}")
for nm in pols:
    T_, Etot, Ftot = tot[nm]["t"] / n, tot[nm]["E"] / n, tot[nm]["F"] / n
    RL = -Etot * 3600 / 1e5 - LAM * T_
    c, sg, cr = (agg[nm][x] for x in ("curve", "signal", "cruise"))
    print(f"{nm:11s} | {RL:8.2f} {T_:7.0f} {Etot:7.0f} {Ftot:6.0f} {tot[nm]['stops']/n:5.2f} | "
          f"{c['E']/n:9.0f}{c['F']/n:6.0f}{c['v']/n:6.1f} | {sg['E']/n:10.0f}{sg['F']/n:6.0f}{sg['v']/n:6.1f} | "
          f"{cr['E']/n:9.0f}{cr['v']/n:6.1f}")
print(f"\ncurve windows only, warm pack (4 cond) vs cold pack (8 cond): battery Wh / friction Wh")
for nm in pols:
    w, c = warm_cold[nm]["warm"], warm_cold[nm]["cold"]
    print(f"  {nm:11s} warm {w['E']/4:6.0f} / {w['F']/4:5.0f}    cold {c['E']/8:6.0f} / {c['F']/8:5.0f}")
