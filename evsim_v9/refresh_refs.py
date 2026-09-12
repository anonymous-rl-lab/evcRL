"""Recompute references on the current plant, with the archived parameters.

No parameter search and no selection on the reporting split. DP is a tracked
policy generator, not a certified optimum. Both halves use their own baselines.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import env20 as E
import route20 as R

HERE = Path(__file__).resolve().parent
PACK = [(0.85, 288.15), (0.90, 263.15), (0.95, 263.15)]
OFFS = [0., 22.5, 45., 67.5]


def env_fingerprint():
    h = hashlib.sha256()
    for name in ("env20.py", "models.py", "plant.py", "effcal.py", "route20.py",
                 f"scn/{R.NET}_profile.npy"):
        h.update(name.encode()); h.update((HERE / name).read_bytes())
    return h.hexdigest()


def main():
    import dp20L as D
    import dp20I as I
    E.REWARD_MODE = "L"; E.CURVE_MODE = "envelope"; E.SHAPE_C = 0.
    suffix = "_mini" if R.MINI else ""
    old = json.loads((HERE / "frozen" / f"refs{suffix}_v3.json").read_text())
    out = dict(schema=2, environment_sha256=env_fingerprint(),
               protocol="Fixed archived DP lambdas and driver parameters; current physical rollout rewards. Validation offsets 0/45; reporting offsets 22.5/67.5.",
               LAM_T=E.LAM_T, grid={}, route_length=R.LENGTH)
    t0 = time.monotonic()
    for soc, T in PACK:
        tabs = D.tables(soc, T)
        info_lam = old["grid"][f"{soc}_{T}_0.0"]["info"]["lam"]
        info = I.solve(info_lam, soc, T, tabs)
        for off in OFFS:
            key = f"{soc}_{T}_{off}"
            offsets = [off] * len(R.SIGNALS)
            lam = old["grid"][key]["ref"]["lam"]
            full = D.solve(lam, soc, T, offsets, tabs)
            if full is None: raise RuntimeError(f"Full DP infeasible: {key}")
            ref = D.track(full["xs"], full["vs"], soc, T, offsets); ref["lam"] = lam
            trajectory = I.rollout(info, offsets)
            if trajectory is None: raise RuntimeError(f"Info DP infeasible: {key}")
            inf = D.track(*trajectory, soc, T, offsets); inf["lam"] = info_lam
            humans, paced = {}, {}
            for name, params in E.HUMANS.items():
                humans[name] = E.Route20(soc, T, offsets).rollout(E.make_pi_human(*params))
                vk = old[f"v_{name}_paced_kmh"]
                paced[name] = E.Route20(soc, T, offsets).rollout(E.make_pi_human(*params, v_cruise=vk/3.6))
            out["grid"][key] = dict(ref=ref, info=inf, humans=humans, humans_paced=paced)
            print(f"{key}: DP={ref['R']:.3f} info={inf['R']:.3f} normal={humans['normal']['R']:.3f} [{time.monotonic()-t0:.1f}s]", flush=True)
    vals = list(out["grid"].values())
    out["R_dp"] = float(np.mean([x["ref"]["R"] for x in vals]))
    out["R_dp_info"] = float(np.mean([x["info"]["R"] for x in vals]))
    for name in E.HUMANS:
        out[f"R_{name}"] = float(np.mean([x["humans"][name]["R"] for x in vals]))
        out[f"R_{name}_paced"] = float(np.mean([x["humans_paced"][name]["R"] for x in vals]))
    path = HERE / "frozen" / f"refs{suffix}_fixed.json"
    path.write_text(json.dumps(out, indent=2, default=float, allow_nan=False))
    print(f"Saved {path.name}; all arrivals={all(y['arrived'] for x in vals for y in [x['ref'], x['info'], *x['humans'].values()])}", flush=True)


if __name__ == "__main__":
    main()
