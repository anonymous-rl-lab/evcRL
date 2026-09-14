"""The 20 km peri-urban arterial: 3 sharp curves, 2 signals, flat.

WHY THIS ROUTE REPLACES THE 2 km DESCENT
  The descent had ONE deceleration event, so the whole policy collapsed to a
  single number and a hand-tuned constant beat a 9-input network. It also had
  constant grade (three observation channels carried no information) and it
  RECOVERED energy net, so "energy saving" had no sensible denominator.

  This route has SIX deceleration events spread over 20 km, it is flat so the
  trip genuinely consumes energy (~117 Wh/km of road load at 80 km/h), and the
  signal phase is an exogenous variable that matters on EVERY episode - unlike
  the pack state, whose acceptance ceiling bound on only 2.7 % of draws.

GEOMETRY (per the specification: 20 km, 80 km/h, 3 sharp curves, 2 signals,
no grade)
      0 -  3460 m   80 km/h
   3460 -  3540 m   CURVE 1, advisory 30 km/h  (R ~ 35 m at 2.0 m/s2 lateral)
   3540 -  7000 m   80 km/h
           7000 m   SIGNAL 1
   7000 - 10960 m   80 km/h
  10960 - 11040 m   CURVE 2, advisory 35 km/h
  11040 - 15000 m   80 km/h
          15000 m   SIGNAL 2
  15000 - 17960 m   80 km/h
  17960 - 18040 m   CURVE 3, advisory 30 km/h
  18040 - 20000 m   80 km/h, stop at the end

WHAT SUMO IS USED FOR, AND WHAT IT IS NOT
  SUMO builds and validates the network (netconvert), and SUMO's own car
  following drives the reference trip. The exported profile is nothing but
  v_limit(x) and the signal positions and cycles - a boundary condition, not a
  traffic model. There is no self-written car-following, lane-changing or
  signal-coordination logic anywhere. Background vehicles, when wanted, are
  replayed from SUMO traces rather than simulated by us.

  Running SUMO inside the RL loop is not affordable (TraCI is one IPC round
  trip per 0.1 s step, and training needs 10^6+ steps), so the profile is
  exported once and stepped by a fast longitudinal integrator.
"""
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCN = os.path.join(HERE, "scn")

import os
MINI = os.environ.get("EVSIM_ROUTE", "") == "mini"
LENGTH = 4000.0 if MINI else 20000.0
V_FREE = 80 / 3.6                      # 22.22 m/s
V_CURVE = 35 / 3.6                     # 9.72 m/s - HARD limit at every curve.
                                       # Above it the vehicle leaves the road:
                                       # the episode fails. It is NOT enforced
                                       # by the safety envelope, so satisfying
                                       # it is the controller's job and braking
                                       # late enough to need friction costs
                                       # energy. That is the decision the curve
                                       # is here to create.
# Positions are multiples of 50 m so that a 50 m DP cell puts every curve edge
# and every signal on a cell boundary. That matters: at 20 m cells the
# free-flow crossing time is 0.9 s, which does not divide the 0.25 s DP time
# bin and rounds up to 1.0 s - an 11 % time inflation per cell that made the
# DP's "fastest trip" slower than a hand policy's actual trip. At 50 m the
# crossing time is 2.25 s = 9 bins exactly.
CURVES = ([(1450.0, 1550.0, V_CURVE)] if MINI else
          [(3450.0, 3550.0, V_CURVE),   # (x0, x1, hard speed limit)
           (10950.0, 11050.0, V_CURVE),
           (17950.0, 18050.0, V_CURVE)])
SIGNALS = [3000.0] if MINI else [7000.0, 15000.0]
NET = "arterial_mini" if MINI else "arterial"
CYCLE = 90.0                           # s, fixed-time signal cycle
GREEN = 30.0                           # s of green per cycle. Was 45 (50 % duty). At 50 %
                                       # a driver who ignores the signal is stopped on only half
                                       # the offsets and one avoided stop is worth 3 units of
                                       # return, the same size as the pack-state spread, so the
                                       # signal channel carried no learnable gradient. 33 % duty
                                       # is the low end of a real arterial and makes the decision
                                       # worth learning.
YELLOW = 4.0                           # s, counted as not-passable

# node boundaries: every speed change and every signal
def _boundaries():
    xs = {0.0, LENGTH}
    for a, b, _ in CURVES:
        xs.update((a, b))
    xs.update(SIGNALS)
    return sorted(xs)


def v_limit_at(x):
    for a, b, v in CURVES:
        if a - 1e-9 <= x <= b + 1e-9:
            return v
    return V_FREE


def build_net():
    """Write nod/edg XML and run netconvert. Flat: every z is 0."""
    xs = _boundaries()
    nodes = ["<nodes>"]
    for i, x in enumerate(xs):
        t = "traffic_light" if x in SIGNALS else "priority"
        nodes.append(f'  <node id="n{i}" x="{x:.1f}" y="0.0" z="0.0" type="{t}"/>')
    nodes.append("</nodes>")
    edges = ["<edges>"]
    for i in range(len(xs) - 1):
        mid = 0.5 * (xs[i] + xs[i + 1])
        edges.append(f'  <edge id="e{i}" from="n{i}" to="n{i+1}" numLanes="1" '
                     f'speed="{v_limit_at(mid):.3f}" priority="1"/>')
    edges.append("</edges>")
    os.makedirs(SCN, exist_ok=True)
    open(os.path.join(SCN, NET+".nod.xml"), "w").write("\n".join(nodes))
    open(os.path.join(SCN, NET+".edg.xml"), "w").write("\n".join(edges))
    r = subprocess.run(["netconvert",
                        "-n", os.path.join(SCN, NET+".nod.xml"),
                        "-e", os.path.join(SCN, NET+".edg.xml"),
                        "--tls.cycle.time", str(int(CYCLE)),
                        "--tls.yellow.time", str(int(YELLOW)),
                        "-o", os.path.join(SCN, NET+".net.xml")],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


DX = 5.0                                # m, resolution of the exported profile


def profile(rebuild=False):
    """v_limit on a 5 m grid, read back from the SUMO network. If sumolib is
    not installed (training box), the profile exported from the built network
    is loaded from scn/<NET>_profile.npy instead - same numbers."""
    cache = os.path.join(SCN, NET + "_profile.npy")
    net_file = os.path.join(SCN, NET + ".net.xml")
    if os.path.exists(cache) and not rebuild:
        g, v = np.load(cache)
        return g, v
    try:
        import sumolib
        assert os.path.exists(net_file)
    except (ImportError, AssertionError):
        g, v = np.load(cache)
        return g, v
    net = sumolib.net.readNet(os.path.join(SCN, NET+".net.xml"))
    xs = _boundaries()
    grid = np.arange(0.0, LENGTH + DX, DX)
    v = np.full(grid.shape, V_FREE)
    for i in range(len(xs) - 1):
        e = net.getEdge(f"e{i}")
        m = (grid >= xs[i]) & (grid < xs[i + 1])
        v[m] = e.getSpeed()             # taken from the built network, not from us
    v[-1] = 0.0                         # stop at the far end
    np.save(cache, np.array([grid, v]))
    return grid, v


def signal_green(x_sig, t, offset):
    """Is the signal at x_sig passable at time t? Fixed-time cycle, per-episode
    offset. Yellow counts as not passable."""
    phase = (t + offset) % CYCLE
    return phase < GREEN


def time_to_change(x_sig, t, offset):
    """Seconds until the current phase ends. This is what a SPaT broadcast
    gives, and it is only exposed to the controller inside SPAT_RANGE."""
    phase = (t + offset) % CYCLE
    return (GREEN - phase) if phase < GREEN else (CYCLE - phase)


if __name__ == "__main__":
    code, log = build_net()
    print("netconvert:", "OK" if code == 0 else "FAILED")
    if code != 0:
        print(log[-2000:])
        raise SystemExit(1)
    g, v = profile(rebuild=True)
    print(f"  length {g[-1]:.0f} m, grid {DX:.0f} m, {len(g)} points")
    print(f"  free-flow {V_FREE*3.6:.0f} km/h; "
          f"curve advisories {[round(c[2]*3.6) for c in CURVES]} km/h")
    print(f"  signals at {[int(s) for s in SIGNALS]} m, cycle {CYCLE:.0f} s, "
          f"green {GREEN:.0f} s ({100*GREEN/CYCLE:.0f} %)")
    # netconvert stores speeds to 3 decimals, so the built value is the
    # authoritative one and the tolerance is set accordingly.
    for a, b, vv in CURVES:
        i = int((0.5 * (a + b)) / DX)
        assert abs(v[i] - vv) < 5e-3, f"curve at {a}: profile {v[i]} vs {vv}"
    print("  profile matches the built network at every curve "
          "(built values are authoritative; netconvert rounds to 1 mm/s)")

    # what the events cost, before any control decision
    import models as M
    print("\n  energy scale of this route (why the effects are visible here):")
    f_road = float(M.VEH.road_load(V_FREE))
    e_road = f_road * LENGTH / 0.85 / 3600.0
    print(f"    road load at {V_FREE*3.6:.0f} km/h: {f_road:.0f} N "
          f"-> {e_road:.0f} Wh over 20 km ({e_road/20:.0f} Wh/km)")
    for k, (a, b, vv) in enumerate(CURVES, 1):
        ke = 0.5 * M.VEH.m * (V_FREE ** 2 - vv ** 2)
        print(f"    curve {k}: shed {ke/1e3:.0f} kJ = {ke/3600:.0f} Wh of kinetic "
              f"energy; all-regen vs all-friction differs by ~{ke*0.82/3600:.0f} Wh")
    ke_stop = 0.5 * M.VEH.m * V_FREE ** 2
    print(f"    a full stop at a signal: {ke_stop/1e3:.0f} kJ = {ke_stop/3600:.0f} Wh "
          f"of kinetic energy, plus the same again to get back up to speed")
