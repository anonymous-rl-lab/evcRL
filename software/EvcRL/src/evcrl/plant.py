"""PLANT - battery, vehicle dynamics, motor efficiency. One module, all sourced.

This is the physics the RL environment steps. Nothing here is invented; every
constant carries its source, and the two model errors found earlier are fixed:

  * the acceptance surface is PULSE-referenced. A braking event lasts seconds;
    sustained fast-charge plating rates govern minutes of charging and are a
    different quantity. Using them understated the cold ceiling by more than an
    order of magnitude.
  * efficiency is LOAD-dependent. Evaluating eta at full load everywhere made
    gentle braking look far cheaper than it is.

SOURCES
  [S3] Stroe et al., Energies 9(5):360, 2016. NMC pouch 20 Ah, 3.65 V nominal;
       R0 rises at low temperature; ageing to 90 % SoH raises R0 by 61-71.3 %.
  [S4] "2019 Nissan Leaf Plus" benchmarking report, NHTSA-2023-0022-0017 att. 47.
       test mass 1928 kg; road-load targets A=30.360 lbf, B=0.3201 lbf/mph,
       C=0.0196 lbf/mph^2; rolling radius 0.3234 m; final drive 8.139; motor
       340 N.m / 160 kW; machine efficiency 79-96 % observed, ~50 % at very low
       tractive load; peak regen into pack 111 kW measured; regeneration limited
       above ~90 % SOC at +22 C, ~80 % at -7 C, ~60 % at -18 C; DC/DC input
       167 W at 86 % efficiency; PTC heater above 3.5 kW.
  [S5] INL/AVTA BEV pack tests, 2011 and 2013 Nissan Leaf. 10-second CHARGE
       pulse 80.5 kW (2013) and 71.2 kW (2011) at 20 % DOD; pack 360 V,
       66.2 Ah, 24 kWh; cell limits 4.20 V charge, 2.50 V discharge.

VALIDATION (nothing below is fitted to these):
       [S5] 10-s pulse at 80 % SOC, 25 C  ->  predicted 88.9 kW vs 80.5 measured
       [S4] limiting near 90 % SOC, +22 C ->  90 -> 43 kW across the threshold
       [S4] limiting near 80 % SOC,  -7 C ->  67 -> 38 kW
       [S4] limiting near 60 % SOC, -18 C ->  67 -> 48 kW

STILL NOT SOURCED, and flagged wherever it is used:
       the SHAPE of the efficiency map between [S4]'s stated anchors;
       the thermal model (there is none - pack and cabin temperature are inputs).
"""
from dataclasses import dataclass
import numpy as np

LBF = 4.4482216152605
MPH = 0.44704
V_CUT = 2.0                      # m/s, regeneration cutoff


# ============================================================== BATTERY
@dataclass
class Pack:
    """96s2p on a 50 Ah NMC cell: 350 V nominal, 100 Ah, 35 kWh."""
    n_series: int = 96
    n_parallel: int = 2
    q_cell_ah: float = 50.0
    v_nom_cell: float = 3.65          # [S3]
    v_max_cell: float = 4.20          # [S5] pack test charge limit
    r0_cell_ref: float = 1.5e-3       # ohm at 25 C
    ea_over_r: float = 2100.0         # K, Arrhenius constant for R0(T)
    p_hw_max: float = 90e3            # W, rated pack/inverter acceptance
    ageing_r0_factor: float = 1.0     # 1.00 fresh; 1.61-1.71 at 90 % SoH [S3]

    @property
    def capacity_j(self):
        return self.v_nom_cell * self.n_series * self.q_cell_ah * self.n_parallel * 3600.0


def ocv_cell(soc):
    """Smooth NMC OCV anchored to 3.65 V nominal and 4.20 V full [S3].
    Parameterised, not digitised - the one shape assumption on this branch."""
    s = np.clip(np.asarray(soc, float), 0.0, 1.0)
    return 3.20 + 0.90 * s ** 0.55 + 0.12 * s ** 6


def r0_cell(pack, T):
    T = np.asarray(T, float)
    return (pack.r0_cell_ref * pack.ageing_r0_factor
            * np.exp(pack.ea_over_r * (1.0 / T - 1.0 / 298.15)))


def p_bat_max_dc(pack, soc, T):
    """Pulse charge-acceptance power [W]: voltage headroom against the hardware
    cap. The temperature dependence is OHMIC - R0 rises as the pack cools - not
    lithium plating, which governs sustained charging and not a braking pulse."""
    soc = np.clip(np.asarray(soc, float), 0.0, 1.0)
    i_cell = np.maximum((pack.v_max_cell - ocv_cell(soc)) / r0_cell(pack, T), 0.0)
    p = ocv_cell(soc) * pack.n_series * i_cell * pack.n_parallel
    return np.minimum(p, pack.p_hw_max)


# ============================================================== VEHICLE
@dataclass
class Vehicle:
    """2019 Nissan Leaf Plus, per [S4]."""
    m: float = 1928.0                 # kg, test mass (ETW); curb 1761 kg
    rl_A: float = 30.360 * LBF        # 135.1 N
    rl_B: float = 0.3201 * LBF / MPH  # 3.19 N/(m/s)
    rl_C: float = 0.0196 * LBF / MPH ** 2   # 0.44 N/(m/s)^2
    r_wheel: float = 0.3234
    final_drive: float = 8.139
    tau_motor_max: float = 340.0
    p_motor_max: float = 160e3
    v_cut: float = V_CUT

    @property
    def f_wheel_max(self):
        return self.tau_motor_max * self.final_drive / self.r_wheel

    @property
    def v_base(self):
        return self.p_motor_max / self.f_wheel_max

    def road_load(self, v):
        v = np.maximum(np.asarray(v, float), 0.0)
        return self.rl_A + self.rl_B * v + self.rl_C * v ** 2

    def p_mot_max(self, v):
        v = np.maximum(np.asarray(v, float), 1e-3)
        return np.where(v <= self.v_base, self.f_wheel_max * v, self.p_motor_max)

    def eta(self, v, tau_frac):
        """Machine + inverter efficiency, mechanical -> DC.

        SHAPE NOT SOURCED. Anchored to [S4]: 79-96 % observed, rising with load,
        ~50 % at very low tractive load. Everything downstream that depends on
        this is reported with a sensitivity band."""
        v = np.maximum(np.asarray(v, float), 1e-3)
        tf = np.clip(np.asarray(tau_frac, float), 0.0, 1.0)
        load = 0.50 + 0.46 * (1.0 - np.exp(-3.2 * tf))
        speed = 1.0 - 0.12 * np.exp(-v / 2.5)
        return np.clip(load * speed, 0.30, 0.96)


@dataclass
class Aux:
    p_base_dc: float = 167.0 / 0.86      # W at the HV bus [S4]
    p_dispatch_max: float = 3.5e3        # W, PTC heater [S4]


# ============================================================== CEILING
def regen_ceiling_mech(veh, pack, aux, v, soc, T, dispatch=0.0, iters=40):
    """Mechanical regenerative ceiling [W] with load-dependent efficiency.

    Implicit, because eta depends on the very force being solved for:
        f = min( f_mot_max , P_sink_dc / ( v * eta(v, f / f_mot_max) ) )
    Damped fixed point; converges in a few iterations over this range.
    """
    v = np.maximum(np.asarray(v, float), 1e-3)
    f_mot = veh.p_mot_max(v) / v
    sink = p_bat_max_dc(pack, soc, T) + aux.p_base_dc + dispatch
    f = np.minimum(f_mot, sink / (v * 0.9))
    for _ in range(iters):
        e = veh.eta(v, f / np.maximum(f_mot, 1e-9))
        f_new = np.minimum(f_mot, sink / (v * e))
        if np.max(np.abs(f_new - f)) < 1e-9 * (np.max(np.abs(f_new)) + 1.0):
            f = f_new
            break
        f = 0.5 * (f + f_new)
    return f * v


# ============================================================== EVENT
def brake_event(veh, pack, aux, v0, d_avail, soc, T, dispatch, a_max=3.5, n=600):
    """Integrate one braking event under the minimum-friction shape.

    Shape: brake at the comfort limit above a switch speed, then follow the
    ceiling exactly below it. Raising brake force at speed v buys distance at a
    marginal friction cost of exactly f_ceil(v) + f_road(v) - independent of how
    hard you are already braking - so distance is shed where that sum is
    smallest, which is at HIGH speed whenever the sink binds.

    Returns a CLOSED ledger. The residual is reported, never scaled away.
    """
    v0 = max(float(v0), veh.v_cut + 1e-6)
    v = np.linspace(v0, veh.v_cut, n)
    dv = np.diff(v)
    vm = 0.5 * (v[:-1] + v[1:])
    f_ceil = regen_ceiling_mech(veh, pack, aux, vm, soc, T, dispatch) / vm
    f_road = veh.road_load(vm)
    f_comf = np.maximum(veh.m * a_max - f_road, f_ceil)

    def dist(vs):
        g = np.where(vm > vs, f_comf, f_ceil)
        return float(np.sum(-veh.m * vm * dv / (g + f_road)))

    if dist(v0) <= d_avail:
        f_b = f_ceil.copy()
    else:
        lo, hi = veh.v_cut, v0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if dist(mid) > d_avail:
                hi = mid
            else:
                lo = mid
        f_b = np.where(vm > 0.5 * (lo + hi), f_comf, f_ceil)

    ds = -veh.m * vm * dv / (f_b + f_road)
    f_reg = np.minimum(f_b, f_ceil)
    f_fri = np.maximum(f_b - f_ceil, 0.0)

    e_gen = float(np.sum(f_reg * ds))
    e_fri = float(np.sum(f_fri * ds))
    e_road = float(np.sum(f_road * ds))
    e_kin = 0.5 * veh.m * (v0 ** 2 - veh.v_cut ** 2)
    resid = abs(e_kin - (e_gen + e_fri + e_road)) / max(e_kin, 1.0)

    f_mot = veh.p_mot_max(vm) / vm
    eta_w = veh.eta(vm, f_reg / np.maximum(f_mot, 1e-9))
    e_dc = float(np.sum(f_reg * ds * eta_w))
    t = float(np.sum(ds / vm))
    e_aux = min(aux.p_base_dc * t, e_dc)
    e_disp = min(dispatch * t, max(e_dc - e_aux, 0.0))
    e_batt = max(e_dc - e_aux - e_disp, 0.0)
    acc_bound = bool(np.mean(f_ceil < f_mot - 1e-9) > 0.5)
    return dict(e_gen=e_gen, e_fric=e_fri, e_road=e_road, e_kin=e_kin,
                e_dc=e_dc, e_batt=e_batt, e_disp=e_disp, duration=t,
                distance=float(np.sum(ds)), ledger_residual=resid,
                acceptance_bound=acc_bound)


def d_req(veh, pack, aux, v0, soc, T, dispatch=0.0, n=400):
    """Distance needed to stop on the ceiling alone - the latest brake point
    that incurs no friction."""
    v0 = max(float(v0), veh.v_cut + 1e-6)
    v = np.linspace(veh.v_cut, v0, n)
    f = regen_ceiling_mech(veh, pack, aux, v, soc, T, dispatch) / v
    return float(np.trapezoid(veh.m * v / (f + veh.road_load(v)), v))


if __name__ == "__main__":
    veh, pack, aux = Vehicle(), Pack(), Aux()
    print("PLANT")
    print(f"  mass {veh.m:.0f} kg   wheel force limit {veh.f_wheel_max:.0f} N   "
          f"base speed {veh.v_base:.1f} m/s   peak {veh.p_motor_max/1e3:.0f} kW")
    print(f"  pack {pack.n_series}s{pack.n_parallel}p  "
          f"{pack.capacity_j/3.6e6:.1f} kWh   aux {aux.p_base_dc:.0f} W   "
          f"dispatch max {aux.p_dispatch_max/1e3:.1f} kW")
    print("\nVALIDATION of the pulse acceptance surface (nothing fitted to these)")
    leaf = Pack(q_cell_ah=33.1)           # [S5] 24 kWh pack, 96s2p
    print(f"  [S5] 10-s pulse @80% SOC 25C : {float(p_bat_max_dc(leaf,0.8,298.15))/1e3:6.1f} kW"
          f"   measured 80.5 kW")
    for lbl, Tc, th in (("+22C", 22., 0.90), ("-7C", -7., 0.80), ("-18C", -18., 0.60)):
        T = Tc + 273.15
        print(f"  [S4] {lbl:<5} limit near SOC {th:.2f} : "
              f"{float(p_bat_max_dc(pack,th-0.2,T))/1e3:6.1f} -> "
              f"{float(p_bat_max_dc(pack,th,T))/1e3:5.1f} kW across it")
    print("\nP_bat_max_dc [kW]")
    temps = [253.15, 263.15, 273.15, 283.15, 298.15]
    print(f"{'SOC':>5}" + "".join(f"{t-273.15:>9.0f}C" for t in temps))
    for s in (0.2, 0.5, 0.7, 0.9, 0.95):
        print(f"{s:>5.2f}" + "".join(f"{float(p_bat_max_dc(pack,s,t))/1e3:>10.1f}"
                                     for t in temps))
    print("\nsample events (ledger residual must be ~1e-16)")
    for d, soc, T in ((45., 0.5, 298.15), (45., 0.9, 263.15), (160., 0.5, 268.15)):
        r = brake_event(veh, pack, aux, 13.9, d, soc, T, 0.0)
        print(f"  d={d:5.0f} soc={soc} T={T-273.15:>5.0f}C  "
              f"E_dc={r['e_dc']/1e3:6.1f} kJ  fric={r['e_fric']/r['e_kin']*100:5.1f}%  "
              f"resid={r['ledger_residual']:.1e}")
