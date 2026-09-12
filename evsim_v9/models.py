"""Both models, side by side, on the same trajectory.

PAPER_*  implements the manuscript's equations exactly as printed, including the
         errors, so that their consequences can be measured rather than argued.
V1_*     implements the revised formulation of Manuscript_v1.md.

Nothing here is tuned. Every PAPER_ function carries the equation number it
implements and a one-line note on what is wrong with it.
"""
import numpy as np

from plant import Vehicle, Pack, Aux, regen_ceiling_mech, p_bat_max_dc
import effcal

G = 9.80665
DT = 0.1

VEH, PACK, AUX = Vehicle(), Pack(), Aux()
# Replace plant.py's efficiency map with the calibrated loss model. plant.py's
# map read 0.578 at 80 km/h cruise because it applied [S4]'s "50 % at 250 N"
# system-level anchor (measured with the climate load on) as a drivetrain map,
# while models.py added P_aux again separately. See effcal.py. Patched on the
# INSTANCE so regen_ceiling_mech - which calls veh.eta - uses the same map.
VEH.eta = effcal.eta_cal
E_BATT_J = PACK.capacity_j                      # 35.0 kWh, from the sourced pack
ETA_TR = 0.97                                   # single-speed reduction gear
DELTA_R = 0.0                                   # rotating-mass factor; EPA ETW
                                                # already carries an allowance.
                                                # Flagged, not silently assumed.


# ==================================================================== ENERGY
def paper_power(v, a, grade=None, soc=None, T=None):
    """Manuscript Eq. (25):  P_{t+1} = m (a_t + da_t)(v_t + a_t dt) = m a v.

    WRONG: contains no rolling resistance, no aerodynamic drag, no grade, no
    auxiliary load, no drivetrain or machine efficiency, and no regeneration
    ceiling. Integrates to (1/2) m d(v^2), i.e. exactly zero over any trip that
    starts and ends at the same speed, whatever the policy does in between.
    grade/soc/T are accepted and ignored - that is the point.
    """
    return VEH.m * np.asarray(a, float) * np.asarray(v, float)


def v1_power(v, a, grade, soc, T, dispatch=0.0):
    """Manuscript_v1 Eqs. (1)-(5). Returns (P_batt, P_fric, P_wheel).

    P_batt > 0 draws from the pack, P_batt < 0 charges it.
    P_fric is the braking power that exceeds what the pack can accept.
    """
    v = float(max(v, 0.0))
    a = float(a)
    f_res = float(VEH.road_load(v)) + VEH.m * G * float(grade)
    f_w = VEH.m * (1.0 + DELTA_R) * a + f_res
    p_w = f_w * v
    if v < VEH.v_cut and p_w < 0.0:         # cutoff applies to REGEN, never to traction
        return AUX.p_base_dc + dispatch, max(-p_w, 0.0), p_w
    f_mot = VEH.p_mot_max(max(v, 1e-3)) / max(v, 1e-3)
    if p_w >= 0.0:
        eta = float(VEH.eta(v, min(abs(f_w) / f_mot, 1.0)))
        return p_w / (ETA_TR * eta) + AUX.p_base_dc + dispatch, 0.0, p_w
    # braking
    f_av = -f_w                                       # available braking force
    f_ceil = regen_ceiling_mech(VEH, PACK, AUX, v, soc, T, dispatch) / v
    f_reg = min(f_av, f_ceil)
    f_fri = max(f_av - f_ceil, 0.0)
    eta = float(VEH.eta(v, min(f_reg / f_mot, 1.0)))
    p_dc = f_reg * v * ETA_TR * eta                   # into the DC bus
    return -(p_dc - AUX.p_base_dc - dispatch), f_fri * v, p_w


def integrate(traj, model="v1", soc0=0.80, T=298.15):
    """Run an energy ledger over a recorded trajectory.

    traj: list of dicts with v, a, grade (a is the ACHIEVED acceleration).
    Returns gross / regen / net energy in Wh, friction energy in Wh, dSOC, and
    the implied pack capacity E_net/dSOC, which under Eq. (9) must equal the
    true pack capacity.
    """
    soc = soc0
    e_gross = e_reg = e_fric = 0.0
    for o in traj:
        if model == "paper":
            p_b = paper_power(o["v"], o["a"])
            p_f = 0.0
        else:
            p_b, p_f, _ = v1_power(o["v"], o["a"], o["grade"], soc, T)
        e_gross += max(p_b, 0.0) * DT
        e_reg += max(-p_b, 0.0) * DT
        e_fric += p_f * DT
        soc = min(1.0, max(0.0, soc - p_b * DT / E_BATT_J))
    e_net = e_gross - e_reg
    d_soc = soc0 - soc
    return dict(
        E_gross_Wh=e_gross / 3600.0,
        E_reg_Wh=e_reg / 3600.0,
        E_net_Wh=e_net / 3600.0,
        E_fric_Wh=e_fric / 3600.0,
        dSOC=d_soc,
        soc_end=soc,
        implied_kWh=(e_net / 3600.0 / d_soc / 1000.0) if abs(d_soc) > 1e-12 else np.nan,
    )


# ==================================================================== REWARD
def paper_reward(v, a, du, p_batt, band=(0.0, np.inf), A=1e-4,
                 comfort_du=0.6):
    """Manuscript Eqs. (3)-(6).

        r1 = +5 if the consumption rate is inside 'the standard range' else -5
        r2 = -A * P_bat
        r3 = +3 if the comfort specification is met else -3
        r  = r1 + r2 + r3

    UNDEFINED IN THE MANUSCRIPT: the bounds of 'the standard range', and A.
    Both are exposed here as arguments precisely because the manuscript never
    gives them, and the sweep in checks.py shows they decide the optimum.
    """
    r1 = 5.0 if (band[0] <= p_batt <= band[1]) else -5.0
    r2 = -A * p_batt
    r3 = 3.0 if abs(du) <= comfort_du else -3.0
    return r1 + r2 + r3, (r1, r2, r3)


def v1_reward(p_batt, j, done, safety, w=None):
    """Manuscript_v1 Eqs. (16)-(17) on a fixed-DISTANCE episode."""
    w = w or dict(wE=1.0, wT=1.0, wJ=0.2, wC=50.0, wG=200.0,
                  E_ref=2.0e7, T_ref=1800.0, j_max=6.0)
    r = (-w["wE"] / w["E_ref"] * p_batt * DT
         - w["wT"] / w["T_ref"] * DT
         - w["wJ"] / w["j_max"] ** 2 * j ** 2
         - w["wC"] * float(safety))
    if done:
        r += w["wG"]
    return r


# ==================================================================== JERK
def paper_jerk(a_seq):
    """Manuscript Eq. (39): j_t = a_t / dt.  Not jerk - this is 10 * a."""
    return np.asarray(a_seq, float) / DT


def v1_jerk(a_seq):
    """Manuscript_v1 Eq. (15): j_t = (a_{t+1} - a_t) / dt."""
    a = np.asarray(a_seq, float)
    return np.diff(a, prepend=a[0]) / DT


# ================================================================ CONSTRAINT
def paper_bounds(u_prev, delta=0.6, eps=0.2):
    """Manuscript Eqs. (30) + (34) exactly as printed.

        G = [1, -1, 1, -1]^T,  h = [d, d, u_prev+e, -u_prev+e]^T
        lb = max{ -h_i / G_i : G_i < 0 }      <-- the spurious minus sign
        ub = min{  h_i / G_i : G_i > 0 }
    """
    Gm = np.array([1.0, -1.0, 1.0, -1.0])
    h = np.array([delta, delta, u_prev + eps, -u_prev + eps])
    lb = max(-h[i] / Gm[i] for i in range(4) if Gm[i] < 0)
    ub = min(h[i] / Gm[i] for i in range(4) if Gm[i] > 0)
    return lb, ub


def v1_bounds(u_prev, a_now, delta=0.6, eps=0.2, a_lo=-3.5, a_hi=2.6):
    """Manuscript_v1 Eqs. (33)-(36) with the sign fixed and the acceleration
    box added. Returns the HARD interval; the rate limit is applied inside it,
    per Lemma 1, which is what keeps the set non-empty."""
    lo_hard = max(-delta, a_lo - a_now)
    hi_hard = min(delta, a_hi - a_now)
    return lo_hard, hi_hard


def v1_project(u_cand, u_prev, a_now, delta=0.6, eps=0.2, a_lo=-3.5, a_hi=2.6):
    """Eq. (36): rate limit inside, hard safety/actuator box outside."""
    u = np.clip(u_cand, u_prev - eps, u_prev + eps)
    lo, hi = v1_bounds(u_prev, a_now, delta, eps, a_lo, a_hi)
    return float(np.clip(u, lo, hi))
