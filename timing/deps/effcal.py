"""Motor+inverter efficiency: physical loss model, calibrated against [S4].

WHAT WAS WRONG. plant.py's map was eta = f(torque fraction) anchored on [S4]'s
"~50 % at very low tractive load". [S4] states that anchor precisely:

    "At low tractive loads (approximately 250 N), the motor efficiency is
     approximately 50 percent (measured at 0% grade)"
    "at high tractive load, approaches 96 percent (measured at 6% grade)"

250 N on the flat is ~13 m/s (47 km/h), i.e. 3.25 kW of mechanical power. For
that to be 50 % efficient the losses must be ~3.2 kW, which no motor of this
class dissipates at 3 kW output. That figure is a SYSTEM number measured with
the climate load on ([S4] lists a PTC heater up to 3.5 kW). The old map baked
that overhead into the drivetrain map AND models.py added P_aux again on top -
the same watts counted twice. At 80 km/h cruise it gave eta = 0.58 and
205 Wh/km, about 30 % above what this car actually uses.

THE REPLACEMENT. Standard separated-loss model, losses in watts:

    P_loss(F, v) = P0 + a*v + b*F^2          (standby+ , iron ~ speed,
                                              copper ~ torque^2)
    eta_traction = P_mech / (P_mech + P_loss)
    eta_regen    = (P_mech - P_loss) / P_mech

Calibrated on two points:
  (1) high tractive load -> 96 %      [S4]
  (2) highway cruise 80 km/h -> 90 %  engineering value for a PMSM+inverter at
                                      this operating point; NOT from [S4], and
                                      flagged as the one fitted number.
P0 = 300 W standby (inverter + pumps), also not from [S4].

The [S4] 250 N / 50 % point is then REPRODUCED, not discarded: it comes back
when ~2.5 kW of climate load is added, which is the reading that makes it
consistent with (1).
"""
import numpy as np
import plant as PL

VEH = PL.Vehicle()
P0 = 300.0
F_HI, V_HI, ETA_HI = 4000.0, 15.0, 0.96
F_CR, V_CR, ETA_CR = 421.0, 22.22, 0.90


def _fit():
    L1 = F_HI * V_HI * (1.0 / ETA_HI - 1.0) - P0
    L2 = F_CR * V_CR * (1.0 / ETA_CR - 1.0) - P0
    A = np.array([[V_HI, F_HI ** 2], [V_CR, F_CR ** 2]])
    a, b = np.linalg.solve(A, np.array([L1, L2]))
    return float(a), float(b)


A_FE, B_CU = _fit()


def p_loss(F, v):
    return P0 + A_FE * np.maximum(v, 0.0) + B_CU * np.asarray(F, float) ** 2


def eta_cal(v, tau_frac):
    """Drop-in for Vehicle.eta: same signature (speed, torque fraction)."""
    v = np.maximum(np.asarray(v, float), 1e-3)
    tf = np.clip(np.asarray(tau_frac, float), 0.0, 1.0)
    f_mot = VEH.p_mot_max(v) / v
    F = tf * f_mot
    pm = F * v
    return np.clip(pm / (pm + p_loss(F, v)), 0.05, 0.97)


if __name__ == "__main__":
    print(f"fitted loss model:  P_loss = {P0:.0f} + {A_FE:.2f}*v + {B_CU:.3e}*F^2  [W]")
    print(f"\ncheck against the two calibration anchors:")
    for F, v, tgt in ((F_HI, V_HI, ETA_HI), (F_CR, V_CR, ETA_CR)):
        e = F * v / (F * v + p_loss(F, v))
        print(f"  F={F:5.0f} N v={v:5.1f} m/s -> eta {e:.4f}  (target {tgt})")
    print(f"\n[S4]'s 250 N / 50 % point, and what climate load reproduces it:")
    F, v = 250.0, 13.0
    pm = F * v
    print(f"  F=250 N, v=13.0 m/s ({v*3.6:.0f} km/h): P_mech {pm/1e3:.2f} kW, "
          f"drivetrain loss {p_loss(F,v):.0f} W -> drivetrain eta {pm/(pm+p_loss(F,v)):.3f}")
    need = pm / 0.50 - pm - p_loss(F, v)
    print(f"  to read 50 % at the DC bus you need {need:.0f} W of extra load "
          f"-> that is the climate system ([S4] PTC up to 3500 W)")
    print(f"\nefficiency map, traction:")
    print(f"  {'v km/h':>8}" + "".join(f"{f:>8.0f}N" for f in (250, 421, 1000, 2000, 4000, 6000)))
    for vk in (20, 40, 60, 80, 100):
        v = vk / 3.6
        row = "".join(f"{float(eta_cal(v, min(f/(VEH.p_mot_max(v)/v),1.0))):>9.3f}"
                      for f in (250, 421, 1000, 2000, 4000, 6000))
        print(f"  {vk:>8}" + row)
    print(f"\n  old map at 80 km/h cruise (F=421): "
          f"{float(VEH.eta(22.22, 421/(VEH.p_mot_max(22.22)/22.22))):.3f}")
    print(f"  new map at 80 km/h cruise (F=421): {float(eta_cal(22.22, 421/(VEH.p_mot_max(22.22)/22.22))):.3f}")
    # steady cruise consumption, the number that has to be plausible
    print(f"\nsteady-cruise consumption (flat, base aux 194 W, no climate):")
    print(f"  {'km/h':>6}{'F road N':>10}{'eta':>7}{'P_elec kW':>11}{'Wh/km':>8}")
    aux = PL.Aux().p_base_dc
    for vk in (40, 60, 80, 100, 110):
        v = vk / 3.6
        F = float(VEH.road_load(v)) / 0.97
        e = float(eta_cal(v, min(F / (VEH.p_mot_max(v) / v), 1.0)))
        pe = F * v / e + aux
        print(f"  {vk:>6}{F*0.97:>10.0f}{e:>7.3f}{pe/1e3:>11.2f}{pe/v/3.6:>8.1f}")
