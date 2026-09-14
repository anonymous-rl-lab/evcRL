"""Verify the retained preparation boundaries, optimized costs, and complete trajectories.

No RL training or uploaded experiment code is used. Requires NumPy and SciPy.
Run: python verify_preparation.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog



def solve(v0, h, j, p, delayed):
    """LP in original nodal accelerations, without the reduced q formula.

    Variables: a1_R, a2_R, a1_G, a2_G. Terminal a3=0 is fixed.
    The endpoint velocity, stop/pass location, three jerk limits and
    velocity-consistent signs define the complete declared maneuver family.
    """
    s, r = v0/h, j*h
    A, b = [], []
    for offset in [0, 2]:
        for coeff in [(1, 0), (-1, 0), (-1, 1), (1, -1), (0, 1), (0, -1)]:
            row = np.zeros(4)
            row[offset:offset+2] = coeff
            A.append(row)
            b.append(r)
    # x_R(3h)<=2v0 h; x_G(3h)>=2v0 h.
    A.extend([[2., 1., 0., 0.], [0., 0., -2., -1.]])
    b.extend([-s, s])
    eq, target = [[1., 1., 0., 0.], [0., 0., 1., 1.]], [-s, 0.]
    if delayed:
        eq.append([1., 0., -1., 0.])
        target.append(0.)
    c = -h*h/v0*np.array([2*p, p, 2*(1-p), 1-p])
    return linprog(c, A_ub=A, b_ub=b, A_eq=eq, b_eq=target,
                   bounds=[(-s, 0), (-s, 0), (-s, 0), (0, s)], method="highs")


def complete_trip(v0, h, j, q, red, L, W, mass=1500., rolling=200., aux=800.):
    s = v0/h
    nodes = [0., -q, q-s if red else q, 0.]
    x, v, a, time, energy = 0., v0, 0., 0., 0.
    vmax, vmin, amax, jmax = v0, v0, 0., 0.
    def step(dt, jerk):
        nonlocal x, v, a, time, energy, vmax, vmin, amax, jmax
        # Inspect the exact interior speed extremum, not just sampled endpoints.
        candidates = [0., dt]
        if abs(jerk)>1e-14:
            zero = -a/jerk
            if 0 < zero < dt:
                candidates.append(zero)
        speeds = [v+a*t+.5*jerk*t*t for t in candidates]
        vmin, vmax = min(vmin, *speeds), max(vmax, *speeds)
        dx = v*dt+.5*a*dt*dt+jerk*dt**3/6
        vnext = v+a*dt+.5*jerk*dt*dt
        anext = a+jerk*dt
        energy += .5*mass*(vnext*vnext-v*v)+rolling*dx+aux*dt
        x, v, a = x+dx, vnext, anext
        time += dt
        amax = max(amax, abs(a), abs(anext))
        jmax = max(jmax, abs(jerk))
    for first, second in zip(nodes[:-1], nodes[1:]):
        step(h, (second-first)/h)
    event_x = x
    if red:
        assert abs(v)<1e-10*max(1,v0) and abs(a)<1e-10
        assert x <= 2*v0*h+1e-9
        step(W, 0.)
        launch = [0., s/2, s/2, 0.]
        for first, second in zip(launch[:-1], launch[1:]):
            step(h, (second-first)/h)
    else:
        assert x >= 2*v0*h-1e-9
    step((L-x)/v0, 0.)
    assert vmin >= -1e-9 and vmax <= v0+1e-9
    assert amax <= s+1e-9 and jmax <= j+1e-9
    assert abs(x-L)<1e-8 and abs(v-v0)<1e-9 and abs(a)<1e-9
    reference = L/v0+(2.5*h+W if red else 0.)+q*h*h/v0
    return {"time":time,"energy":energy,"event_x":event_x,
            "time_error":abs(time-reference),
            "energy_identity_error":abs(energy-(rolling*L+aux*time))}


def vehicle_model():
    ratios = [.4, .4999, .5, .6, 2/3-1e-4, 2/3, .75, .8, .95, 1., 1.2, 2.2]
    rows, errors, trajectory_errors, energy_errors = [], [], [], []
    trajectories = 0
    for v0 in [4., 8., 12., 20.]:
        for h in [2., 4., 6.]:
            for p in [.2, .5, .8]:
                for rho in ratios:
                    j = rho*v0/(h*h)
                    early = solve(v0,h,j,p,False)
                    delayed = solve(v0,h,j,p,True)
                    assert early.success == (rho>=.5)
                    assert delayed.success == (rho>=2/3)
                    record = {"v0":v0,"h":h,"p_red":p,"rho":rho,
                              "early_feasible":early.success,"delayed_feasible":delayed.success}
                    if delayed.success:
                        q = max(v0/h-j*h,0.)
                        gap = delayed.fun-early.fun
                        expected = (1-p)*h*max(1-rho,0.)
                        errors.extend([gap-expected, -delayed.x[0]-q,
                                       -early.x[0]-q, early.x[2]])
                        record.update({"delay_s":gap,"formula_s":expected,"q":q})
                        L, W = 10*v0*h, 2*h
                        er = complete_trip(v0,h,j,q,True,L,W)
                        eg = complete_trip(v0,h,j,0.,False,L,W)
                        dr = complete_trip(v0,h,j,q,True,L,W)
                        dg = complete_trip(v0,h,j,q,False,L,W)
                        trajectories += 4
                        mean_gap = p*(dr["time"]-er["time"])+(1-p)*(dg["time"]-eg["time"])
                        errors.append(mean_gap-expected)
                        # Auxiliary-power energy/time objective, common E0 and lambda.
                        E0, lam = 100000., .08
                        earlyJ = p*(er["energy"]/E0+lam*er["time"])+(1-p)*(eg["energy"]/E0+lam*eg["time"])
                        delayJ = p*(dr["energy"]/E0+lam*dr["time"])+(1-p)*(dg["energy"]/E0+lam*dg["time"])
                        errors.append(delayJ-earlyJ-(800/E0+lam)*expected)
                        for trip in [er,eg,dr,dg]:
                            trajectory_errors.append(trip["time_error"])
                            energy_errors.append(trip["energy_identity_error"])
                    rows.append(record)
    assert max(abs(x) for x in errors) < 1e-8
    assert max(trajectory_errors) < 1e-8
    assert max(energy_errors) < 1e-6
    example=[]
    for name,qR,qG in [("early",.6,0.),("delayed",.6,.6)]:
        R=complete_trip(12.,4.,.6,qR,True,480.,8.)
        G=complete_trip(12.,4.,.6,qG,False,480.,8.)
        example.append({"information":name,"red_time":R["time"],"green_time":G["time"],
                        "mean_time":.5*(R["time"]+G["time"]),"q_red":qR,"q_green":qG})
    return {"cases":len(rows),"lp_solves":2*len(rows),
            "complete_trajectories":trajectories,
            "max_abs_formula_or_optimizer_error":max(abs(x) for x in errors),
            "max_abs_trip_time_error":max(trajectory_errors),
            "max_abs_energy_identity_error_J":max(energy_errors),
            "worked_example":example,"records":rows}


if __name__ == "__main__":
    core=vehicle_model()
    records=core.pop("records")
    root=Path(__file__).parent
    (root/"preparation_formula_records.json").write_text(json.dumps(records,indent=2)+"\n")
    result={"status":"PASS","vehicle_model":core,
            "scope":"Independent optimization and exact integration for the declared analytical executor; no vehicle-RL experiment or unrestricted driving optimum."}
    (root/"preparation_formula_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
