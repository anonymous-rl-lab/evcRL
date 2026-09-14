"""The 20 km arterial environment. Six deceleration events, flat, energy is spent.

WHAT CHANGED FROM THE 2 km DESCENT, AND WHY
  events        1  ->  6   (3 curves, 2 signals, 1 terminal stop)
                A single event collapses the policy to one number; a hand-tuned
                constant beat a 9-input network there. Six independent events
                cannot be expressed by a constant.
  grade         -5 % constant  ->  flat
                Constant grade made three observation channels carry zero
                information, and made the trip a net energy SOURCE, so
                "energy saving" had no denominator. Flat and 20 km long, the
                trip consumes ~2.8 kWh of road load; savings are a percentage
                of something real.
  exogenous     pack state only (ceiling bound on 2.7 % of draws)
                ->  pack state AND signal offset (matters on every episode)

SIGNAL INFORMATION, DELIBERATELY BOUNDED
  The controller sees a signal's phase and its time-to-change only within
  SPAT_RANGE metres, which is what a real SPaT broadcast gives. Outside that
  range it sees distance only. Handing the controller the exact switch time
  from arbitrary distance is the leak that makes a planner look like a learner.

CONSTRAINT LAYER (unchanged in structure from env.py, extended to the route)
  1. jerk limit         |a - a_prev| <= J_MAX * dt
  2. actuator box       A_LO <= a <= A_HI
  3. safety envelope    a <= a_safe(next binding constraint)
     The binding constraint is the nearest downstream of: a curve advisory
     speed, a red signal's stop line, the terminal stop. Guarantees every
     policy is legal, so the learner optimises energy and time only.
"""
import numpy as np

import models as M
import route20 as R

DT = 0.5
A_LO, A_HI = -3.5, 2.6
J_MAX = 2.0
SPAT_RANGE = 1000.0            # m, how far out the signal phase is broadcast. Was 400 m:
                                # at 22 m/s that is 18 s of lookahead, less than the 45 s needed
                                # to reach the next green window, so on 20 % of phases no speed
                                # inside the zone could catch a green and stopping was optimal.
                                # 1000 m is the upper end of a C-V2X roadside deployment.
STOP_MARGIN = 2.0              # m, stop this far before the line
MARGIN = 15.0                  # m, controller margin at a curve entry
PREVIEW = 400.0                # m, how far ahead the speed-limit profile is known
MAX_STEPS = 4000               # (unused cap; episodes end at 1.25 x budget)
BOOTSTRAP_MODE = False         # True: the episode's TIME LIMIT is a pure compute bound, not part
                               # of the task.  No terminal cost-to-go, no deadline in the state,
                               # and the trainer bootstraps through the truncation (D=0), so the
                               # cut-off cannot affect the return.  This removes at one stroke:
                               # the hidden t_end (the Markov violation), the -300 terminal cliff
                               # (which dominated the critic's target range) and the need for a
                               # clock channel.  The Lagrangian time price lambda_T is KEPT -- it
                               # is the smooth, stop-compatible version of "going slow is punished".
CLOCK_OFF = False              # blank the clock channel; set ONCE, identical in training and
                               # evaluation, so the policy never sees a different observation space
TRUNC_FACTOR = 3.0             # training truncation = TRUNC_FACTOR * t_budget (compute bound only)
CUTOFF_VMIN = None             # if set, the episode cut-off is (L - x) / CUTOFF_VMIN instead of
                               # 1.25 * (L - x) / pace.  The proportional rule makes the arrival
                               # criterion SCALE-FREE: a policy must average pace/1.25 = 15.24 m/s
                               # to arrive from ANY start, so exploring starts near the finish are
                               # no easier than a cold start, and a run that settles below 15.24
                               # collects ZERO completed episodes and can never recover.
T_BUDGET = 260.0 if R.MINI else 1050.0   # mini: 4 km at 80 km/h ~ 200 s incl. the
                               # curve, plus room for one red-light wait
CTG_PER_KM = 15.0              # cost-to-go for an UNFINISHED trip, units of
                               # 100 kJ per km left (= 194 Wh/km, above any
                               # driver on this route). Makes every early exit
                               # dearer than driving the rest: the previous
                               # flat -100 / -20 made crashing (-117) and
                               # parking (-116) indistinguishable, and both
                               # cheaper-looking than the objective itself.
FLAT_PEN = 0.0                 # ablation only: flat non-arrival penalty (v1 accounting)
OFFROAD_PEN = 5.0              # the violation itself, on top of cost-to-go
C_TIME = 0.1                   # per second of overrun, units of 100 kJ
K_P = 10.0                     # potential-shaping scale
REWARD_MODE = "L"
OBS_DIM = 13                    # remaining deadline AND elapsed clock (needed by reward P)
CURVE_ENVELOPE = False         # if True the safety envelope also enforces the curve entry
                               # speed, so a curve can never end the episode; arriving too fast
                               # is then paid for in friction (the envelope brakes at A_LO),
                               # not by termination.
REWARD_SCALE = 1.0             # multiplies every step reward (objective unchanged up to scale)
CURVE_MODE = "penalty"         # how the curve entry speed is enforced:
                               #   "envelope"  the safety envelope brakes for the controller
                               #   "penalty"   a continuous cost on the excess speed, no termination
                               #   "terminate" exceeding it ends the episode (the v1 behaviour)
K_CURVE = 0.05                 # per (m/s)^2 of excess per second inside a curve. Entering a
                               # 100 m curve at the legal limit (12.5 m/s over the 35 km/h cap,
                               # ~5 s inside) costs ~39 units against a whole-trip cost of ~168,
                               # so it never pays; unlike a termination it is differentiable, so
                               # the learner is told HOW MUCH too fast it is, not only that it
                               # failed.
CLAMP_LIMIT = False            # ablation only: clamp the legal limit inside the envelope
V_OVER_HARD = 5.0               # m/s above the limit: hard clamp, exploration bound only
K_OVERSPEED = 0.02             # per (m/s)^2 of overspeed per second. At 1 m/s over the
                               # limit this is 0.02/s against a time price of 0.08/s, at 3 m/s
                               # over it is 0.18/s: speeding never pays, but the action stays
                               # live because nothing is clamped.
SHAPE_C = 0.0                  # potential-based shaping on DISTANCE: Phi(s) = -SHAPE_C*(L-x),
                               # so the step reward gains +SHAPE_C*dx. With gamma=1 this telescopes
                               # and cannot change the optimal policy (Ng et al. 1999), but it
                               # removes the part of the value function that is just "how far is
                               # left", which is ~170 of the ~180 units of return. What the critic
                               # then has to fit is only the DEVIATION from a nominal cost per
                               # metre - and the dependence of the value on speed, which decides
                               # the cruise set-point, stops being a 1 % correction on a large
                               # number.
LAM_T = 0.08                   # units of 100 kJ per second = 8 kW, the DP's shadow price of
                               # time. In REWARD_MODE "L" every second costs this, so the
                               # objective is the same Lagrangian the DP of Sec. IV optimises
                               # and slowing down is priced at the margin everywhere, instead
                               # of being free until the deadline and then penalised.

GRID, VLIM = R.profile()


def v_limit(x):
    return float(np.interp(x, GRID, VLIM))


class Route20:
    def __init__(self, soc0=0.85, T=288.15, offsets=None, t_budget=T_BUDGET,
                 rng=None):
        self.soc0, self.T = soc0, T
        rng = rng or np.random.default_rng(0)
        self.offsets = (list(offsets) if offsets is not None
                        else [float(rng.uniform(0, R.CYCLE)) for _ in R.SIGNALS])
        self.t_budget = t_budget
        self.obs_dim = OBS_DIM

    # ------------------------------------------------------------- envelope
    def _stop_targets(self):
        """(position, max speed there) for every downstream constraint."""
        # NOTE. Curves are deliberately NOT in the envelope. When they were,
        # the vehicle physically could not enter one too fast, so meeting the
        # advisory cost nothing and required no decision - the constraint layer
        # was doing the control again. Now exceeding V_CURVE inside a curve
        # ends the episode as an off-road excursion, and the only way to avoid
        # that cheaply is to lift off early; braking late enough to exceed the
        # pack's acceptance ceiling sends the difference to the friction brakes.
        out = []
        for k, xs in enumerate(R.SIGNALS):
            if xs > self.x - 1e-9 and not self._passable(k, xs):
                out.append((xs - STOP_MARGIN, 0.0))
        if CURVE_ENVELOPE or CURVE_MODE == "envelope":
            for ca, cb, vc in R.CURVES:
                if cb > self.x - 1e-9:                  # until the curve EXIT, not its entry:
                    out.append((max(ca - MARGIN, self.x), vc, 2.0))   # inside it the cap still holds
        out.append((R.LENGTH, 0.0))
        return out

    def _passable(self, k, xs):
        """Green RIGHT NOW. Nothing else.

        The first version projected arrival at the current speed and required
        green both now and then. As the vehicle slowed to a crawl at the line
        that projection ran 20 s into the future and the test broke: the
        vehicle crept over the stop line at 2-3 m/s ON RED, after which the
        signal was no longer "ahead" and the constraint vanished, so it went to
        full throttle through a red light. Signals then cost nothing and the
        offset sweep was flat.

        Deciding whether it WILL be green on arrival is the controller's job,
        from the SPaT observation. The envelope's job is only: do not cross on
        red."""
        return R.signal_green(xs, self.t, self.offsets[k])

    def a_safe(self):
        """Largest acceleration that still respects every downstream limit."""
        b = abs(A_LO)
        best = A_HI
        for tg in self._stop_targets():
            xg, vg = tg[0], tg[1]
            bt = tg[2] if len(tg) > 2 else b   # curve targets assume a jerk-feasible 2.0
            d = max(xg - self.x, 0.0)
            # v_next^2 <= vg^2 + 2 b (d - v_next dt)
            disc = (bt * DT) ** 2 + vg ** 2 + 2.0 * bt * d
            u = -bt * DT + np.sqrt(max(disc, 0.0))
            best = min(best, (u - self.v) / DT)
        # The FREE-FLOW legal limit is NOT clamped here any more. Clamping it made
        # the executed acceleration independent of the action whenever the vehicle
        # sat at the limit - 100 % of cruise steps, 98 % of all steps - so the actor
        # saturated at full throttle and had no gradient at all. The limit is now a
        # continuous cost on the state (K_OVERSPEED in step), which keeps the action
        # live everywhere. The envelope still guarantees the stop lines, the curve
        # entry speeds and the terminal stop.
        if CLAMP_LIMIT:                       # ablation: the v1 envelope, which clamped the
            return float(min(best, (R.V_FREE - self.v) / DT))   # legal limit and killed the action
        # A hard clamp remains 5 m/s ABOVE the legal limit. It bounds the state space
        # and the reward during exploration; the operating region (0 to V_FREE) is
        # untouched, so the action stays live where the policy actually lives.
        best = min(best, (R.V_FREE + V_OVER_HARD - self.v) / DT)
        return float(best)

    # ---------------------------------------------------------------- reset
    def reset(self):
        self.x = 0.0
        self.v = R.V_FREE
        self.a = 0.0
        self.soc = float(self.soc0)
        self.t = 0.0
        self.e_batt = self.e_fric = self.e_reg = 0.0
        self.steps = 0
        self.n_stop = 0
        self.n_violation = 0
        self.n_offroad = 0
        self._was_moving = True
        self.t_over = 0.0
        self.e_curve_pen = 0.0
        self.curve_excess = 0.0
        self.v_over_max = 0.0
        self.jerk_sq = 0.0
        self.jerk_max = 0.0
        self.a_min = 0.0
        self.a_max = 0.0
        self.jerk_overrides = 0
        self.safety_infeasible = 0
        self.e_overspeed_pen = 0.0
        self.terminal_pen = 0.0
        self.t_end = (TRUNC_FACTOR * self.t_budget if BOOTSTRAP_MODE else
                      (R.LENGTH / CUTOFF_VMIN if CUTOFF_VMIN else 1.25 * self.t_budget))  # cut-off; reset_from() rescales it for a
        return self.obs()                      # training episode that starts mid-route

    def reset_from(self, x0, v0, t0):
        """Exploring start. The cut-off is made proportional to the distance this
        episode actually has to cover, so a start at 15 km is not held open for the
        full 1312 s of a 20 km trip: 95 % of transitions came from timed-out episodes
        and the learner saw very few episodes per million steps."""
        self.reset()  # reset the ledger as well as position, even on a reused instance
        self.x, self.v, self.t = float(x0), float(v0), float(t0)
        self._was_moving = self.v >= 0.1
        pace = R.LENGTH / self.t_budget
        d = max(R.LENGTH - self.x, 1.0)
        self.t_end = self.t + (TRUNC_FACTOR * d / pace if BOOTSTRAP_MODE else
                               (d / CUTOFF_VMIN if CUTOFF_VMIN else 1.25 * d / pace))
        return self.obs()

    # ------------------------------------------------------------------ obs
    def obs(self):
        d_end = max(R.LENGTH - self.x, 0.0)
        # nearest curve ahead within preview
        dc, vc = PREVIEW, R.V_FREE
        for a, b, vv in R.CURVES:
            if a >= self.x and a - self.x < dc:
                dc, vc = a - self.x, vv
        # nearest signal ahead, with SPaT only inside range
        ds, phase, ttc = SPAT_RANGE, -1.0, 1.0
        for k, xs in enumerate(R.SIGNALS):
            if xs >= self.x and xs - self.x < ds:
                ds = xs - self.x
                if ds <= SPAT_RANGE:
                    phase = 1.0 if R.signal_green(xs, self.t, self.offsets[k]) else 0.0
                    ttc = min(R.time_to_change(xs, self.t, self.offsets[k]) / R.CYCLE, 1.0)
        return np.array([
            self.v / R.V_FREE,
            self.a / 3.5,
            d_end / R.LENGTH,
            v_limit(self.x) / R.V_FREE,
            min(dc, PREVIEW) / PREVIEW,
            vc / R.V_FREE,
            min(ds, SPAT_RANGE) / SPAT_RANGE,
            phase,
            ttc,
            (self.soc - 0.5) / 0.5,
            (self.T - 283.15) / 30.0,
            # MARKOV FIX.  This channel used to be (t_budget - t)/t_budget with a FIXED
            # t_budget, but the episode actually terminates at self.t_end, which for an
            # exploring start is t0 + 1.25 (L - x0)/pace and therefore depends on the
            # episode's hidden start.  Two episodes with byte-identical observations had
            # deadlines differing by up to hundreds of seconds (measured range 927-1644 s
            # against evaluation's fixed 1312 s, 57 % of training episodes off by >60 s),
            # so the same state had different true values and the critic could not fit a
            # consistent value function.  Reporting the REAL remaining budget restores the
            # Markov property in the observation.
            (0.0 if CLOCK_OFF else (self.t_end - self.t) / self.t_budget),
            self.t / self.t_budget,  # reward P also depends on the absolute overtime boundary
        ], dtype=np.float32)

    # ------------------------------------------------------------------ phi
    def phi(self):
        sched = R.LENGTH * min(self.t / self.t_budget, 1.0)
        return -K_P * max(sched - self.x, 0.0) / R.LENGTH

    # ----------------------------------------------------------------- step
    def project(self, a_cmd):
        a1 = float(np.clip(a_cmd, self.a - J_MAX * DT, self.a + J_MAX * DT))
        a2 = float(np.clip(a1, A_LO, A_HI))
        a3 = min(a2, self.a_safe())
        # An infeasible stop must never create unlimited braking. Safety may
        # override comfort (recorded in step), but not the actuator box.
        return float(np.clip(a3, max(A_LO, -self.v / DT), A_HI))

    def step(self, a_cmd):
        x_prev_shape = self.x
        phi_old = self.phi() if REWARD_MODE == "P" else 0.0
        a_prev = self.a
        safe = self.a_safe()
        self.safety_infeasible += int(safe < max(A_LO, -self.v / DT) - 1e-7)
        a = self.project(float(a_cmd))
        jerk = (a - a_prev) / DT
        self.jerk_sq += jerk * jerk
        self.jerk_max = max(self.jerk_max, abs(jerk))
        self.jerk_overrides += int(abs(jerk) > J_MAX + 1e-7)
        self.a_min = min(self.a_min, a)
        self.a_max = max(self.a_max, a)
        v0 = self.v
        v1 = max(0.0, v0 + a * DT)
        vm = 0.5 * (v0 + v1)
        p_b, p_f, _ = M.v1_power(vm, a, 0.0, self.soc, self.T)
        self.e_batt += p_b * DT
        self.e_fric += p_f * DT
        self.e_reg += max(-p_b, 0.0) * DT
        self.soc -= p_b * DT / M.E_BATT_J
        x_new = self.x + vm * DT
        # hard backstop: never cross a stop line while it is not passable.
        # With the margin above this should fire ~never; n_violation is
        # reported so that "never" is measured, not assumed.
        red_crossing = False
        for k, xs in enumerate(R.SIGNALS):
            if self.x < xs <= x_new:
                # Signal phase at the actual crossing, not at the start of the step.
                distance = xs - self.x
                vcross = np.sqrt(max(v0 * v0 + 2.0 * a * distance, 0.0))
                crossing_dt = 2.0 * distance / max(v0 + vcross, 1e-12)
                if not R.signal_green(xs, self.t + crossing_dt, self.offsets[k]):
                    red_crossing = True
                    self.n_violation += 1
        # Keep the physical position, velocity and energy ledger. A violation
        # terminates with a cost; it does not teleport the vehicle behind a line.
        self.x = x_new
        self.v, self.a = v1, a
        self.t += DT
        self.steps += 1
        if self._was_moving and v1 < 0.1 and self.x < R.LENGTH - 10.0:
            self.n_stop += 1                  # stops en route; the terminal stop is not one
        self._was_moving = v1 >= 0.1

        offroad = False
        self.curve_excess = 0.0
        for a, b, vc in R.CURVES:
            if a - 1e-9 <= self.x <= b + 1e-9 and self.v > vc + 1e-6:
                offroad = True
                self.n_offroad += 1
                self.curve_excess = self.v - vc
                self.e_curve_pen += K_CURVE * self.curve_excess ** 2 * DT
                break
        arrived = self.x >= R.LENGTH - 1.0 and v1 <= 0.3
        overtime = self.t > self.t_budget
        done = arrived or red_crossing or (offroad and CURVE_MODE == "terminate") or self.t >= self.t_end

        r = -p_b * DT / 1e5
        over = max(self.v - R.V_FREE, 0.0)
        if over > 0.0:
            penalty = K_OVERSPEED * min(over, V_OVER_HARD) ** 2 * DT
            r -= penalty
            self.e_overspeed_pen += penalty
            self.t_over += DT
            self.v_over_max = max(self.v_over_max, self.v)
        if REWARD_MODE == "L":
            r -= LAM_T * DT
            if SHAPE_C:
                r += SHAPE_C * (min(self.x, R.LENGTH) - min(x_prev_shape, R.LENGTH))
                if done:
                    # Phi(absorbing) = 0 (Ng et al. 1999). Without this payout the total
                    # shaping over an episode is c*x_T, which is POLICY-DEPENDENT and adds
                    # a real progress incentive instead of redistributing value in time.
                    # With it the total is exactly c*L for every episode, so the optimal
                    # policy is provably unchanged and the target spread is unchanged.
                    r += SHAPE_C * max(R.LENGTH - self.x, 0.0)
        else:
            if overtime:
                r -= C_TIME * DT
            if REWARD_MODE == "P":
                r += (0.0 if (done and arrived) else self.phi()) - phi_old
        if done and not arrived and not BOOTSTRAP_MODE:
            self.terminal_pen = CTG_PER_KM * max(R.LENGTH - self.x, 0.0) / 1000.0 + FLAT_PEN
            r -= self.terminal_pen
        if red_crossing:
            r -= OFFROAD_PEN
            self.terminal_pen += OFFROAD_PEN
        if offroad:
            if CURVE_MODE == "penalty":
                r -= K_CURVE * self.curve_excess ** 2 * DT
            elif CURVE_MODE == "terminate":
                r -= OFFROAD_PEN
        r *= REWARD_SCALE
        return self.obs(), r, done, dict(arrived=arrived, overtime=overtime,
                                         offroad=offroad, red_crossing=red_crossing,
                                         terminated=arrived or red_crossing or (offroad and CURVE_MODE == "terminate"),
                                         deadline=not arrived and self.t >= self.t_end)

    # -------------------------------------------------------------- rollout
    def rollout(self, pi):
        o = self.reset(); Rr = 0.0
        while True:
            o, r, d, info = self.step(pi(self, o)); Rr += r
            if d:
                break
        return dict(R=Rr, t=self.t, E_Wh=self.e_batt / 3600.0,
                    fric_Wh=self.e_fric / 3600.0, reg_Wh=self.e_reg / 3600.0,
                    Wh_km=self.e_batt / 3600.0 / max(self.x / 1000.0, 1e-12),
                    stops=self.n_stop, viol=self.n_violation, off=self.n_offroad,
                    arrived=info["arrived"], x=self.x,
                    jerk_rms=float(np.sqrt(self.jerk_sq / max(self.steps, 1))),
                    jerk_max=self.jerk_max, jerk_overrides=self.jerk_overrides,
                    safety_infeasible=self.safety_infeasible, a_min=self.a_min,
                    a_max=self.a_max, overspeed_pen=self.e_overspeed_pen,
                    terminal_pen=self.terminal_pen)


# ------------------------------------------------------------- baselines
def pi_legal(env, o):
    """Drive as fast as the law and the envelope allow. This is the
    'no eco-driving' reference: it brakes as late as it legally can."""
    return A_HI


def make_pi_gentle(a_b):
    """Approach every constraint - curves included, since the envelope no
    longer covers them - at a fixed deceleration a_b. One parameter: the same
    policy family that beat the network on the descent, carried over so the
    identical test applies here."""
    def pi(env, o):
        v_t = R.V_FREE
        # stop lines and the finish: no margin, or the vehicle parks short of
        # the line and never arrives (it did: every policy stalled 15 m from
        # the finish and timed out at the step cap).
        for tg in env._stop_targets():
            xg, vg = tg[0], tg[1]
            d = max(xg - env.x, 0.0)
            v_t = min(v_t, np.sqrt(max(vg ** 2 + 2.0 * a_b * d, 0.0)))
        # curves: margin, because with 0.5 s steps at ~10 m/s the vehicle
        # advances 5 m per step and aiming to reach the limit exactly AT the
        # entry lands it inside the curve still too fast - an off-road, not a
        # rounding error.
        for a, b, vc in R.CURVES:
            if b > env.x - 1e-9:
                d = max(a - MARGIN - env.x, 0.0)
                v_t = min(v_t, np.sqrt(max(vc ** 2 + 2.0 * a_b * d, 0.0)))
        return (v_t - env.v) / DT
    return pi


def make_pi_human(preview, a_brake, a_accel=1.5, v_cruise=None):
    """A driver with a SIGHT DISTANCE. Sees a curve, a red light or the finish
    only when within `preview` metres; until then holds the speed limit. Brakes
    at a_brake once something is seen. No SPaT, no map beyond sight, no
    knowledge of the pack state.

        distracted   preview 100 m, a_brake 3.0   (brakes hard, late)
        normal       preview 200 m, a_brake 1.5
        expert       preview 500 m, a_brake 0.5   (~ the old 'gentle' policy)

    The safety envelope still applies underneath - the car will not leave the
    road or run a red - but with 100 m of sight at 80 km/h the envelope has to
    do real work, and the driver pays for it in friction."""
    v_free = R.V_FREE if v_cruise is None else float(v_cruise)   # a paced driver cruises below the limit
    def pi(env, o):
        v_t = v_free
        for tg in env._stop_targets():              # red lights, finish
            xg, vg = tg[0], tg[1]
            d = xg - env.x
            if 0.0 <= d <= preview:
                v_t = min(v_t, np.sqrt(max(vg ** 2 + 2.0 * a_brake * max(d, 0.0), 0.0)))
        for a, b, vc in R.CURVES:
            d = a - env.x
            if -1e-9 <= d <= preview or (a <= env.x <= b + 1e-9):
                dm = max(a - MARGIN - env.x, 0.0)      # be at speed BEFORE the entry
                v_t = min(v_t, np.sqrt(max(vc ** 2 + 2.0 * a_brake * dm, 0.0)))
        return float(np.clip((v_t - env.v) / DT, -a_brake, a_accel))
    return pi


HUMANS = {"distracted": (140.0, 3.0, 2.0),
          "normal": (160.0, 2.0, 1.5),
          "attentive": (160.0, 1.4, 1.0)}
# SIGHT IS CAPPED AT 160 m for every driver. A 500 m sight distance is not a driver,
# it is a map: on a peri-urban arterial a curve is visible at most ~160 m ahead. With
# that cap the three drivers differ only in how hard they brake once they see, which
# is the realistic axis: the gentlest braking that still makes 80 -> 35 km/h inside
# 160 m (15 m entry margin, 2 m/s^3 jerk ramp) is about 1.4 m/s^2, so "attentive" is
# at the physical edge, "normal" brakes at 2.0 and "distracted" at 3.0 and pays for it
# in friction whenever the pack cannot accept the power.


if __name__ == "__main__":
    print(f"20 km arterial, budget {T_BUDGET:.0f} s, dt {DT} s, "
          f"SPaT range {SPAT_RANGE:.0f} m")
    print(f"{'policy':<20}{'R':>9}{'trip s':>9}{'Wh':>9}{'Wh/km':>8}"
          f"{'regen':>8}{'fric':>8}{'stops':>7}{'off':>5}{'arr':>5}")
    offs = [10.0, 10.0]
    for name, pi in [("legal (no eco)", pi_legal)] + \
                    [(f"gentle a_b={ab}", make_pi_gentle(ab))
                     for ab in (0.3, 0.5, 0.8, 1.2, 2.0, 3.0)]:
        e = Route20(0.85, 288.15, offsets=offs)
        r = e.rollout(pi)
        print(f"{name:<20}{r['R']:>9.3f}{r['t']:>9.1f}{r['E_Wh']:>9.1f}"
              f"{r['Wh_km']:>8.1f}{r['reg_Wh']:>8.1f}{r['fric_Wh']:>8.1f}"
              f"{r['stops']:>7}{r['off']:>5}{str(r['arrived'])[0]:>5}")

    print("\nsame policy, pack state swept (a_b = 0.8):")
    print(f"{'SOC/T':<14}{'R':>9}{'Wh':>9}{'Wh/km':>8}{'regen':>8}{'fric':>8}{'%fric':>8}")
    for soc, Tc in ((0.60, 25.0), (0.85, 25.0), (0.85, -10.0), (0.95, -10.0)):
        e = Route20(soc, Tc + 273.15, offsets=offs)
        r = e.rollout(make_pi_gentle(0.8))
        tot = r["reg_Wh"] + r["fric_Wh"]
        print(f"{soc}/{Tc:+.0f}C{'':<5}{r['R']:>9.3f}{r['E_Wh']:>9.1f}{r['Wh_km']:>8.1f}"
              f"{r['reg_Wh']:>8.1f}{r['fric_Wh']:>8.1f}"
              f"{100*r['fric_Wh']/max(tot,1e-9):>8.1f}")

    print("\nsame policy, signal offset swept (SOC 0.85 / +15 C, a_b = 0.8):")
    print(f"{'offset s':<14}{'R':>9}{'trip s':>9}{'Wh':>9}{'stops':>7}{'viol':>6}")
    for off in (0., 15., 30., 45., 60., 75.):
        e = Route20(0.85, 288.15, offsets=[off, off])
        r = e.rollout(make_pi_gentle(0.8))
        print(f"{off:<14.0f}{r['R']:>9.3f}{r['t']:>9.1f}{r['E_Wh']:>9.1f}{r['stops']:>7}{r['viol']:>6}")
