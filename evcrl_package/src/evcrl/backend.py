"""Archived numerical backend with parameters and route bound per instance."""
import numpy as np
from types import SimpleNamespace
from . import models as M
DT = 0.5
A_LO, A_HI = (-3.5, 2.6)
J_MAX = 2.0
SPAT_RANGE = 1000.0
STOP_MARGIN = 2.0
MARGIN = 15.0
PREVIEW = 400.0
MAX_STEPS = 4000
BOOTSTRAP_MODE = False
CLOCK_OFF = False
TRUNC_FACTOR = 3.0
CUTOFF_VMIN = None
T_BUDGET = 1050.0
CTG_PER_KM = 15.0
FLAT_PEN = 0.0
OFFROAD_PEN = 5.0
C_TIME = 0.1
K_P = 10.0
REWARD_MODE = 'L'
OBS_DIM = 13
CURVE_ENVELOPE = False
REWARD_SCALE = 1.0
CURVE_MODE = 'penalty'
K_CURVE = 0.05
CLAMP_LIMIT = False
V_OVER_HARD = 5.0
K_OVERSPEED = 0.02
SHAPE_C = 0.0
LAM_T = 0.08
HUMANS = {'distracted': (140.0, 3.0, 2.0), 'normal': (160.0, 2.0, 1.5), 'attentive': (160.0, 1.4, 1.0)}

def parameters(**overrides):
    p=SimpleNamespace(**{k:v for k,v in globals().items() if k in ['DT', 'A_LO', 'A_HI', 'J_MAX', 'SPAT_RANGE', 'STOP_MARGIN', 'MARGIN', 'PREVIEW', 'MAX_STEPS', 'BOOTSTRAP_MODE', 'CLOCK_OFF', 'TRUNC_FACTOR', 'CUTOFF_VMIN', 'T_BUDGET', 'CTG_PER_KM', 'FLAT_PEN', 'OFFROAD_PEN', 'C_TIME', 'K_P', 'REWARD_MODE', 'OBS_DIM', 'CURVE_ENVELOPE', 'REWARD_SCALE', 'CURVE_MODE', 'K_CURVE', 'CLAMP_LIMIT', 'V_OVER_HARD', 'K_OVERSPEED', 'SHAPE_C', 'LAM_T', 'HUMANS']})
    for k,v in overrides.items():
        if not hasattr(p,k): raise ValueError(k)
        setattr(p,k,v)
    return p

class Backend:

    def __init__(self, soc0=0.85, T=288.15, offsets=None, t_budget=T_BUDGET, rng=None, *, route, cfg):
        self.route = route
        self.cfg = cfg
        self.soc0, self.T = (soc0, T)
        rng = rng or np.random.default_rng(0)
        self.offsets = list(offsets) if offsets is not None else [float(rng.uniform(0, self.route.CYCLE)) for _ in self.route.SIGNALS]
        self.t_budget = t_budget
        self.obs_dim = self.cfg.OBS_DIM

    def _stop_targets(self):
        """(position, max speed there) for every downstream constraint."""
        out = []
        for k, xs in enumerate(self.route.SIGNALS):
            if xs > self.x - 1e-09 and (not self._passable(k, xs)):
                out.append((xs - self.cfg.STOP_MARGIN, 0.0))
        if self.cfg.CURVE_ENVELOPE or self.cfg.CURVE_MODE == 'envelope':
            for ca, cb, vc in self.route.CURVES:
                if cb > self.x - 1e-09:
                    out.append((max(ca - self.cfg.MARGIN, self.x), vc, 2.0))
        out.append((self.route.LENGTH, 0.0))
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
        return self.route.signal_green(xs, self.t, self.offsets[k])

    def a_safe(self):
        """Largest acceleration that still respects every downstream limit."""
        b = abs(self.cfg.A_LO)
        best = self.cfg.A_HI
        for tg in self._stop_targets():
            xg, vg = (tg[0], tg[1])
            bt = tg[2] if len(tg) > 2 else b
            d = max(xg - self.x, 0.0)
            disc = (bt * self.cfg.DT) ** 2 + vg ** 2 + 2.0 * bt * d
            u = -bt * self.cfg.DT + np.sqrt(max(disc, 0.0))
            best = min(best, (u - self.v) / self.cfg.DT)
        if self.cfg.CLAMP_LIMIT:
            return float(min(best, (self.route.V_FREE - self.v) / self.cfg.DT))
        best = min(best, (self.route.V_FREE + self.cfg.V_OVER_HARD - self.v) / self.cfg.DT)
        return float(best)

    def reset(self):
        self.x = 0.0
        self.v = self.route.V_FREE
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
        self.t_end = self.cfg.TRUNC_FACTOR * self.t_budget if self.cfg.BOOTSTRAP_MODE else self.route.LENGTH / self.cfg.CUTOFF_VMIN if self.cfg.CUTOFF_VMIN else 1.25 * self.t_budget
        return self.obs()

    def reset_from(self, x0, v0, t0):
        """Exploring start. The cut-off is made proportional to the distance this
        episode actually has to cover, so a start at 15 km is not held open for the
        full 1312 s of a 20 km trip: 95 % of transitions came from timed-out episodes
        and the learner saw very few episodes per million steps."""
        self.reset()
        self.x, self.v, self.t = (float(x0), float(v0), float(t0))
        self._was_moving = self.v >= 0.1
        pace = self.route.LENGTH / self.t_budget
        d = max(self.route.LENGTH - self.x, 1.0)
        self.t_end = self.t + (self.cfg.TRUNC_FACTOR * d / pace if self.cfg.BOOTSTRAP_MODE else d / self.cfg.CUTOFF_VMIN if self.cfg.CUTOFF_VMIN else 1.25 * d / pace)
        return self.obs()

    def obs(self):
        d_end = max(self.route.LENGTH - self.x, 0.0)
        dc, vc = (self.cfg.PREVIEW, self.route.V_FREE)
        for a, b, vv in self.route.CURVES:
            if a >= self.x and a - self.x < dc:
                dc, vc = (a - self.x, vv)
        ds, phase, ttc = (self.cfg.SPAT_RANGE, -1.0, 1.0)
        for k, xs in enumerate(self.route.SIGNALS):
            if xs >= self.x and xs - self.x < ds:
                ds = xs - self.x
                if ds <= self.cfg.SPAT_RANGE:
                    phase = 1.0 if self.route.signal_green(xs, self.t, self.offsets[k]) else 0.0
                    ttc = min(self.route.time_to_change(xs, self.t, self.offsets[k]) / self.route.CYCLE, 1.0)
        return np.array([self.v / self.route.V_FREE, self.a / 3.5, d_end / self.route.LENGTH, self.route.profile_limit(self.x) / self.route.V_FREE, min(dc, self.cfg.PREVIEW) / self.cfg.PREVIEW, vc / self.route.V_FREE, min(ds, self.cfg.SPAT_RANGE) / self.cfg.SPAT_RANGE, phase, ttc, (self.soc - 0.5) / 0.5, (self.T - 283.15) / 30.0, 0.0 if self.cfg.CLOCK_OFF else (self.t_end - self.t) / self.t_budget, self.t / self.t_budget], dtype=np.float32)

    def phi(self):
        sched = self.route.LENGTH * min(self.t / self.t_budget, 1.0)
        return -self.cfg.K_P * max(sched - self.x, 0.0) / self.route.LENGTH

    def project(self, a_cmd):
        a1 = float(np.clip(a_cmd, self.a - self.cfg.J_MAX * self.cfg.DT, self.a + self.cfg.J_MAX * self.cfg.DT))
        a2 = float(np.clip(a1, self.cfg.A_LO, self.cfg.A_HI))
        a3 = min(a2, self.a_safe())
        return float(np.clip(a3, max(self.cfg.A_LO, -self.v / self.cfg.DT), self.cfg.A_HI))

    def step(self, a_cmd):
        x_prev_shape = self.x
        phi_old = self.phi() if self.cfg.REWARD_MODE == 'P' else 0.0
        a_prev = self.a
        safe = self.a_safe()
        self.safety_infeasible += int(safe < max(self.cfg.A_LO, -self.v / self.cfg.DT) - 1e-07)
        a = self.project(float(a_cmd))
        jerk = (a - a_prev) / self.cfg.DT
        self.jerk_sq += jerk * jerk
        self.jerk_max = max(self.jerk_max, abs(jerk))
        self.jerk_overrides += int(abs(jerk) > self.cfg.J_MAX + 1e-07)
        self.a_min = min(self.a_min, a)
        self.a_max = max(self.a_max, a)
        v0 = self.v
        v1 = max(0.0, v0 + a * self.cfg.DT)
        vm = 0.5 * (v0 + v1)
        p_b, p_f, _ = M.v1_power(vm, a, 0.0, self.soc, self.T)
        self.e_batt += p_b * self.cfg.DT
        self.e_fric += p_f * self.cfg.DT
        self.e_reg += max(-p_b, 0.0) * self.cfg.DT
        self.soc -= p_b * self.cfg.DT / M.E_BATT_J
        x_new = self.x + vm * self.cfg.DT
        red_crossing = False
        for k, xs in enumerate(self.route.SIGNALS):
            if self.x < xs <= x_new:
                distance = xs - self.x
                vcross = np.sqrt(max(v0 * v0 + 2.0 * a * distance, 0.0))
                crossing_dt = 2.0 * distance / max(v0 + vcross, 1e-12)
                if not self.route.signal_green(xs, self.t + crossing_dt, self.offsets[k]):
                    red_crossing = True
                    self.n_violation += 1
        self.x = x_new
        self.v, self.a = (v1, a)
        self.t += self.cfg.DT
        self.steps += 1
        if self._was_moving and v1 < 0.1 and (self.x < self.route.LENGTH - 10.0):
            self.n_stop += 1
        self._was_moving = v1 >= 0.1
        offroad = False
        self.curve_excess = 0.0
        for a, b, vc in self.route.CURVES:
            if a - 1e-09 <= self.x <= b + 1e-09 and self.v > vc + 1e-06:
                offroad = True
                self.n_offroad += 1
                self.curve_excess = self.v - vc
                self.e_curve_pen += self.cfg.K_CURVE * self.curve_excess ** 2 * self.cfg.DT
                break
        arrived = self.x >= self.route.LENGTH - 1.0 and v1 <= 0.3
        overtime = self.t > self.t_budget
        done = arrived or red_crossing or (offroad and self.cfg.CURVE_MODE == 'terminate') or (self.t >= self.t_end)
        r = -p_b * self.cfg.DT / 100000.0
        over = max(self.v - self.route.V_FREE, 0.0)
        if over > 0.0:
            penalty = self.cfg.K_OVERSPEED * min(over, self.cfg.V_OVER_HARD) ** 2 * self.cfg.DT
            r -= penalty
            self.e_overspeed_pen += penalty
            self.t_over += self.cfg.DT
            self.v_over_max = max(self.v_over_max, self.v)
        if self.cfg.REWARD_MODE == 'L':
            r -= self.cfg.LAM_T * self.cfg.DT
            if self.cfg.SHAPE_C:
                r += self.cfg.SHAPE_C * (min(self.x, self.route.LENGTH) - min(x_prev_shape, self.route.LENGTH))
                if done:
                    r += self.cfg.SHAPE_C * max(self.route.LENGTH - self.x, 0.0)
        else:
            if overtime:
                r -= self.cfg.C_TIME * self.cfg.DT
            if self.cfg.REWARD_MODE == 'P':
                r += (0.0 if done and arrived else self.phi()) - phi_old
        if done and (not arrived) and (not self.cfg.BOOTSTRAP_MODE):
            self.terminal_pen = self.cfg.CTG_PER_KM * max(self.route.LENGTH - self.x, 0.0) / 1000.0 + self.cfg.FLAT_PEN
            r -= self.terminal_pen
        if red_crossing:
            r -= self.cfg.OFFROAD_PEN
            self.terminal_pen += self.cfg.OFFROAD_PEN
        if offroad:
            if self.cfg.CURVE_MODE == 'penalty':
                r -= self.cfg.K_CURVE * self.curve_excess ** 2 * self.cfg.DT
            elif self.cfg.CURVE_MODE == 'terminate':
                r -= self.cfg.OFFROAD_PEN
        r *= self.cfg.REWARD_SCALE
        return (self.obs(), r, done, dict(arrived=arrived, overtime=overtime, offroad=offroad, red_crossing=red_crossing, terminated=arrived or red_crossing or (offroad and self.cfg.CURVE_MODE == 'terminate'), deadline=not arrived and self.t >= self.t_end))

    def rollout(self, pi):
        o = self.reset()
        Rr = 0.0
        while True:
            o, r, d, info = self.step(pi(self, o))
            Rr += r
            if d:
                break
        return dict(R=Rr, t=self.t, E_Wh=self.e_batt / 3600.0, fric_Wh=self.e_fric / 3600.0, reg_Wh=self.e_reg / 3600.0, Wh_km=self.e_batt / 3600.0 / max(self.x / 1000.0, 1e-12), stops=self.n_stop, viol=self.n_violation, off=self.n_offroad, arrived=info['arrived'], x=self.x, jerk_rms=float(np.sqrt(self.jerk_sq / max(self.steps, 1))), jerk_max=self.jerk_max, jerk_overrides=self.jerk_overrides, safety_infeasible=self.safety_infeasible, a_min=self.a_min, a_max=self.a_max, overspeed_pen=self.e_overspeed_pen, terminal_pen=self.terminal_pen)
