"""Instance-based form of the archived release-aware execution routines."""
import math
class Executor:

    def __init__(self, jerk=2.0, dt=0.5):
        self.j = float(jerk)
        self.dt = float(dt)
        self.q = self.j * self.dt
        self.lo = -3.5
        self.hi = 2.6
        if self.j not in (2.0, 3.0, 4.0) or self.dt != 0.5:
            raise ValueError('Validated jerk limits are 2/3/4 at dt=0.5')

    def reserve(self, v, a):
        """Exact discrete speed margin for releasing negative acceleration."""
        b = max(-a, 0.0)
        return v - self.dt * sum((max(b - i * self.q, 0.0) for i in range(1, 5)))

    def brake_bound(self, v):
        """Largest b with DT*sum_{i>=0} max(b-i*J*DT,0) <= v."""
        v = max(v, 0.0)
        for n in range(1, 5):
            b = (v / self.dt + self.q * n * (n - 1) / 2) / n
            if b <= n * self.q + 1e-12:
                return min(b, -self.lo)
        return -self.lo

    def backup_distance(self, v, a, target_v=0.0):
        """Distance of a specific feasible brake-and-release backup; not a DP oracle.

    A velocity cap above zero tolerates temporary undershoot during release.
    Zero-speed infeasibility returns infinity; no unlimited braking is used.
    """
        if self.reserve(v, a) < -1e-08:
            return math.inf
        d = 0.0
        for k in range(100):
            if v <= target_v + 1e-09 and (a <= 0 if target_v > 0 else abs(a) < 1e-09):
                return d
            if v <= target_v + 1e-09:
                candidate = min(0.0, a + self.q) if a < 0 else max(0.0, a - self.q)
            else:
                candidate = max(self.lo, a - self.q, -self.brake_bound(v - target_v))
                if candidate > a + self.q + 1e-08:
                    candidate = max(self.lo, a - self.q, -self.brake_bound(v))
            if candidate > a + self.q + 1e-07:
                return math.inf
            vn = v + self.dt * candidate
            if vn < -1e-08:
                return math.inf
            vn = max(vn, 0.0)
            d += (v + vn) * self.dt / 2
            v, a = (vn, candidate)
        return math.inf

    def interval(self, v, a, targets, vmax=None):
        """Return a verified nonempty interval or an explicit infeasibility marker.

    targets is (remaining distance, maximum terminal speed). Unexpected red
    transitions may make this interval empty. Caller retains emergency policy.
    """
        lower = max(self.lo, a - self.q, -self.brake_bound(v))
        upper = min(self.hi, a + self.q)
        if vmax is not None:
            upper = min(upper, self.brake_bound(max(vmax - v, 0.0)))
        if lower > upper + 1e-09:
            return (None, dict(reason='velocity_release_infeasible'))

        def fits(acc, d, vg, tolerance=0.0):
            vn = max(v + self.dt * acc, 0.0)
            travel = (v + vn) * self.dt / 2
            if vg > 0:
                rise = self.dt * sum((max(acc - i * self.q, 0.0) for i in range(1, 4)))
                if vn + rise <= vg + 1e-09:
                    return True
            bdist = self.backup_distance(vn, acc, vg)
            return travel + bdist <= d + tolerance
        for d, vg in targets:
            d = max(d, 0.0)
            if not fits(lower, d, vg, 1e-07):
                return (None, dict(reason='stop_distance_infeasible', distance=d, target_v=vg))
            if not fits(lower, d, vg):
                upper = lower
                continue
            if fits(upper, d, vg):
                continue
            low, high = (lower, upper)
            for _ in range(32):
                mid = (low + high) / 2
                if fits(mid, d, vg):
                    low = mid
                else:
                    high = mid
            upper = low
        if lower > upper + 1e-08:
            return (None, dict(reason='empty_intersection'))
        return ((lower, upper), dict(reason='feasible', lower=lower, upper=upper))

    def project(self, v, a, cmd, targets, fallback, vmax=None):
        bounds, info = self.interval(v, a, targets, vmax)
        if bounds is None:
            info['fallback'] = True
            return (fallback, info)
        low, high = bounds
        result = min(max(cmd, low), high)
        info['fallback'] = False
        return (result, info)

    def forward_distance(self, v, a, duration, vmax):
        """A reachable fastest-forward backup under the discrete actuator limits."""
        d = 0.0
        while duration > 1e-12:
            acc = min(self.hi, a + self.q, self.brake_bound(max(vmax - v, 0.0)))
            span = min(self.dt, duration)
            vn = v + acc * span
            d += (v + vn) * span / 2
            v, a = (vn, acc)
            duration -= span
        return d

    def signal_project(self, v, a, cmd, bounds, distance, green_remaining, vmax):
        """Project onto a stop interval OR a clear-before-red interval.

    This anticipates the known end of green. Picking merely 'green now' can
    strand the vehicle outside its stop set when the light switches.
    """
        lower, upper = bounds

        def stop_ok(acc):
            vn = max(v + self.dt * acc, 0.0)
            travel = (v + vn) * self.dt / 2
            return travel + self.backup_distance(vn, acc) <= max(distance - 2.0, 0.0)

        def go_ok(acc):
            span = min(self.dt, green_remaining)
            travel = v * span + 0.5 * acc * span * span
            if travel >= distance + 1e-07:
                return True
            if green_remaining <= self.dt:
                return False
            vn = max(v + self.dt * acc, 0.0)
            first = (v + vn) * self.dt / 2
            return first + self.forward_distance(vn, acc, green_remaining - self.dt, vmax) >= distance + 1e-07
        intervals = []
        if stop_ok(lower):
            lo, hi = (lower, upper)
            if not stop_ok(hi):
                for _ in range(32):
                    mid = (lo + hi) / 2
                    if stop_ok(mid):
                        lo = mid
                    else:
                        hi = mid
                hi = lo
            intervals.append((lower, hi, 'stop'))
        if go_ok(upper):
            lo, hi = (lower, upper)
            if not go_ok(lo):
                for _ in range(32):
                    mid = (lo + hi) / 2
                    if go_ok(mid):
                        hi = mid
                    else:
                        lo = mid
                lo = hi
            intervals.append((lo, upper, 'clear'))
        if not intervals:
            return (None, dict(signal_reason='no_stop_or_clear_backup'))
        choices = [(min(max(cmd, lo), hi), mode) for lo, hi, mode in intervals]
        action, mode = min(choices, key=lambda p: abs(p[0] - cmd))
        return (action, dict(signal_reason=mode, green_remaining=green_remaining, signal_distance=distance))
