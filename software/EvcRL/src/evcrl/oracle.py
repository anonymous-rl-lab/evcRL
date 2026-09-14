"""Truth-based detector for tests, debugging and executor studies.

The base camera environment has no detector: without one the executor sees no
road events. ``OracleDetector`` reproduces the archived ``source='oracle'``
probe: it reads the renderer's truth distances of the objects that are drawn
in the current frame and returns them in the detector output format. It is an
explicit, logged truth channel (``info['detector_kind'] == 'oracle'``) meant to
exercise memory + executor without pretrained weights; it is NOT a perception
model and results obtained with it must not be reported as camera results.
"""
import copy
import numpy as np

COLOR_CLASSES = ("red", "yellow", "green", "off", "unknown")
DETECTION_CLASSES = ("traffic_light", "curve_sign", "end_marker", "release_sign")


class OracleDetector:
    requires_truth = True
    kind = "oracle"

    def __init__(self, seed=0, dist_noise_rel=0.0, dist_noise_abs=0.0, dropout=0.0, min_range_m=1.5):
        """dist_noise_rel/abs: Gaussian distance noise (relative, absolute metres); dropout: probability that a
        visible object is reported with score 0 in a frame; min_range_m: objects closer than this are not reported."""
        self.seed = int(seed); self.dist_noise_rel = float(dist_noise_rel); self.dist_noise_abs = float(dist_noise_abs)
        self.dropout = float(dropout); self.min_range_m = float(min_range_m)
        self.last = None; self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def _noisy(self, d):
        if d is None: return 0.0
        d = float(d)
        if self.dist_noise_rel or self.dist_noise_abs:
            d += self.rng.normal(0.0, self.dist_noise_rel * d + self.dist_noise_abs)
        return max(d, 0.0)

    def __call__(self, rgb, truth):
        def seen(d): return d is not None and d > self.min_range_m
        raw = {"traffic_light": (int(truth.get("visible", 0)) == 1, truth.get("light_m")),
               "curve_sign": (seen(truth.get("curve_sign_m")), truth.get("curve_sign_m")),
               "end_marker": (seen(truth.get("end_m")), truth.get("end_m")),
               "release_sign": (seen(truth.get("release_sign_m")), truth.get("release_sign_m"))}
        dets = {}
        for name, (hit, d) in raw.items():
            if hit and self.dropout and self.rng.random() < self.dropout: hit = False
            dets[name] = dict(score=float(hit), dist_m=self._noisy(d) if hit else 0.0, box=None)
        probs = None
        if int(truth.get("visible", 0)) == 1 and truth.get("color") in COLOR_CLASSES:
            probs = np.zeros(5, np.float32); probs[COLOR_CLASSES.index(truth["color"])] = 1.0
        self.last = dict(dets=dets, probs=None if probs is None else probs.tolist())
        return dets, probs

    def get_state(self):
        return dict(rng=self.rng.bit_generator.state, last=copy.deepcopy(self.last))

    def set_state(self, state):
        self.rng.bit_generator.state = state["rng"]; self.last = copy.deepcopy(state["last"])
