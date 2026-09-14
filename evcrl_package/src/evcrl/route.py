"""Instance-local versions of the two archived route profiles.

The speed-limit profile is read from the exported SUMO profiles bundled with the
package (``data/arterial_mini_profile.npy`` for 4 km, ``data/arterial_profile.npy``
for 20 km) so that the structured observation channel is bit-identical to the
archived environment (the exported values are rounded, e.g. 22.22 m/s, and are
NOT recomputed from 80/3.6).
"""
from dataclasses import dataclass
from importlib import resources
import numpy as np

_PROFILE_FILES = {4000.: "arterial_mini_profile.npy", 20000.: "arterial_profile.npy"}
_PROFILE_CACHE = {}


def load_profile(length):
    """(grid, v_limit) arrays of the exported profile for a supported route length."""
    length = float(length)
    if length not in _PROFILE_CACHE:
        with resources.files("evcrl.data").joinpath(_PROFILE_FILES[length]).open("rb") as f:
            g, v = np.load(f)
        _PROFILE_CACHE[length] = (np.asarray(g, float), np.asarray(v, float))
    return _PROFILE_CACHE[length]


@dataclass(frozen=True)
class Route:
    length: float = 4000.
    V_FREE: float = 80 / 3.6
    V_CURVE: float = 35 / 3.6
    CYCLE: float = 90.
    GREEN: float = 30.

    def __post_init__(self):
        if float(self.length) not in _PROFILE_FILES:
            raise ValueError("The release contains the archived 4 km and 20 km routes")
        object.__setattr__(self, "length", float(self.length))

    @property
    def LENGTH(self): return self.length
    @property
    def MINI(self): return self.length == 4000.
    @property
    def CURVES(self):
        return [(1450., 1550., self.V_CURVE)] if self.MINI else [(3450., 3550., self.V_CURVE), (10950., 11050., self.V_CURVE), (17950., 18050., self.V_CURVE)]
    @property
    def SIGNALS(self): return [3000.] if self.MINI else [7000., 15000.]
    def signal_green(self, position, t, offset): return (t + offset) % self.CYCLE < self.GREEN
    def time_to_change(self, position, t, offset):
        p = (t + offset) % self.CYCLE
        return self.GREEN - p if p < self.GREEN else self.CYCLE - p
    def profile_limit(self, x):
        g, v = load_profile(self.length)
        return float(np.interp(x, g, v))
