"""Frozen configuration profiles and their identities.

Every rule parameter that decides the behaviour of the memory, the executor and
the physical backend is written out here explicitly, so that

* ``paper_v25`` reproduces the archived TIV v25 rules (legacy propagation
  ``v*dt``, no range guard) and cannot drift when a module default changes;
* corrected variants (``odometry_v26``, ``guard_v26``) carry their own identity;
* a snapshot or a logged experiment records ``profile_identity`` and refuses to
  be restored into an environment whose frozen configuration differs.

``profile_identity(name)`` hashes the frozen dictionary together with
``CONTRACT_VERSION``; it is independent of the package version so that a pip
upgrade which does not touch the rules keeps old snapshots loadable, while any
change to a rule parameter or to the environment contract produces a new hash.
``source_identity()`` additionally hashes the rule-implementing modules.
"""
import copy
import hashlib
import json
from importlib import resources

CONTRACT_VERSION = 1   # observation/action/info/snapshot contract of EcoDriveEnv

_MEMORY_V25 = dict(
    det_thr={"traffic_light": 0.4, "curve_sign": 0.5, "end_marker": 0.55, "release_sign": 0.2},
    hold_s=3.0, curve_len_max=1000., release_grace_m=5., init_votes=2, vote_window=3, consistency_m=40.,
    expire_s=4.0, light_expire_s=10.0, gain=0.4, margin_rel=0.15, margin_abs=6.0, color_freeze_m=8.0,
    dist_freeze_m=15.0, expire_far_m=60.0, maintain_ratio=0.5, end_bias_m=15.0, no_light_near_end_m=100.0,
    stationary_expire_s=10.0)

_EXECUTION_V25 = dict(jerk=2.0, dt=0.5, a_comfort=1.5, assumed_green_remaining=30.0)

_EPISODE_V25 = dict(
    initial=dict(soc=0.85, temperature=288.15, speed=16.0),
    settle=dict(v=0.05, a=0.05), overshoot_m=100.0,
    routes={"4000.0": dict(curve_mode="envelope", t_budget=480.0, deadline_s=600.0),
            "20000.0": dict(curve_mode="penalty", t_budget=1050.0, deadline_s=1312.5)})

PROFILES = {
    "paper_v25": dict(
        description="Archived TIV v25 rules: memory propagates by v*dt, no range guard at the 15 m distance freeze.",
        status="frozen",
        memory=dict(_MEMORY_V25, propagation="legacy", guard=False),
        execution=dict(_EXECUTION_V25),
        episode=copy.deepcopy(_EPISODE_V25)),
    "odometry_v26": dict(
        description="Corrected variant: memory propagates by the physical displacement of the substep (odometry-consistent).",
        status="corrected",
        memory=dict(_MEMORY_V25, propagation="consistent", guard=False),
        execution=dict(_EXECUTION_V25),
        episode=copy.deepcopy(_EPISODE_V25)),
    "guard_v26": dict(
        description="Corrected variant: odometry propagation plus a prospective inward-crossing guard at the 15 m freeze threshold (experimental).",
        status="experimental",
        memory=dict(_MEMORY_V25, propagation="consistent", guard=True),
        execution=dict(_EXECUTION_V25),
        episode=copy.deepcopy(_EPISODE_V25)),
}

_RULE_MODULES = ("backend.py", "camera.py", "effcal.py", "env.py", "execution.py", "memory.py", "models.py", "plant.py", "profiles.py", "route.py")


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def get_profile(name):
    if name not in PROFILES:
        raise ValueError("unknown profile %r; available: %s" % (name, sorted(PROFILES)))
    return copy.deepcopy(PROFILES[name])


def profile_identity(name):
    """Hash of the frozen rule configuration (independent of the package version)."""
    return _digest(dict(profile=name, contract=CONTRACT_VERSION, rules=PROFILES[name]))


def source_identity():
    """Hash of the installed rule-implementing modules (changes with any code edit)."""
    pkg = resources.files("evcrl")
    h = {m: hashlib.sha256(pkg.joinpath(m).read_bytes()).hexdigest() for m in _RULE_MODULES}
    return _digest(h)
