# EvcRL 0.0.1

A [Gymnasium](https://gymnasium.farama.org) environment for longitudinal electric-vehicle eco-driving with a procedural RGB camera, a sourced energy model (regenerative braking, load-dependent efficiency, pack acceptance ceiling), a camera-event memory and a sampled jerk-bounded execution layer. It packages the environment used in the accompanying manuscript (TIV v25/v26) so that policies, detectors and executors can be exchanged against a fixed, versioned reference.

Installing and importing the package needs NumPy, Pillow and Gymnasium only. No Torch and no CUDA are imported; the base environment runs on a CPU laptop.

| capability | this release |
|---|---|
| environments | `EvcRL-Camera-v0` (4 km, RGB 96×160 at 2 Hz), `EvcRL-Structured-v0` (20 km, 13-dim structured), `EvcRL-StructuredMini-v0` (4 km structured) |
| perception | procedural camera; signals visible within 200 m of the stop line; curve-warning sign 400 m before the curve; release sign at the curve exit; end markers |
| control | any policy through `step`; replaceable detector callback; per-instance `VisionMemory` and `Executor` |
| physics | vehicle energy ledger, regeneration ceiling, actual accelerations and jerk under the archived plant |
| diagnostics | per-substep execution record (command, applied acceleration, interval, fallback reason, target distances, speed limit), memory snapshot, memory update diagnostic |
| restoration | complete episode snapshot (vehicle, memory, camera and RNG state, frame history, command boundary, detector state) with configuration-identity checks |
| interface | Gymnasium `reset/step`; `terminated` = task outcome, `truncated` = external cap only |

## Install

```bash
python -m pip install evcrl-0.0.1-py3-none-any.whl      # or: python -m pip install .
evcrl-smoke --mode camera --oracle --episode             # finite API smoke, not an evaluation
python -m pip install 'evcrl[test]' && python -m pytest tests -q
```

Optional extras: `analysis` (SciPy, Matplotlib for the companion scripts), `torch` (only needed by an external learned detector or a training framework; nothing in this package imports it), `test`.

## Quick start

```python
import gymnasium as gym, numpy as np, evcrl

env = gym.make("EvcRL-Camera-v0", profile="paper_v25", detector=evcrl.OracleDetector(seed=0))
obs, info = env.reset(seed=0, options={"signal_offset": 40.0})
obs, reward, terminated, truncated, info = env.step(np.array([0.3], np.float32))
print(info["command_a"], info["applied_a"], info["termination_reason"])
state = env.unwrapped.get_state();  env.unwrapped.set_state(state)
```

`obs` is a dict with `rgb` (uint8 96×160×3) and `ego` (speed, acceleration, SOC, temperature, elapsed time, remaining task time). The structured environments return the archived 13-dimensional observation (speed, acceleration, remaining distance, profile limit, curve preview, signal distance/phase/time-to-change within 1000 m, SOC, temperature, clock).

## Action semantics

`step` receives one normalized command `u ∈ [-1, 1]` (shape `(1,)`; a scalar is accepted; values outside the box are clipped and flagged in `info["command_clipped"]`, or rejected with `clip_actions=False`). The raw acceleration command is `2.6·u` for `u > 0` and `3.5·u` otherwise (`evcrl.normalized_to_command`), and it is **held for four 0.5 s substeps**. In every substep the executor decides the acceleration actually applied. The execution record is returned, not summarised:

* `info["applied_a"]` — the four applied accelerations;
* `info["substeps"]` — per substep: `command_u`, `command_a`, `applied_a`, `v0`, `v1`, `odom_ds`, `jerk`, `jerk_exceeded`, `fallback`, `reason`, `envelope_braking`, `signal_reason`, `interval`, `v_limit`, `target_distances`, `reward`, `reward_components`.

The command is the environment action. The package does not offer any applied acceleration (first substep or otherwise) as a learning label; a learner that needs the executed acceleration reads the record. `evcrl.command_to_normalized` is the inverse mapping for adapters.

## Termination and truncation

* `terminated` with `info["termination_reason"]` in `settled` (arrived and at rest), `red_crossing`, `task_deadline`, `overshoot` (camera task, > 100 m past the destination).
* The intrinsic deadline (`task_deadline_s="historical"`: 600 s for 4 km, 1312.5 s for 20 km) is a task outcome and carries the archived terminal cost-to-go; `task_deadline_s=None` removes it (the clock observation channel is then zero).
* `truncated` only through `max_episode_seconds`; it carries no terminal cost. Without either limit an episode is unbounded — wrap with `gymnasium.wrappers.TimeLimit` if a policy can stall.
* Stepping an ended environment raises `RuntimeError`.

## Perception

Without a detector the camera environment is blind: the executor has no road-event targets and the archived judge scores the outcome (red crossings, curve overspeed, arrival). Attach a detector with `detector=callable`:

```python
def detector(rgb: np.ndarray) -> tuple[dict, np.ndarray | None]:
    # dets: {"traffic_light" | "curve_sign" | "end_marker" | "release_sign": {"score": float, "dist_m": float}}
    # color_probs: five probabilities red/yellow/green/off/unknown, or None when no signal is seen
```

Outputs are validated (unknown classes, non-finite values or a malformed probability vector raise `ValueError`). Optional hooks: `reset()`, and `get_state()/set_state()` (required for snapshots). The callback receives only the RGB frame; renderer labels and truth are never passed to it.

`evcrl.OracleDetector` is a truth-based detector (`detector.requires_truth = True`; the environment then passes the renderer's truth distances). It reproduces the archived `oracle` probe and exists for tests, executor studies and debugging; `info["detector_kind"] == "oracle"` marks such runs, which must not be reported as camera results. The manuscript's learned detector, its weights and the training framework live in the research repository; wrap them in the callback above.

`env.unwrapped.memory_observation(extended=False)` returns the legacy 13-dim memory-generated observation and the 6 (or 10) extra memory entries for frozen-policy adapters.

## Profiles and identities

| profile | status | rule difference |
|---|---|---|
| `paper_v25` | frozen | archived TIV v25 rules: memory propagates event distances by `v·dt`; no range guard at the 15 m distance freeze |
| `odometry_v26` | corrected | memory propagates by the physical displacement of the substep |
| `guard_v26` | experimental | odometry propagation plus a prospective inward-crossing guard at the freeze threshold |

`evcrl.PROFILES` lists every rule parameter explicitly (memory thresholds and margins, executor jerk family, comfort envelope, initial state, settle thresholds, route budgets and deadlines). `evcrl.profile_identity(name)` hashes that frozen dictionary with `evcrl.CONTRACT_VERSION`; it is pinned in `tests/expected_identities.json`, so a change of any rule or of the interface contract fails the test suite instead of silently altering old experiments after a `pip` upgrade. `evcrl.source_identity()` hashes the installed rule-implementing modules. Both are reported in `info` and stored in snapshots; `set_state` refuses a snapshot whose configuration identity differs and, by default, one taken with different sources.

`paper_v25` was checked against the archived research code: with the same commands, camera seed and oracle detection stream, position, speed, applied acceleration, reward, energy ledger, memory snapshots and rendered frames are bit-identical over complete 4 km episodes, and the structured 4 km and 20 km backends are bit-identical to the archived `env20` (see `SOFTWARE_VERIFICATION.json`).

Supported release-bound jerk limits are 2, 3 and 4 m/s³ at the 0.5 s physical step (`jerk=`). Route geometry, profile and executor parameters are per instance; importing the package or creating environments modifies no module-level configuration and reads no environment variables, so several differently configured environments coexist in one process.

## Reset options

`options` accepts `soc` (0.85), `temperature` (288.15 K), `speed` (16 m/s), `signal_offset` or `signal_offsets` (seconds; by default sampled per episode from the seeded RNG), `camera_seed` (by default drawn from the seeded RNG). The seed determines signal offsets, camera appearance and sensor noise.

## Snapshots

`get_state()` between public steps returns a deep copy of the vehicle, the memory, the camera (appearance and noise RNG), the environment RNG, the frame history, the command boundary and the detector state. `set_state()` restores it into an environment with the same configuration identity. Snapshots are environment states, not optimizer or replay checkpoints; restore only trusted local data.

## Verification and scope

`python -m pytest tests -q` covers import hygiene (no Torch, no global mutation), Gymnasium `check_env` for the registered ids, action parsing and the command mapping, command holding and the execution record, the reward ledger, deadline vs. cap, blind and oracle-driven full episodes, detector validation, seed determinism, instance isolation, snapshot round trips (camera and structured, mid-episode, cross-instance), pickling, pinned identities, the memory guard, odometry propagation, the release-bound executor and the exported route profiles. `SOFTWARE_VERIFICATION.json` records the checked release including the equivalence runs against the archived code.

This release contains no trained detector, no policy weights, no training framework, no multi-vehicle traffic and no real-road validation. It is the environment, its rules and their identities.

## Licence and provenance

Apache License 2.0 (see `LICENSE`). `SOURCE_PROVENANCE.json` lists the SHA-256 digests of the archived research sources each module derives from and of the packaged modules. See `CHANGELOG.md` for the changes from the 0.1.0 draft.
