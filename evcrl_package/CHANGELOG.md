# Changelog

## 0.0.1 — first robust release (supersedes the 0.1.0 draft)

Fixes and hardening found while testing the draft in a clean environment:

* **Route profile**: the structured observation channel `profile_limit` was recomputed from `80/3.6`; the archived environment uses the exported SUMO profile (rounded values such as 22.22 m/s). Both exported profiles (4 km, 20 km) are now bundled and loaded, making the observation bit-identical to `env20`.
* **Import hygiene**: `camera.py` read `RENDER_CELL_ASSIGN` from the process environment at import; the validated `floor` assignment is now a constant. Importing the package reads no environment variables and loads no Torch.
* **Frozen profiles**: every rule parameter is written out in `evcrl.profiles.PROFILES`; `profile_identity()` and `CONTRACT_VERSION` are pinned by the tests; `source_identity()` digests the rule modules. Snapshots store both and `set_state` checks them.
* **Action handling**: scalars and shape-`(1,)` arrays accepted; commands are parsed in float64 (a float32 cast changed the raw command by ~1e-7); out-of-box values are clipped and flagged (`command_clipped`) or rejected with `clip_actions=False`; NaN and wrong shapes raise.
* **Execution record**: `info["applied_a"]`, per-substep `interval`, `v_limit`, `target_distances`, `envelope_braking`, `signal_reason`; `info["termination_reason"]`; memory snapshot and update diagnostic in `info`.
* **Detector contract**: outputs validated; optional `reset/get_state/set_state` hooks; `OracleDetector` (truth-based, explicitly labelled) for tests and executor studies; snapshot refuses a detector attachment mismatch.
* **Structured mode** no longer renders a camera frame every substep unless `render_mode="rgb_array"` (about 3× faster); `EvcRL-StructuredMini-v0` registered.
* **Reset options** validated (unknown keys raise); signal offsets and camera seed default to the seeded RNG so that `reset(seed=k)` fully determines an episode; explicit `signal_offset(s)`/`camera_seed` still supported.
* **Packaging**: version 0.0.1; Apache-2.0 licence from the research repository; SPDX metadata; `torch` extra declared for external detectors/training only; tests and identity pins shipped in the sdist.
* **Verification**: bit-exact equivalence runs against the archived research code (4 km camera with oracle stream for three signal offsets, 4 km and 20 km structured) recorded in `SOFTWARE_VERIFICATION.json`.
