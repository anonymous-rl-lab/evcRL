# Reproducing the archived visual evaluation convention

This companion makes the two published terminal conventions explicit. It is an inference entry point over the original extracted experiment package, not a revised learner or a new experimental result.

| Published records | Seed indices | Required protocol | Terminal command |
|---|---|---|---|
| F / S / J | 0, 1, 2 | current | Retain the held actor command |
| JH | 0, 1, 2 | legacy | Original true-position `cmd=0` branch |

Seed indices 0/1/2 correspond to training seeds 7/8/9 and run directories `v4r`, `v4r_s1`, `v4r_s2`. All nine development conditions retain their original enumeration and camera streams. The evaluation camera seed remains 1000; the training camera seeds 77/78/79 are not substituted for it.

Use the safely extracted core archive `evcRL_v4r_vol1_core.zip`, with its original relative dependency layout. The separate layer-log archive is needed for raw-record audit but not for inference. Full original archive hashes appear in `../verification/INPUT_SHA256.json`.

First check source identity and the exact protocol change, using only Python's standard library:

```bash
python reproduction/reproduce_visual.py check --root /absolute/path/to/extracted_visual_root
```

`protocol_identity.json` binds the relevant learner, detector, memory, simulator and evaluator source files. A changed source is rejected. `current_protocol.patch` shows the exact one-line difference; the wrapper applies it only in memory. Do not apply the patch to the input directory before using the wrapper, because that would invalidate its source identity.

Optional inference requires the original experiment's NumPy, PyTorch and other runtime dependencies. Load only the supplied trusted checkpoints. For the main S configuration, seed 0:

```bash
python reproduction/reproduce_visual.py run --root /absolute/path/to/extracted_visual_root --seed 0 --arm supervised --protocol current --out /absolute/path/to/reproduction_s0
```

For legacy JH:

```bash
python reproduction/reproduce_visual.py run --root /absolute/path/to/extracted_visual_root --seed 0 --arm joint_head --protocol legacy --out /absolute/path/to/reproduction_jh0
```

The output directory must be new and outside the original source archive. The wrapper writes nine inference outcomes and their trajectories there; it does not overwrite publication records, run training, probe Z, or repeat masking experiments. It reports checkpoint and detector hashes along with the selected protocol. It deliberately does not offer JH under the current protocol as a reproduction of the published legacy results.

The entry-point checks cover source hashes, a one-node executable AST difference, and successful syntax compilation. Recorded independent replays of seed-0 F/JH condition 0 under both conventions are retained in `../verification/replay_checks.json`. No new vehicle rollouts were performed to validate this wrapper; complete numerical reproduction across every seed/arm and runtime is not claimed. The documented training action-interface mismatch is preserved and documented in Supplementary Section B.4.
