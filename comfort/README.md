# P-C: fixed-actor complete-executor comparison (archived package TIV_comfort_v2)

Paper: main Section VI-A (Table III) and Supplementary Sections F.1–F.4 (Tables S2–S4, Fig. S2).

Two frozen actors from development seed 7 (A: λ_c = 0; B: λ_c = 0.1) are driven through the original executor and through the release-aware complete executor (`code/curve_layer.py`, development revision r2) on the nine development conditions. No network is trained here.

| Path | Contents |
|---|---|
| `code/curve_layer.py`, `code/rollout_layer.py` | executor and the frozen-actor closed-loop evaluation with terminal settling |
| `code/diagnose.py`, `code/curve_witness.py`, `code/test_layer.py`, `code/summarize.py`, `code/write_report.py` | archived diagnostics, offline feasibility witnesses, layer tests and independent recomputation |
| `deps/` | archived 4 km simulator and `study.py` (byte-identical to `../visual/v19_deps/code`), the v1 protocol and initialization record |
| `runs/frozen_s7`, `frozen_s7_r1`, `frozen_s7_r2` | the three executor configurations of Table S2 (initial backup, target-type/signal handling, complete); r2 is the reported complete executor |
| `reports/settled_traces/` | original-executor trajectories with the terminal settling tail appended (baseline of Table III) |
| `reports/FINAL_ANALYSIS.json`, `reports/diagnosis.json`, `reports/curve_witness.json`, `reports/fixed_time_witnesses/` | archived analysis outputs |
| `actor_evaluations/` | archived nine-condition evaluations of A and B under the original arrival cut (77.861 / 49.083) |
| `frozen_actor_replay/actor_{A,B}.pt` | the six actor tensors extracted from the complete training checkpoints (`extract_and_replay.py` documents the extraction and needs the external archive) |
| `protocol/` | frozen probe protocol and the two implementation fixes; hashed by `rollout_layer.py` together with the code |
| `reproduce.py` | recomputes Tables III/S3/S4 and the arrival-cut numbers from the traces and checks them against the paper; `--rollout` re-runs both executors with the extracted actors and compares the trajectories with the archived traces (`reports/reproduction_checks.json`) |

```bash
cd comfort
python reproduce.py                    # record checks, NumPy only
OMP_NUM_THREADS=1 python reproduce.py --rollout --out reports/reproduction_checks.json   # PyTorch CPU, a few minutes
python code/summarize.py               # archived independent recomputation (expects the v1 evaluations in actor_evaluations/ via deps)
```
