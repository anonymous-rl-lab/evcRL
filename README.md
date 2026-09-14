# evcRL — Environment-Aware Reinforcement Learning for Electric Vehicle Eco-Driving with Smooth Execution

Companion repository of the manuscript *Environment-Aware Reinforcement Learning for Electric Vehicle Eco-Driving with Smooth Execution* (TIV, v27; main paper and supplement under `paper/`). The repository contains the environment package, the frozen experiment code, the archived records behind every table of the paper, and scripts that regenerate those tables from the records or re-run the frozen policies through the archived simulators.

## Paper protocol → repository map

| Paper | Directory | Contents | Regenerate / verify |
|---|---|---|---|
| Sec. III-D, Supp. C — reusable environment interface | `software/EvcRL/` | `evcrl` Gymnasium package (camera 4 km, structured 20 km), wheel + sdist, contract tests, bit-exact equivalence records against the archived simulators | `pip install software/EvcRL/dist/evcrl-0.0.1-py3-none-any.whl`; `pytest software/EvcRL/tests` |
| P-V — Sec. VI-C, Supp. B, G (Tables V, VI, S6–S8) | `visual/` | procedural camera, detector pretraining, event memory, release-aware executor, TD3 arms F/S/J/JH; frozen detector and 12 policy checkpoints; 27 evaluation records per arm; constant-command probe; reproduction wrapper | `python visual/tables.py`; `python visual/reproduction/reproduce_visual.py check --root visual` |
| P-B, P-LP — Sec. IV–V, VI-D/E, Supp. E, H (Tables VII, VIII, S9–S17, Figs. 2, S3) | `visual/runs/v5_timing/`, `visual/visual_dev/v5_timing_experiment.py`, `visual/visual_dev/v6_theory_lp.py`; `paper/verification/` | 864 short-branch records from two frozen approach states; sampled LP reference; record calculators | `python visual/visual_dev/v5_timing_experiment.py summarize6 --tag v6_formal_v22`; paper calculators below |
| P-C — Sec. VI-A, Supp. F.1–F.4 (Tables III, S2–S4, Fig. S2) | `comfort/` | archived complete-executor development package (r0/r1/r2), settled baseline traces, extracted frozen actors A/B | `python comfort/reproduce.py [--rollout]` |
| P-T — Sec. VI-B, Supp. F.5–F.8 (Table IV, S5, Fig. 3) | `timing/` | self-contained timing-use experiment: frozen actor A, paired color/timing records | `python timing/code/audit.py`; `python timing/verify_tables.py` |
| Reference context — Sec. VI-F, Supp. I | `long_route/` | frozen 20 km simulator and TD3 trainer, eight archived seeds (curves, selected checkpoints, references), gates, checkpoint replay, retraining scripts | `bash long_route/scripts/run_gates.sh`; `python long_route/scripts/replay_selected_checkpoints.py` |
| Manuscript, supplement, figures, record calculators — Supp. J | `paper/` | Markdown/PDF/TeX sources, figure generators, verification calculators and their frozen inputs/outputs | `python paper/verification/check_v27.py` and the calculators listed in `paper/verification/VERIFICATION_v27.md` |

Seed indices 0/1/2 of the visual study are training seeds 7/8/9 and run directories `visual/runs/v4r`, `v4r_s1`, `v4r_s2`. Arms F/S/J/JH are `frozen`, `supervised`, `joint`, `joint_head`.

## Environment

Python ≥ 3.10, NumPy, SciPy, Pillow, Gymnasium (for `evcrl`), PyTorch CPU (policy inference and training; no CUDA is used anywhere), Matplotlib (figures). The record calculators and table scripts need NumPy/SciPy only. Every simulation runs single-threaded on a CPU; set `OMP_NUM_THREADS=1`.

The archived 20 km code reads `EVSIM_ROUTE` at import: the visual, comfort and timing packages set it to `mini` (4 km) themselves; the long-route scripts unset it. Do not import the 4 km and 20 km stacks in one process.

## What was verified in this repository state

| Check | Result |
|---|---|
| `visual/tables.py`: Table V, S6, S7 regenerated from the 108 evaluation records and the constant-command probe | all values match the paper |
| `visual/reproduction/reproduce_visual.py`: seed-0 S (current protocol) and JH (legacy protocol) re-run through the repository sources | all nine conditions bit-identical to the archived records (`visual/reproduction/verification_seed0.json`) |
| `comfort/reproduce.py`: Tables III, S3, S4 and the arrival-cut numbers from the archived traces; frozen actors A/B re-run through both executors | all values match; complete-executor trajectories bit-identical, original-executor trajectories within 1e-8 (`comfort/reports/reproduction_checks.json`) |
| `timing/code/audit.py`, `timing/code/analyze.py pilot`, `timing/verify_tables.py` | source/weight identity intact; Tables IV and S5 and the condition-7 event match (`timing/reports/table_checks.json`) |
| `paper/verification/*`: framework, diagnostics, target-memory calculators, preparation checks, `check_v27.py` (418 assertions) | all pass; regenerated outputs identical to the delivered package |
| `long_route/scripts/run_gates.sh` | check_env, six verify.py checks, eight regression tests pass (`long_route/results/gates.txt`) |
| `software/EvcRL`: 26 contract tests; camera `paper_v25` vs archived `VisionEnv`, structured 4/20 km vs archived `env20` | pass; bit-identical (`software/EvcRL/SOFTWARE_VERIFICATION.json`) |

The manuscript text refers to EvcRL 0.1.0 with ten software tests; this repository ships the hardened 0.0.1 release (26 tests, frozen `paper_v25` rule identity, bit-exact against the archived simulators). The packaged rules are unchanged; only the packaging and its verification changed.

## Assets not in the repository

Regenerable large assets are ignored: the supervised image pools and evaluation sets (`visual/runs/pretrain_v4/pool.pt`, `eval_sets.pt`; regenerate with `visual/visual_dev/regen_pool.py`), training resume checkpoints and replay stores, and the 20 km training outputs under `long_route/evsim_v9/out`. The complete `resume.pt` checkpoints of the comfort study's actors are replaced by the extracted actor tensors in `comfort/frozen_actor_replay/`, whose provenance is recorded there.

## Licence

Apache License 2.0 (`LICENSE`).
