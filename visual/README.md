# P-V, P-B and P-LP: the 4 km visual system

Paper: main Sections III-B/C, IV–V, VI-C/D/E; Supplementary Sections B, E, G, H.

## Layout

| Path | Contents |
|---|---|
| `v19_deps/` | archived 4 km plant, energy model, route, `study.py` (StudyEnv, rewards, metrics), release-aware execution base `curve_layer.py`; short-route actor weights used for adaptation; SHA-256 provenance |
| `visual_dev/renderer.py` | procedural camera (96×160 RGB, 2 Hz): curve-warning sign 400 m before the curve, release sign at the exit, lamps within 200 m, endpoint markers, per-episode appearance and sensor noise |
| `visual_dev/vision_state.py` | `VisionMemory`: detection voting, range propagation, near-line freezes, phase hold and commitment; executor targets |
| `visual_dev/pipeline.py` | `SmoothEnv`/`VisionEnv` (memory-driven release-aware executor), `Perception` (frozen detector), frame store, replay, visual TD3 trainer, evaluation loop |
| `visual_z/` | encoder, input adapter, contracts, learner (F/S/J/JH gradient routes), replay, checkpoint |
| `visual_dev/regen_pool.py`, `pretrain.py`, `v4_calibrate_thresholds.py` | supervised pool generation, detector/encoder pretraining, detection-threshold calibration |
| `visual_dev/run_stage.py`, `reevaluate.py`, `summarize_pilot.py` | audit → adaptation → arm training → report; re-evaluation of exported arms; run summaries |
| `visual_dev/v4_probe_closed_loop.py` | constant-command probe (u=1) with oracle or visual memory and cue-removal interventions |
| `visual_dev/v5_timing_experiment.py`, `v6_theory_lp.py` | P-B short branches from frozen approach states (`freeze`, `run6`, `summarize6`); sampled two-stage LP reference (P-LP) |
| `visual_dev/v6_finetune_off.py` | excluded dark-lamp detector candidate (Supp. H.1); its report is `runs/pretrain_v4_off/finetune_report.json`, its probe `runs/probes/v6_probe_off_encoder.json` |
| `configs/v4r_cpu*.json` | the three training configurations (seeds 7/8/9, camera seeds 77/78/79) |
| `runs/pretrain_v4/` | frozen detector `encoder.pt`, calibrated `thresholds.json`, pretraining report, fixed audit sample |
| `runs/v4r/common/` | common 2400-substep adaptation networks |
| `runs/{v4r,v4r_s1,v4r_s2}/v4_pilot/{frozen,supervised,joint,joint_head}/` | `final_nets.pt`, `evaluation.json` (nine development conditions), `evaluation_traces/dev_XX.npz` and `_layer.json` (substep traces and executor records) |
| `runs/probes/v4r_probe_closed_loop.json` | historical constant-command probe (Table V, last row) and cue-removal probes |
| `runs/v5_timing/` | frozen approach states (22.2 and 16 m/s), 408 + 456 branch records (`v6_formal_v22`, `v6_formal_v16`, `closure.md` summaries), LP results `theory_lp*.json` |
| `reproduction/` | repository-bound copy of the paper's evaluation wrapper and protocol identity; `verification_seed0.json` |
| `tables.py` | regenerates main Table V and Supplementary Tables S6/S7 from the records and checks them against the paper (`tables.json`) |

F/S/J records use the current protocol (no true-position terminal command override); JH records retain the legacy override and are reported separately (Supp. G.1).

## Commands

```bash
cd visual
python tables.py                                                  # Tables V, S6, S7 from records (NumPy only)
python reproduction/reproduce_visual.py check --root .            # source identity + exact protocol difference
python reproduction/reproduce_visual.py run --root . --seed 0 --arm supervised --protocol current --out /tmp/s0_S   # inference only, PyTorch CPU
python visual_dev/v4_probe_closed_loop.py --config v4r_cpu.json --out /tmp/probe.json                              # constant-command probe with cue removal
python visual_dev/v6_theory_lp.py                                  # sampled LP reference (SciPy)
python visual_dev/v5_timing_experiment.py summarize6 --tag v6_formal_v22   # P-B summaries from the branch records
```

Training from scratch (CPU, several hours per arm) follows `configs/v4r_cpu.json`: `regen_pool.py --out pretrain_v4 --coverage v4` → `pretrain.py` → `v4_calibrate_thresholds.py` → `VISUAL_CONFIG=v4r_cpu.json python visual_dev/run_stage.py audit|adapt|train --arm <arm> --tag v4_pilot|report`. The P-B branches were generated with `v5_timing_experiment.py freeze` (both approach states) and `run6 --tag v6_formal_v22|v6_formal_v16 --encoder runs/pretrain_v4/encoder.pt --hide-key color_mask` (see the script's argument defaults for the policy list, jerk limits, information conditions and adoption distances).
