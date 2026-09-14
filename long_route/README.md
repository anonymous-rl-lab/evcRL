# Structured 20 km reference study

Paper: main Section VI-F and Supplementary Section I (eight archived structured-input seeds, numerical references, programmed drivers, checkpoint selection and failures).

| Path | Contents |
|---|---|
| `evsim_v9/` | frozen 20 km simulator and TD3 trainer (`env20.py`, `models.py`, `plant.py`, `effcal.py`, `route20.py`, `td3_run.py`, `run_batch.py`, `analyze.py`, `evaluate_checkpoint.py`, DP references `dp20L.py`/`dp20I.py`, gates `check_env.py`, `verify.py`, `test_regressions.py`); `MANIFEST.sha256` records its identity |
| `reference/` | archive of the eight paper seeds: `bench_curves/` (twenty checkpoints × twelve conditions per seed), `bench_results.json`, `selected_checkpoints/` (validation-selected weights), `refs_fixed.json` (DP, driver and constant-speed references), `cruise_baseline.json`, `recomputed_v17.json` |
| `scripts/` | `run_gates.sh`, `smoke_bit_identity.sh`, `replay_selected_checkpoints.py`, `train_repro.sh`, `compare_curves.py`, `plot_curves.py`, `_paths.py` |
| `results/gates.txt`, `results/smoke/` | gate outputs and the 12k-step bit-identity smoke test |
| `results/replay/` | the eight selected checkpoints replayed through the frozen simulator: twelve-condition grids identical to the archived training logs (max |ΔR| = 0) |
| `results/train/` | additional reproduction runs under the frozen protocol (seeds 0/1/2, 5 million steps each, CPU) — curves, selection, comparison with the archived curves; not part of the paper's eight-seed evidence |

```bash
bash scripts/run_gates.sh                       # check_env, verify.py, regression tests
bash scripts/smoke_bit_identity.sh              # 12k-step smoke: R=-387.723, last-3=-394.262 with --log 6000
python scripts/replay_selected_checkpoints.py   # replays the eight selected checkpoints; grid error must be 0
SEEDS="0 1 2" JOBS=3 bash scripts/train_repro.sh; python scripts/compare_curves.py   # optional retraining (hours)
```

`EVSIM_ROUTE` must be unset for this package (the scripts unset it). Retraining across PyTorch/NumPy versions is not bit-identical beyond the first checkpoints; exact per-seed numbers of the paper are reproduced by checkpoint replay, not by retraining.
