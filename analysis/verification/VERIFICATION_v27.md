# Verification records (v27 calculators)

## Current checks

- offline_reproducibility_v27.json: three record calculators completed. Twelve outputs match frozen outputs byte for byte. All 195 target-accounting scientific rows and summary fields are identical after removing reviewer-comparison fields.
- preparation_formula_checks.json: 432 analytical parameter settings, 864 independent nodal LP solves, and exact integration of 1008 mathematical trajectories verify the retained continuous proposition. Maximum optimizer/formula discrepancy is below 4.0e-15. This is not vehicle evaluation.
- preparation_residual_checks.json: 288 deterministic and 72 randomized-preparation checks passed; maximum identity discrepancy is below 1.06e-14 s.

## Evidence boundaries

No new RL training or vehicle-driving comparison was run. Mathematical integration and fixed-record memory replay are distinct from driving experiments. The environment implementation and its previous finite API-test results are unchanged; no new software-performance claim is made.

Complete memory replay uses 27 P-V detection streams and reproduces all 12,081 recorded snapshots at tolerance 1e-8. This tests recorded memory fields, not unlogged hidden-state equality. The candidate guard changes selected static margins and two passed-signal track-clearing times; future observations under altered actions remain untested.

Current visual, branch and event statistics are recomputed from packaged records. Retained comfort and long-route tables are also checked against frozen source-table cells. Full scientific regeneration of those experiments requires their external records; frozen-table retention is not presented as a new experiment regeneration.

## Entry points

From the analysis folder:

```bash
python scripts/check_manifest.py
python verification/framework/audit_offline.py
python verification/diagnostics/audit_events.py
python verification/target_memory/audit_memory.py
python verification/verify_preparation.py
python verification/verify_preparation_residual.py
```

Check the manifest before regeneration. NumPy/SciPy support numerical checks; the memory calculator imports the environment package from ../software/EvcRL/src.

The environment companion is supplied as source and a local wheel. It has no default trained detector/driver, no public package-index release, and no full training-state snapshot facility. Experimental memory profiles remain explicit.
