# Frozen supplementary validation (99 formal frozen-inference episodes)

Implements and runs the task book `CODEX_TASK.md`: R1 (three frozen S policies × nine conditions × `paper_v25`/`guard_v26`, 54 episodes), R2 (constant command u=+1 × nine conditions paired with S0/`paper_v25`, 9 episodes), R3 (frozen actor A × eighteen extension conditions × current-color/color+timing, 36 rollouts). No training, no model selection, no parameter search.

| File | Role |
|---|---|
| `protocol_lock.py` → `protocol_lock.json`, `preflight.json`, `jobs.csv` | asset/code identities, protocol fields, 99-job list with fingerprints |
| `run_pv.py` | R1/R2 runner derived from `visual/visual_dev/pipeline.py::evaluate` (current protocol, explicit memory profile, actual displacement to memory, augmented substep log, atomic resumable episodes); `evaluate_derivation.diff` shows the derivation |
| `run_pt.py` | R3 runner reusing `timing/code/probe.py` unchanged on `conditions('reporting')`; the archived gated `reporting` stage is not modified |
| `metrics.py` | independent recomputation from raw records → `episodes.csv`, `pairs_R{1,2,3}.csv`, `events_R1.csv`, `guard_events_R1.csv`, `tables.md`, `metrics_summary.json`, figures |
| `baseline_check.py` → `baseline_check.json` | exactness of the `paper_v25` arm against the archived S records |
| `runtime_budget.py` → `runtime_budget.json` | measured per-episode rates and the formal estimate |
| `REPORT_zh.md`, `paper_paragraphs_en.md` | report and replaceable manuscript paragraphs |
| `results/` | every raw episode record (trace, executor log, augmented substep log, summary, DONE flag) and run logs |
| `manifest.py` → `MANIFEST_SHA256.json` | delivered-file identities |

Resume/run: `cd validation_frozen && python3 protocol_lock.py && OMP_NUM_THREADS=1 python3 run_pv.py --select R1,R2 && OMP_NUM_THREADS=1 python3 run_pt.py && python3 metrics.py && python3 baseline_check.py && python3 runtime_budget.py && python3 manifest.py`
