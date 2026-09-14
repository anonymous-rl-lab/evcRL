# P-T: executor timing-use intervention (TIV_SPaT_Execution_v1)

Paper: main Section VI-B (Table IV, Fig. 3) and Supplementary Sections F.5–F.8 (Table S5).


This self-contained experiment reuses the frozen unregularized short-route actor A from development seed 7. It adds no training and does not change the v18 manuscript or eight-seed long-route evidence.

Read `protocol/PROTOCOL.md` for the frozen design and `reports/decision.json` for the stage decision; `python verify_tables.py` checks main Table IV and Supplementary Table S5 of the manuscript against `reports/pilot.json` and `reports/pilot_pairs.csv`. The paired effect is confined to one late-green event; eight completed pairs have exactly identical saved traces. The planned eighteen reporting conditions were not opened.

Environment: Python 3.12, NumPy 2.3.5, PyTorch 2.8.0+cpu. Matplotlib is used only for the event figure. Install PyTorch using its CPU distribution; no GPU or external paid service is required.

From this directory, verify or resume:

```bash
python code/audit.py
python code/probe.py smoke
python code/analyze.py smoke
python code/probe.py pilot
python code/analyze.py pilot
python reports/make_event_figure.py
```

Completed conditions are skipped and interrupted conditions resume from decision-boundary snapshots. The actual run retains its pre-evaluation hash in `audit/freeze.json`. Calling the audit again regenerates audit output; the archived original is additionally retained in `audit/original_preflight.json` and `audit/original_freeze.json`. Do not change code, weights, dependency files or protocol under an existing experiment identity. The runner rejects a source-hash mismatch.

`python code/probe.py reporting` is intentionally gated and must reject the present failed pilot gate. A future differently scoped study requires a separately documented protocol and output directory, not bypassing this gate.

`deps/` contains byte-identical source dependencies from the released v18 experiments, including historical training classes that are not called. `weights/actor_A.pt` contains only the six unchanged actor tensors extracted from the complete source training checkpoint. `protocol/provenance.json` records their origin. New trajectories, ledger columns, selected/shadow actions, condition progress, and independent metric recomputations are included. Shadow actions are same-state diagnostics and never drive the alternative physical rollout.

The raw analyzer includes descriptive means over all recorded trips. Because the color-only arm fails in condition 7, these are not matched complete-route cost comparisons. In particular, its nominal aggregate signal-jerk reduction must not be presented as an efficiency or comfort benefit. The interpretation file and report explain this explicitly.
