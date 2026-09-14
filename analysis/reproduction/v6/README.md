# Sampled planning reference and raw-record audit

`v6_theory_lp.py` is the supplied standalone generator, preserved without changing its modeling choices. Supplementary Section E.1 and the `*_complete.json` files give complete sampled scans. Every latest feasible delay is followed by an infeasible grid point. The supplied shorter scans remain available as generator inputs; they are not used to censor the current boundaries.

`theory_lp.json` and `theory_lp_v16.json` are supplied results. Paired objectives reproduce exactly at the saved reference horizons. The higher-speed reference horizon is 13.5000135000135 s while the action grid has 27 half-second steps; the common offset cancels in the paired contrast. The auditor uses saved horizons explicitly.

From the analysis folder, after extracting the supplied v6 archive:

```bash
python verification/audit_v6.py --root /absolute/path/to/extracted/evcRL_v6_closure --out /absolute/path/to/new_audit_results.json
```

This reads JSON and imports only the LP generator. It needs Python, NumPy and SciPy, and does not load pickle, model weights or the vehicle simulator. It launches no training or new closed-loop branch.

The increment contains five source files, frozen-state bundles, reports and branch records. Vehicle inference additionally requires the original configuration, rendering, learner and weight dependencies. The increment alone is not a complete experiment repository.
