# Offline adoption and LP-reference calculations

Run `python verification/framework/audit_offline.py` from the paper root. The calculator reads `log_excerpts.json.gz` and the unchanged mathematical routines in `source/`. It produces 432 stopping-adoption rows, 27 visual-adoption rows, 102 passing-reference rows, a cross-table, and JSON summaries without running a vehicle or training a network.

`source_hashes.json` identifies the external source records used to create the field-selected excerpts. Source filenames preserve experimental identity. These excerpts are sufficient for this calculator, but do not replace full trajectories, image pools, or learned weights. The optional `--workspace` extraction path requires the separately supplied source archives in their original working layout; default reproduction uses the packaged excerpts.

The current manuscript is checked by `../check_v27.py`. The release/target hierarchy is specified in main Section IV and Supplementary Section D. Shared-execution identity does not imply equal wrappers, actor information, or target histories.
