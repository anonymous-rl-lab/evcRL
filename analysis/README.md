# Record calculators, verification and figures

Everything here recomputes the paper's comparisons from saved records and mathematical routines; nothing trains a network, runs a camera network, or steps the vehicle. The manuscript text is not part of the repository.

- `verification/framework/`: adoption and passing-reference calculator (`audit_offline.py`) over the field-selected P-B/P-V excerpts (`log_excerpts.json.gz`) and the unchanged rule sources in `source/`; outputs 432 stopping rows, 27 visual rows, 102 passing rows, cross-table, JSON summary.
- `verification/diagnostics/`: state-fixed target substitutions, visual event localization and the complete LP scans (`audit_events.py`).
- `verification/target_memory/`: trigger-aligned event accounting and recursive memory replay over the 27 complete P-V detection streams (`audit_memory.py`; imports `../software/EvcRL/src`).
- `verification/verify_preparation.py`, `verify_preparation_residual.py`: analytical checks of the continuous preparation proposition and the residual identity.
- `verification/audit_v6.py`, `audit_records.py`: auditors for the external v6 branch and layer-log archives (paths given on the command line).
- `verification/*.json`: frozen check results and record identities (`INPUT_SHA256.json`, `record_checks.json`, `replay_checks.json`, `protocol_entry_checks.json`, `v6_*`, `visual_rows.json`, `timing_event_rows.json`, ...).
- `reproduction/`: the published visual evaluation conventions (`reproduce_visual.py`, `protocol_identity.json`, bound to the original archive; the repository-bound copy is `../visual/reproduction/`) and the sampled LP generator with complete scans (`reproduction/v6/`).
- `scripts/`, `figs/`: figure generators and the figures they produce from the records above.
- `MANIFEST_SHA256.json`: identities of the files in this folder (`python scripts/check_manifest.py`).

Entry points and evidence boundaries: `verification/VERIFICATION_v27.md`. Requirements: `requirements_verification.txt` (NumPy, SciPy, Matplotlib).
