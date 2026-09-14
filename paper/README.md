# Paper package (TIV v27)

*Environment-Aware Reinforcement Learning for Electric Vehicle Eco-Driving with Smooth Execution* — 10-page main paper and 22-page supplement.

- `Manuscript_v27.md` / `.pdf`, `Supplementary_v27.md` / `.pdf`: canonical Markdown and reading copies.
- `latex/`: synchronized TeX, IEEEtran class and PDF builder (`python latex/build_pdfs.py`; needs Pandoc and pdfLaTeX).
- `figs/`, `scripts/`: current figures and their generators (`scripts/make_*_figure*.py`).
- `verification/`: record calculators, their frozen inputs and outputs, formula checks and the manuscript checker. Entry points and evidence boundaries: `verification/VERIFICATION_v27.md`.
- `reproduction/`: the published visual evaluation conventions (`reproduce_visual.py`, `protocol_identity.json`) and the sampled LP generator with its complete scans (`reproduction/v6/`).
- `MANIFEST_SHA256.json`: identities of the files in this folder (`python scripts/check_manifest.py`).

Verification from this folder (NumPy/SciPy; the memory calculator imports the environment package from `../software/EvcRL/src`):

```bash
python scripts/check_manifest.py
python verification/framework/audit_offline.py
python verification/diagnostics/audit_events.py
python verification/target_memory/audit_memory.py
python verification/verify_preparation.py
python verification/verify_preparation_residual.py
python verification/check_v27.py
```

Relation to the rest of the repository: the environment companion named in Supplementary Section C lives in `../software/EvcRL` (release 0.0.1); the visual sources bound by `reproduction/protocol_identity.json` are those of the archive `evcRL_v4r_vol1_core.zip`, and the repository-bound wrapper with the current source identities is `../visual/reproduction/`.
