# evsim v9 — verified benchmark batch

**What this package is.** The v8 tree (my Markov fix + the external expert's physics and
metric corrections + the FIFO replay buffer), with the launch path verified end-to-end and a
single entry point sized for a 12-core CPU box. **It contains no new algorithmic fix.** The
purpose of the run is to produce the seed distribution that the benchmark baseline table needs,
from a tree in which every known bug is fixed and every gate passes.

Read `WHAT IS AND IS NOT FIXED` below before interpreting the results.

---

## 1. Run it

```bash
tar xzf evsim_v9.tgz
cd evsim_v9
python3 check_env.py                 # NOT pip install -- see below
./run_12core.sh
```

**There is nothing to install.** The only third-party imports on the training and reporting
path are `numpy` and `torch`, which any AutoDL image already has — that is why no previous
package in this series needed an install step either. `check_env.py` reports the interpreter,
the two versions, the effective CPU quota, the memory and disk limits, and a measured
integration/update rate for *this* box. If it prints those, you are ready.

**Do not `pip install -r requirements.txt` on that box.** On a machine with a CUDA device,
plain `pip install torch` fetches the ~2.5 GB CUDA wheel and can move the torch version other
projects on the box depend on — for no benefit, since this code never touches CUDA (zero
`.cuda()` / `.to(device)` calls anywhere in the tree). Install only if `check_env.py` actually
fails on a missing import, and then pin the CPU index:

```bash
pip install "numpy>=1.26" "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
```

Version floor, tested rather than assumed: `numpy 1.26.4 + torch 2.2.2+cpu` and
`numpy 2.4.4 + torch 2.13.0+cpu` both pass all gates and produce a **bit-identical** 12k-step
training curve (seed 0: R = −387.723, last-3 = −394.262). So an older AutoDL image needs no
upgrade. `matplotlib` is used by `figs.py` only and is not needed to run or report.

That is the whole thing. Defaults: 8 seeds × 5,000,000 integration steps, 8 parallel workers,
one BLAS thread each, checkpoint every 250,000 steps (20 checkpoints per seed).

**Do not give it the GPU.** The actor and both critics are 64-unit MLPs; the environment
integration is the bottleneck. CUDA makes this slower. The script pins
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` so eight workers do not fight over BLAS.

Overrides, if wanted:

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9" JOBS=10 STEPS=5000000 TAG=bench2 ./run_12core.sh
```

**Expected wall-clock.** Measured 875 integration steps/s per worker on a 2-core reference box
(pure training; per-checkpoint evaluation adds ~10 s each, ~3% of the run). 5M steps ≈ 100
min/seed. With 8 workers in parallel on 12 cores, budget **2–3 hours** for the batch. The
`--max-minutes` cap defaults to 360.

**Interruption is safe.** The curve JSON (`out/<tag>_s<seed>.json`) and a `.pt` checkpoint are
written at *every* log point, atomically. If the batch is killed, nothing already computed is
lost — just run `python3 analyze.py`.

## 2. Reporting

`run_batch.py` calls `analyze.py` on success. To re-report at any time:

```bash
python3 analyze.py --json results.json
```

Send me `results.json` plus `out/*.json` (the `.pt` files are not needed; they are ~70 KB each,
160 of them).

Selection protocol, enforced in code: checkpoints are selected on the **validation** offsets
{0.0°, 45.0°} and scored on the **disjoint reporting** offsets {22.5°, 67.5°}. Reporting data
never enters selection. Each split is six unique (SoC, T, offset) conditions.

Baselines are matched **per condition** to the reporting rows — this is the bug the expert
caught and I confirmed: earlier tables compared a reporting-half learner score against
full-grid driver values. The correct reporting-half references are

| | R |
|---|---|
| DP (optimum) | −168.176 |
| attentive | **−177.473** |
| normal | −179.423 |
| distracted | −181.006 |

Beating `attentive` requires **6/6 arrivals AND** R > −177.473.

## 3. Three gates before a single training step

`run_12core.sh` will not reach `run_batch.py` unless all three pass.

1. **`verify.py`** — 6 checks. Shaping invariance (total potential-based shaping must be the
   policy-independent constant c\*L = 168.800; the absence of this check is what let the v4
   package ship broken); shaping constant read from the plant (c = 0.00844 = argmin g(v)/v at
   v\* = 19.98 m/s, not a tuned value); objective consistency (step rewards sum to the reported
   R_L to 1e-6); action sensitivity A1 (26.7% dead states, must be < 50%); the two
   arrival-threshold cruises; and **the Markov check** — two states with identical (x, v, t) but
   different episode deadlines must produce different observations.
2. **`test_regressions.py`** — 8 unit tests over the physics and the reward ledger, including
   the low-speed traction fix and signal-checked-at-crossing-time.
3. **Reference fingerprint** — `td3_run.py` refuses to start if the SHA-256 of
   `env20.py + models.py + plant.py + effcal.py + route20.py + scn/<net>_profile.npy` differs
   from the hash stored in `frozen/refs_fixed.json`. The learner can therefore never be scored
   against stale references. **If you edit the environment, re-run `refresh_refs.py` first.**

`verify.py` also prints, as INFO not as a check, the structural fact that governs this task:
the episode cut-off is `t_end = t + 1.25·(L−x)/pace`, so **arrival requires an average of
15.24 m/s from any start**. The criterion is scale-free — an exploring start near the finish is
no easier than a cold start — so a policy that settles below 15.24 m/s completes nothing and its
replay buffer holds no evidence of what a finished trip is worth.

## 4. WHAT IS AND IS NOT FIXED

**Fixed, verified, and measured:**

- **The Markov violation.** The observation carried `(T_budget − t)/T_budget` with a fixed
  budget while episodes actually end at `t_end`, which for an exploring start is
  `t₀ + 1.25(L−x₀)/pace`. Two trajectories with deadlines 25 s apart produced byte-identical
  observations (max|diff| = 0.00e+00). `t_end` spans 927–1644 s in training; 57% of episodes
  differ from evaluation's 1312.5 s by more than 60 s, and an unfinished trip costs up to −300 —
  so one input carried wildly different truths and the critic could only regress to their
  average. Irreducible bias, not noise. Effect of the fix: arrival collapse on seed 1 went from
  19/20 checkpoints to 3/11; Q(s₀) bias halved, −73…−97 → −44…−54. Pinned by `verify.py` check 6.
- **The TD3 update math was never broken.** Unit-tested on a 657-decision trajectory: 657 rows
  written, 657 uniquely matched, 0 mismatches on the n-step return, the bootstrap state, and the
  done flag; the target formula matched hand computation to 0.00e+00.
- **Low-speed traction** (expert's find, confirmed): the 2 m/s regen cutoff was being applied to
  traction, so v = 1.5 m/s at a = 2.6 m/s² charged 194 W against 7730 W at the wheel. Baseline
  impact 0.058 units.
- **Baseline scale** (expert's find, confirmed): see §2.

**NOT fixed:**

- **Residual critic bias, −45 to −58.** Retrace(λ) confirmed off-policy contamination is a real
  second cause — it cut the bias 91.6 → 58.6 — but shortening the trace shortens the effective
  horizon, and this task needs a long one, so the policy got *worse* (−179.54 → −189.16 on the
  reporting half). Evidence in `prior_runs/RT_off_s0.json` and `prior_runs/RT_on_s0.json`.
- **Peak-then-degrade.** Every configuration peaks and then degrades. Peak location varies
  250k–5M steps across seeds. This is why 5M steps with 20 checkpoints and validation-based
  selection is the protocol rather than "train to convergence".
- **The post-fix peak is *lower* than the best pre-fix run** (−177.5 vs −172.1), which says the
  Markov violation was accidentally acting as a regularizer and the binding cause of the
  performance ceiling has not been found.

Honest summary of learner quality with this protocol: 7/16 checkpoints beat `attentive`, 11/16
beat `normal`, best −172.13 with 12/12 completed. Usable as a benchmark baseline; the
instability must be reported as a measured phenomenon, not hidden.

## 5. Directions already tried — do not re-run

| Direction | Result |
|---|---|
| Markov fix | ✅ collapse 19/20 → 3/11, bias −90 → −45 |
| Expert's physics / metric fixes | ✅ correctness |
| FIFO buffer (vs full-buffer) | ✅ +11–15 units |
| Progress-potential shaping (both versions) | ❌ |
| PRIME scripted warm-up | ❌ fixes the early trap, degrades to −187/−194 by 5M |
| cost-to-go 15 → 9 | ❌ arrival → 0.00 |
| γ 1 → 0.995 | ❌ R −391 |
| Relax training cut-off to 12 m/s | ❌ trades one trap for another |
| Remove the 1050 s deadline (bootstrap truncation) | ❌ 200–340 units worse |
| Retrace(λ), K = 10, σ_c = 1.5 | ❌ REP −189.16 vs −179.54 control |

The common pattern: **γ = 1 and the cost-to-go cliff are what make this task well-posed.**
Anything that shortens the effective horizon, or makes quitting or over-speeding cheaper, breaks
it. `prior_runs/` holds the curve JSON for the informative ones —
`python3 analyze.py --input prior_runs --include-smoke` reproduces that table.

`td3_retrace.py` is kept as the documented negative-result ablation. It is **not** in the main
batch and should not be.

## 6. File map

| File | Role |
|---|---|
| `run_12core.sh` | the only command you need |
| `run_batch.py` | bounded parallel runner; no detached processes; failures propagate |
| `td3_run.py` | the learner (default: n-step 20, FIFO 125k, γ = 1, OU noise, reward mode L) |
| `env20.py` | environment; the Markov fix is at the clock channel of `obs()` |
| `plant.py`, `models.py`, `effcal.py` | vehicle and battery physics |
| `route20.py`, `scn/` | 20 km arterial route and signal profile |
| `dp20L.py`, `dp20I.py` | dynamic-programming optimum and information-limited DP |
| `refresh_refs.py` | regenerates `frozen/refs_fixed.json` — **required after any env edit** |
| `verify.py`, `test_regressions.py` | the two gates |
| `analyze.py` | validation-selected, condition-matched reporting |
| `td3_retrace.py` | negative-result ablation, not part of the batch |
| `prior_runs/` | curve JSON for the ablations in §5 |
| `DEBUG_REPORT_zh.md`, `README_v8_history.md` | the v8 debugging record |
