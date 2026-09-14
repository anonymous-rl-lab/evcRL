#!/usr/bin/env bash
# ============================================================================
#  Main benchmark batch -- 12-core CPU box.
#
#  The GPU is NOT used and must not be used: the actor and both critics are
#  64-unit MLPs and the environment is the bottleneck, so a CUDA device makes
#  this SLOWER.  Every worker is pinned to one BLAS thread; 8 workers on 12
#  cores leaves headroom for the OS and for the per-checkpoint evaluation.
#
#  Runtime measured at 875 integration steps/s per worker on a 2-core box.
#  5,000,000 steps  ->  about 100 min/seed.  Expect 2-3 h wall for the batch.
#
#  Nothing is detached.  If the batch is interrupted, no data is lost: the
#  curve JSON and a .pt checkpoint are written at EVERY log point, so
#      python3 analyze.py
#  reports whatever completed.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset EVSIM_ROUTE

# --- Gate 1: environment pre-flight (6 checks + the arrival-threshold report).
#     Includes the shaping-invariance check whose absence caused the v4 failure
#     and the Markov check that pins the deadline channel.  Exits 1 on any fail.
python3 verify.py

# --- Gate 2: 8 unit tests over the physics and reward ledger.
python3 test_regressions.py

# --- Gate 3: td3_run.py itself refuses to start if env20/models/plant/effcal/
#     route20/profile hash differs from the hash stored in frozen/refs_fixed.json.
#     That guarantees the learner is never scored against stale references.

python3 run_batch.py \
  --seeds ${SEEDS:-0 1 2 3 4 5 6 7} \
  --jobs "${JOBS:-8}" \
  --steps "${STEPS:-5000000}" \
  --log "${LOG:-250000}" \
  --nstep 20 \
  --buf 125000 \
  --tag "${TAG:-bench}" \
  --max-minutes "${MAX_MINUTES:-360}"

# run_batch.py calls analyze.py on success.  To re-report at any time:
#   python3 analyze.py --json results.json
