#!/usr/bin/env bash
# Bounded foreground batch: failures propagate; no detached processes.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset EVSIM_ROUTE
python3 verify.py
python3 test_regressions.py
python3 run_batch.py --steps "${STEPS:-40000}" --log "${LOG:-20000}" \
  --seeds ${SEEDS:-0} --jobs "${JOBS:-1}" --nstep "${NSTEP:-20}" \
  --tag "${TAG:-run}" --max-minutes "${MAX_MINUTES:-60}"
