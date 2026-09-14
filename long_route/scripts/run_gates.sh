#!/usr/bin/env bash
# Three gates of the frozen 20 km package: check_env (versions, machine rate), verify.py (six invariance checks),
# test_regressions.py (eight physics/ledger regression tests). Do not start training if any gate fails.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset EVSIM_ROUTE   # 必须未设置：否则 route20 会切换到 4 km mini 路线
mkdir -p "$ROOT/results"
cd "$ROOT/evsim_v9"
{
  echo "== 门禁 0：check_env.py（版本与本机速率）";        python3 check_env.py
  echo "== 门禁 1：verify.py（整形不变性/目标一致性/动作灵敏度/Markov 时钟）"; python3 verify.py
  echo "== 门禁 2：test_regressions.py（8 项物理与账本回归）"; python3 test_regressions.py
} 2>&1 | tee "$ROOT/results/gates.txt"
echo "All gates finished; record in results/gates.txt"
