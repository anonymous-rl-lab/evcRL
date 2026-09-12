#!/usr/bin/env bash
# 12k 步 seed 0 烟测。冻结包 README 记录的位一致数值为 R = -387.723、last-3 = -394.262；
# 其中 last-3 只有在 --log 6000（两个记录点）时才能复现，因此这里固定用 --log 6000。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset EVSIM_ROUTE
OUT="$ROOT/results/smoke"; mkdir -p "$OUT"; TAG="smoke12k_log6000_$(date +%s)"
cd "$ROOT/evsim_v9"
python3 -u td3_run.py --seed 0 --steps 12000 --log 6000 --tag "$TAG" --output "$OUT" | tee "$OUT/$TAG.log"
python3 - "$OUT/${TAG}_s0.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); last = d['final']['last']['R']; l3 = d['final']['last3_mean_R']
ok = abs(last + 387.723) < 5e-4 and abs(l3 + 394.262) < 5e-4
print(f"末检查点 R {last:.6f}（期望 -387.723） last-3 {l3:.6f}（期望 -394.262） -> {'位一致复现通过' if ok else '不一致'}")
sys.exit(0 if ok else 1)
PY
