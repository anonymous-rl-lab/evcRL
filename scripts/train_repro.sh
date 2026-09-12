#!/usr/bin/env bash
# 按冻结协议（论文 VI-A 与附录 H）重新训练长程环境感知 TD3。
# 默认：种子 0 1 2，各 500 万积分步，3 个 CPU 进程，每 25 万步一个检查点（共 20 个）。4 核约 3 小时。
# 用法：SEEDS="0 1 2" STEPS=5000000 JOBS=3 TAG=repro scripts/train_repro.sh
# 注意：run_batch.py 拒绝覆盖已存在的 TAG（out/<TAG>_s<seed>.json 已存在时报错），换一个 TAG 即可。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset EVSIM_ROUTE
cd "$ROOT/evsim_v9"
echo "先跑门禁 1/2 ..."; python3 verify.py >/dev/null && python3 test_regressions.py 2>/dev/null
echo "开始训练：seeds=${SEEDS:-0 1 2} steps=${STEPS:-5000000} jobs=${JOBS:-3} tag=${TAG:-repro}"
python3 -u run_batch.py --seeds ${SEEDS:-0 1 2} --jobs "${JOBS:-3}" --steps "${STEPS:-5000000}" \
  --log "${LOG:-250000}" --nstep 20 --buf 125000 --tag "${TAG:-repro}" --max-minutes "${MAX_MINUTES:-900}"
mkdir -p "$ROOT/results/train"
cp out/${TAG:-repro}_s*.json "$ROOT/results/train/"
python3 analyze.py --json "$ROOT/results/train/analyze_${TAG:-repro}.json"
echo "训练曲线已复制到 results/train/，随后可运行 python3 scripts/compare_curves.py 与归档八种子逐检查点比较"
