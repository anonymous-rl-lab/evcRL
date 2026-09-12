#!/usr/bin/env bash
# 共同适配 → 三臂并行小规模（错开启动避免加载峰值）→ 等待退出并检查退出码。
# 用法：runs/run_pilot.sh <TAG> [SUBSTEPS]；已存在的 runs/common 或 runs/<TAG> 不会被删除（拒绝覆盖，需另取 TAG 或用 --resume）。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
TAG="${1:-pilot}"; SUBSTEPS="${2:-9000}"
if [ -d "runs/$TAG" ]; then echo "runs/$TAG 已存在，拒绝覆盖；换一个 TAG 或手动清理"; exit 2; fi
if [ ! -f runs/common/common.pt ]; then
  python3 -u visual_dev/run_stage.py adapt > runs/adapt_stdout.txt 2>&1 || { echo "适配失败，见 runs/adapt_stdout.txt"; exit 1; }
else echo "复用已有 runs/common/common.pt"; fi
mkdir -p "runs/$TAG"; declare -A PIDS
for arm in frozen supervised joint; do
  nohup python3 -u visual_dev/run_stage.py train --arm $arm --tag "$TAG" --substeps "$SUBSTEPS" --seconds 3600 > "runs/$TAG/${arm}_stdout.txt" 2>&1 &
  PIDS[$arm]=$!; sleep 40
done
FAIL=0
for arm in "${!PIDS[@]}"; do wait "${PIDS[$arm]}"; code=$?; echo "$arm 退出码 $code"; [ $code -ne 0 ] && FAIL=1; done
python3 visual_dev/run_stage.py report --tag "$TAG" > /dev/null && python3 visual_dev/summarize_pilot.py --tag "$TAG" > /dev/null
[ $FAIL -eq 0 ] && echo "PILOT_DONE" || echo "PILOT_FAILED"
