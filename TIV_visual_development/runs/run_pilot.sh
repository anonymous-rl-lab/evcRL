#!/usr/bin/env bash
# 共同适配 → 三臂并行小规模（错开启动避免加载峰值）→ 等待退出
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
rm -rf runs/common runs/pilot; mkdir -p runs/pilot
python3 -u visual_dev/run_stage.py adapt > runs/adapt_stdout.txt 2>&1 || { echo "适配失败"; exit 1; }
for arm in frozen supervised joint; do
  nohup python3 -u visual_dev/run_stage.py train --arm $arm --tag pilot --substeps 9000 --seconds 3600 > runs/pilot/${arm}_stdout.txt 2>&1 &
  sleep 40
done
wait
echo "PILOT_DONE"
