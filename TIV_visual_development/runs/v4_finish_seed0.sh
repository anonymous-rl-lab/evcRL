#!/usr/bin/env bash
# 种子 0 收尾：等 joint 臂结束后，逐臂（串行，避免 OOM）续跑 frozen 与 joint_head，再出报告。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VISUAL_CONFIG=v4_cpu.json
cd "$(dirname "$0")/.."
while pgrep -f "[r]un_stage.py train --arm joint " >/dev/null; do sleep 20; done
echo "joint 已结束 $(date)"
for arm in frozen joint_head; do
  if [ -f runs/v4/v4_pilot/$arm/evaluation.json ]; then echo "$arm 已有评估，跳过"; continue; fi
  python3 -u visual_dev/run_stage.py train --arm $arm --tag v4_pilot --substeps 9000 --seconds 7200 --resume >> runs/v4/v4_pilot/${arm}_stdout.txt 2>&1
  echo "$arm 续跑退出码 $? $(date)"
done
python3 visual_dev/run_stage.py report --tag v4_pilot > runs/v4/v4_pilot/report_stdout.txt 2>&1
python3 visual_dev/summarize_pilot.py --tag v4_pilot --runs v4 > runs/v4/v4_pilot/summary_stdout.txt 2>&1
echo "SEED0_DONE $(date)"
