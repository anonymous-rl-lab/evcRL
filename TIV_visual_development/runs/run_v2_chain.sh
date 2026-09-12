#!/usr/bin/env bash
# v2 全链（顺序执行）：预训练 → 审计 → 适配 + 三臂（pilot_v2）→ 汇总
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/.."
rm -rf runs/pretrain runs/common runs/pilot_v2
python3 -u visual_dev/pretrain.py > runs/pretrain_stdout_v2.txt 2>&1 || { echo "CHAIN_FAILED: 预训练"; exit 1; }
python3 -u visual_dev/run_stage.py audit > runs/audit_stdout_v2.txt 2>&1 || { echo "CHAIN_FAILED: 审计未通过"; exit 1; }
bash runs/run_pilot.sh pilot_v2 9000 > runs/pilot_v2_driver.txt 2>&1
tail -1 runs/pilot_v2_driver.txt; echo "CHAIN_DONE"
