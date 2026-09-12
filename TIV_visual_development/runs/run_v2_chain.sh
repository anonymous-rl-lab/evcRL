#!/usr/bin/env bash
# v2 全链：等待预训练 → 审计 → 适配+三臂（pilot_v2）→ 汇总
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/.."
while pgrep -f "visual_dev/pretrain.py" >/dev/null; do sleep 15; done
[ -f runs/pretrain/encoder.pt ] || { echo "CHAIN_FAILED: 预训练未产出 encoder.pt"; exit 1; }
python3 -u visual_dev/run_stage.py audit > runs/audit_stdout_v2.txt 2>&1 || { echo "CHAIN_FAILED: 审计未通过，见 runs/audit_stdout_v2.txt"; exit 1; }
rm -rf runs/common
bash runs/run_pilot.sh pilot_v2 9000 > runs/pilot_v2_driver.txt 2>&1
tail -1 runs/pilot_v2_driver.txt; echo "CHAIN_DONE"
