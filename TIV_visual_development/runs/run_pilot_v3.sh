#!/usr/bin/env bash
# v3 四臂并行（frozen / supervised / joint / joint_head），复用 runs/v3/common/common.pt；拒绝覆盖已有 TAG。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VISUAL_CONFIG="${VISUAL_CONFIG:-v3_cpu.json}"
cd "$(dirname "$0")/.."
TAG="${1:-v3_pilot}"; SUBSTEPS="${2:-3000}"
SUB=$(python3 -c "import json,os; print(json.load(open('configs/'+os.environ['VISUAL_CONFIG'])).get('runs_subdir',''))")
if [ -d "runs/$SUB/$TAG" ]; then echo "runs/$SUB/$TAG 已存在，拒绝覆盖"; exit 2; fi
[ -f "runs/$SUB/common/common.pt" ] || { python3 -u visual_dev/run_stage.py adapt > runs/v3_adapt_stdout.txt 2>&1 || { echo "适配失败"; exit 1; }; }
mkdir -p "runs/$SUB/$TAG"; declare -A PIDS
for arm in frozen supervised joint joint_head; do
  nohup python3 -u visual_dev/run_stage.py train --arm $arm --tag "$TAG" --substeps "$SUBSTEPS" --seconds 7200 > "runs/$SUB/$TAG/${arm}_stdout.txt" 2>&1 &
  PIDS[$arm]=$!; sleep 40
done
FAIL=0
for arm in "${!PIDS[@]}"; do wait "${PIDS[$arm]}"; code=$?; echo "$arm 退出码 $code"; [ $code -ne 0 ] && FAIL=1; done
python3 visual_dev/run_stage.py report --tag "$TAG" > /dev/null 2>&1; python3 visual_dev/summarize_pilot.py --tag "$TAG" --runs "$SUB" > /dev/null 2>&1
[ $FAIL -eq 0 ] && echo "PILOT_DONE" || echo "PILOT_FAILED"
