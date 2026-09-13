#!/usr/bin/env bash
# 本机串行跑多个种子配置（云端子会话失败后的替代）。用法：runs/run_seeds_local.sh v4b_cpu.json v4b_cpu_s1.json v4b_cpu_s2.json：每个种子——共同适配 → 两臂一批并行（16 GB 机器上限）→ 批后对未完成臂逐臂串行 --resume（后期回放池满、存断点时单进程峰值 9 GB，两臂并行会被 OOM 杀）
# → 已完成臂删除 resume*.pt 腾磁盘 → report/summarize。启动前等待种子 0 重评估结束并删除其断点文件。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
TAG=v4_pilot; SUBSTEPS=9000; ARMS=(frozen supervised joint joint_head)
[ $# -ge 1 ] || { echo "用法：$0 <配置1.json> [配置2.json ...]"; exit 2; }
echo "开始 $(date) 磁盘剩余 $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GB"
for cfg in "$@"; do
  export VISUAL_CONFIG=$cfg; SUB=$(python3 -c "import json,os; print(json.load(open('configs/'+os.environ['VISUAL_CONFIG'])).get('runs_subdir',''))")
  echo "== 种子配置 $cfg → runs/$SUB $(date)"
  if [ ! -f "runs/$SUB/common/common.pt" ]; then python3 -u visual_dev/run_stage.py adapt > "runs/${SUB}_adapt_stdout.txt" 2>&1 || { echo "$cfg 适配失败"; continue; }; fi
  mkdir -p "runs/$SUB/$TAG"
  for batch in "frozen supervised" "joint joint_head"; do
    declare -A PIDS=()
    for arm in $batch; do
      [ -f "runs/$SUB/$TAG/$arm/evaluation.json" ] && continue
      RES=""; [ -f "runs/$SUB/$TAG/$arm/resume.pt" ] && RES="--resume"
      nohup python3 -u visual_dev/run_stage.py train --arm $arm --tag $TAG --substeps $SUBSTEPS --seconds 7200 $RES >> "runs/$SUB/$TAG/${arm}_stdout.txt" 2>&1 &
      PIDS[$arm]=$!; sleep 30
    done
    for arm in "${!PIDS[@]}"; do wait "${PIDS[$arm]}"; echo "$arm 批内退出码 $? $(date)"; done
    for arm in $batch; do   # 批后逐臂串行续跑被杀的臂
      while [ ! -f "runs/$SUB/$TAG/$arm/evaluation.json" ] && [ -f "runs/$SUB/$TAG/$arm/resume.pt" ]; do
        python3 -u visual_dev/run_stage.py train --arm $arm --tag $TAG --substeps $SUBSTEPS --seconds 7200 --resume >> "runs/$SUB/$TAG/${arm}_stdout.txt" 2>&1; code=$?; echo "$arm 续跑退出码 $code $(date)"
        [ $code -ne 0 ] && [ $code -ne 137 ] && break
      done
      [ -f "runs/$SUB/$TAG/$arm/evaluation.json" ] && rm -f "runs/$SUB/$TAG/$arm"/resume*.pt
    done
  done
  python3 visual_dev/run_stage.py report --tag $TAG > "runs/$SUB/$TAG/report_stdout.txt" 2>&1; python3 visual_dev/summarize_pilot.py --tag $TAG --runs "$SUB" > "runs/$SUB/$TAG/summary_stdout.txt" 2>&1
  echo "SEED_DONE $cfg $(date)"
done
echo "ALL_SEEDS_DONE $(date)"
