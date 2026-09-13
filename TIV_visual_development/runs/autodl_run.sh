#!/usr/bin/env bash
# 在租用服务器（如 AutoDL）上一键复现 v4 四臂实验的一个种子。CPU 即可（渲染与仿真是瓶颈，网络很小）；
# 内存要求：每臂进程 4–6 GB，MAX_PARALLEL 按 (内存 GB − 2) / 6 取整；磁盘每臂断点 1–2 GB × 2。
# 用法：bash runs/autodl_run.sh <seed: 0|1|2> [MAX_PARALLEL] [SUBSTEPS]
#   例：bash runs/autodl_run.sh 1 4 9000
# 结果推送到分支 v4-seed<seed>（需要能 push 的 git 凭据；没有的话把 runs/v4_s<seed>/v4_pilot 打包发回即可）。
set -euo pipefail
SEED="${1:?seed 0/1/2}"; MAXP="${2:-2}"; SUB="${3:-9000}"
cd "$(dirname "$0")/.."
python3 - <<'PY' || pip install --quiet "torch>=2.2" numpy pillow matplotlib
import torch, numpy, PIL, matplotlib
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ "$SEED" = "0" ]; then CFG=v4_cpu.json; else CFG=v4_cpu_s$SEED.json; fi
export VISUAL_CONFIG=$CFG
echo "[1/4] 再生成预训练池（确定性）"; python3 visual_dev/regen_pool.py --out pretrain_v4 --coverage v4 --n-train 3200 --n-dev 320
echo "[2/4] 门 1–4（复用编码器：阈值校准、闭环探针、审计、共同适配、烟测）"; SKIP_PRETRAIN=1 STOP_BEFORE_ARMS=1 bash runs/run_v4_gated.sh v4_pilot | tee runs/v4s${SEED}_gated_stdout.txt
grep -q GATES_1_4_PASSED runs/v4s${SEED}_gated_stdout.txt || { echo "门禁未通过，停止"; exit 1; }
echo "[3/4] 四臂 $SUB 子步（并行 $MAXP）"; MAX_PARALLEL=$MAXP bash runs/run_pilot_v3.sh v4_pilot "$SUB" | tee runs/v4s${SEED}_pilot_driver.txt
RUNS_SUB=$(python3 -c "import json; print(json.load(open('configs/$CFG')).get('runs_subdir',''))")
python3 visual_dev/run_stage.py report --tag v4_pilot >/dev/null 2>&1 || true; python3 visual_dev/summarize_pilot.py --tag v4_pilot --runs "$RUNS_SUB" 2>/dev/null | tail -1 || true
echo "[4/4] 提交结果到分支 v4-seed$SEED"
cd "$(git rev-parse --show-toplevel)" && git checkout -B v4-seed$SEED && git add -A && git commit -qm "v4 种子 $SEED 四臂 $SUB 子步结果（AutoDL）" && git push -u origin v4-seed$SEED || echo "推送失败：请把 TIV_visual_development/runs/$RUNS_SUB/v4_pilot 打包发回"
