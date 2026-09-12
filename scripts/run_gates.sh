#!/usr/bin/env bash
# 冻结长程包的三道门禁（环境测速、verify.py 六项检查、8 项物理/账本回归测试），
# 以及视觉 Z 框架的 11 项接口契约测试。任何一项失败都不要开始训练。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset EVSIM_ROUTE   # 必须未设置：否则 route20 会切换到 4 km mini 路线
mkdir -p "$ROOT/results"
cd "$ROOT/evsim_v9"
{
  echo "== 门禁 0：check_env.py（版本与本机速率）";        python3 check_env.py
  echo "== 门禁 1：verify.py（整形不变性/目标一致性/动作灵敏度/Markov 时钟）"; python3 verify.py
  echo "== 门禁 2：test_regressions.py（8 项物理与账本回归）"; python3 test_regressions.py
} 2>&1 | tee "$ROOT/results/gates.txt"
cd "$ROOT/visual_z_framework"
echo "== 视觉 Z 框架契约测试（TIV_BASE=${TIV_BASE:-未设置；设置后额外核对 v19 源码哈希与真实 actor 权重迁移}）" | tee -a "$ROOT/results/gates.txt"
python3 -m unittest discover -s tests -v 2>&1 | tee -a "$ROOT/results/gates.txt"
echo "全部门禁执行完毕，记录见 results/gates.txt"
