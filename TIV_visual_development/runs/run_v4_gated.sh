#!/usr/bin/env bash
# v4 分级门禁：无地图、靠视觉获取道路事件信息的纵向驾驶。
# 门 1 预训练 v4：dev 上四类目标检测召回（存在头门控 ≥0.5）≥0.85、误检 ≤2%、灯色运行时(预测 ROI) 已知类 ≥0.9
# 门 2 闭环探针（执行层驱动常量策略）：oracle 记忆 9/9 到达 0 违规；视觉记忆 到达 ≥8/9、闯红灯 0、弯道超速子步 ≤5
# 门 3 审计（含 v4 无地图检查）；门 4 共同适配 + joint_head/joint 600 子步烟测；门 5 四臂小规模
export OMP_NUM_THREADS=1 VISUAL_CONFIG=v4_cpu.json
cd "$(dirname "$0")/.."
TAG="${1:-v4_pilot}"; SMALL="${SMALL_SUBSTEPS:-3000}"
if [ "${SKIP_PRETRAIN:-0}" = "1" ] && [ -f runs/pretrain_v4/encoder.pt ]; then echo "[门 1] 复用已有 v4 预训练"; else
  rm -rf runs/pretrain_v4; echo "[门 1] 预训练 v4"; python3 -u visual_dev/pretrain.py --coverage v4 --out pretrain_v4 --steps 2400 > runs/pretrain_v4_stdout.txt 2>&1 || { echo "GATE1_FAILED: 预训练异常"; exit 1; }
fi
python3 - <<'PY' || { echo "GATE1_FAILED: 指标不达标"; exit 1; }
import json; r=json.load(open('runs/pretrain_v4/pretrain_report.json')); v=r['dev_v4']
rec={c:v[c]['recall'] for c in ('traffic_light','curve_sign','end_marker','release_sign')}; fa={c:v[c]['false_alarm'] for c in rec}; lc=v['light_color']['runtime_roi_known_acc']
ok=min(rec.values())>=0.85 and max(fa.values())<=0.02 and lc>=0.9
print(f"检测召回 { {k:round(x,3) for k,x in rec.items()} } 误检 { {k:round(x,3) for k,x in fa.items()} } 运行时灯色已知 {lc:.3f} -> {'通过' if ok else '不通过'}"); raise SystemExit(0 if ok else 1)
PY
echo "[门 2] 闭环探针"; python3 -u visual_dev/v4_probe_closed_loop.py --out runs/probes/v4_probe_closed_loop.json > runs/v4_probe2_stdout.txt 2>&1 || { echo "GATE2_FAILED: 探针异常"; exit 1; }
python3 - <<'PY' || { echo "GATE2_FAILED: 闭环未达标，见 runs/v4_probe2_stdout.txt"; exit 1; }
import json; r=json.load(open('runs/probes/v4_probe_closed_loop.json')); o=r['oracle']; v=r['vision']
ok=o['arrived']==9 and o['violations']==0 and v['arrived']>=8 and v['violations']==0 and v['offroad']<=5
print(f"oracle 到达 {o['arrived']} 违规 {o['violations']} | 视觉 到达 {v['arrived']} 违规 {v['violations']} 弯道超速子步 {v['offroad']} 回退 {v['fallback']} -> {'通过' if ok else '不通过'}"); raise SystemExit(0 if ok else 1)
PY
rm -rf runs/v4/common "runs/v4/$TAG" runs/v4/smoke runs/v4/audit
echo "[门 3] 审计"; python3 -u visual_dev/run_stage.py audit > runs/v4_audit_stdout.txt 2>&1 || { echo "GATE3_FAILED: 审计未通过，见 runs/v4_audit_stdout.txt"; exit 1; }
echo "[门 4] 共同适配"; python3 -u visual_dev/run_stage.py adapt > runs/v4_adapt_stdout.txt 2>&1 || { echo "GATE4_FAILED: 适配异常"; exit 1; }
for arm in joint_head joint; do
  echo "[门 4] $arm 600 子步烟测"; python3 -u visual_dev/run_stage.py train --arm $arm --tag smoke --substeps 600 --seconds 900 > "runs/v4_smoke_${arm}_stdout.txt" 2>&1 || { echo "GATE4_FAILED: $arm 烟测异常"; exit 1; }
  test -f "runs/v4/smoke/$arm/evaluation.json" || { echo "GATE4_FAILED: $arm 无评估输出"; exit 1; }
done
if [ "${STOP_BEFORE_ARMS:-0}" = "1" ]; then echo "GATES_1_4_PASSED"; exit 0; fi
echo "[门 5] 四臂 $SMALL 子步：$TAG"; VISUAL_CONFIG=v4_cpu.json bash runs/run_pilot_v3.sh "$TAG" "$SMALL" > "runs/v4_${TAG}_driver.txt" 2>&1; tail -1 "runs/v4_${TAG}_driver.txt"; echo "CHAIN_DONE"
