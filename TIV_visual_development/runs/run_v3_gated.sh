#!/usr/bin/env bash
# v3 分级门禁：结构对齐版本（执行层感知相位、舒适性正则、去饱和 + 投影感知更新、冻结骨干联合臂）。
# 门 1 预训练 v3 覆盖（近距离/黄灯）：dev 上 0–5 m 绿灯召回 ≥0.9、红→绿 =0、20 m 以外已知类 ≥0.9（决定性判据在门 2）
# 门 2 感知-执行层闭环探针：v2 训练后的 F actor 在感知执行层下 9 工况 0 违规、到达 9/9、感知绿/真值红 ≤2 子步
# 门 3 审计（含 v3 新增项）
# 门 4 共同适配 + joint_head/joint 各 600 子步烟测
# 门 5 四臂 3000 子步小规模（STOP_AFTER_SMALL=1 时在此停止），达标后 9000
export OMP_NUM_THREADS=1 VISUAL_CONFIG=v3_cpu.json
cd "$(dirname "$0")/.."
TAG="${1:-v3_pilot}"; SMALL="${SMALL_SUBSTEPS:-3000}"
if [ "${SKIP_PRETRAIN:-0}" = "1" ] && [ -f runs/pretrain_v3/encoder.pt ]; then echo "[门 1] 复用已有 v3 预训练"; else
  rm -rf runs/pretrain_v3; echo "[门 1] 预训练 v3 覆盖"; python3 -u visual_dev/pretrain.py --coverage v3 --out pretrain_v3 --steps 2400 > runs/pretrain_v3_stdout.txt 2>&1 || { echo "GATE1_FAILED: 预训练异常"; exit 1; }
fi
python3 - <<'PY' || { echo "GATE1_FAILED: 指标不达标"; exit 1; }
import json; r=json.load(open('runs/pretrain_v3/pretrain_report.json')); b=r['dev_roi_head_by_distance']
near=b['0-5m']; far=[v['known_acc'] for k,v in b.items() if k not in ('0-5m','5-20m') and v['known_acc'] is not None]; r2g=[v['red_to_green'] for v in b.values() if v['red_to_green'] is not None]
ok=(near['green_recall'] or 0)>=0.9 and max(r2g)==0 and min(far)>=0.9 and r['dev_z_head_latest_frame']['known_acc']>=0.9
print(f"0–5 m 绿召回 {near['green_recall']} 黄→绿 {near['yellow_to_green']} (n_y={near['n_yellow']}) | 红→绿最大 {max(r2g)} | 20 m 外已知类最小 {min(far):.3f} | Z 头已知 {r['dev_z_head_latest_frame']['known_acc']:.3f} | 检测 ≤2px {r['dev_detection']['all']['hit_le2px']:.3f} -> {'通过' if ok else '不通过'}（决定性判据在门 2 闭环）"); raise SystemExit(0 if ok else 1)
PY
echo "[门 2] 感知-执行层闭环探针"; python3 -u visual_dev/v3_probe_perception.py --encoder runs/pretrain_v3/encoder.pt --actor runs/pilot_v2/frozen/final_nets.pt --green-threshold 0 --min-consecutive 1 --gate > runs/v3_probe2_stdout.txt 2>&1 || { echo "GATE2_FAILED: 感知执行层闭环未达标，见 runs/v3_probe2_stdout.txt"; exit 1; }
tail -3 runs/v3_probe2_stdout.txt
rm -rf runs/v3/common "runs/v3/$TAG" runs/v3/smoke runs/v3/audit
echo "[门 3] 审计"; python3 -u visual_dev/run_stage.py audit > runs/v3_audit_stdout.txt 2>&1 || { echo "GATE3_FAILED: 审计未通过，见 runs/v3_audit_stdout.txt"; exit 1; }
echo "[门 4] 共同适配"; python3 -u visual_dev/run_stage.py adapt > runs/v3_adapt_stdout.txt 2>&1 || { echo "GATE4_FAILED: 适配异常"; exit 1; }
for arm in joint_head joint; do
  echo "[门 4] $arm 600 子步烟测"; python3 -u visual_dev/run_stage.py train --arm $arm --tag smoke --substeps 600 --seconds 900 > "runs/v3_smoke_${arm}_stdout.txt" 2>&1 || { echo "GATE4_FAILED: $arm 烟测异常"; exit 1; }
  test -f "runs/v3/smoke/$arm/evaluation.json" || { echo "GATE4_FAILED: $arm 无评估输出"; exit 1; }
done
if [ "${STOP_BEFORE_ARMS:-0}" = "1" ]; then echo "GATES_1_4_PASSED"; exit 0; fi
echo "[门 5] 四臂 $SMALL 子步：$TAG"; bash runs/run_pilot_v3.sh "$TAG" "$SMALL" > "runs/v3_${TAG}_driver.txt" 2>&1; tail -1 "runs/v3_${TAG}_driver.txt"; echo "CHAIN_DONE"
