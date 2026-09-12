#!/usr/bin/env bash
# v2 分级门禁链：每级便宜且可阻断；只有全部通过才启动三臂长跑。
# 门 1 预训练（≈4 min）：检测命中≥0.6、ROI/Z 头已知类正确率≥0.9、标签可表示性断言
# 门 2 审计 19 项（≈2 min）
# 门 3 共同适配（≈4 min）+ 联合臂 600 子步烟测（≈3 min，验证分叉/评估/诊断路径）
# 门 4 三臂 9000 子步（≈40 min）
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/.."
TAG="${1:-pilot_v2}"
rm -rf runs/common "runs/$TAG" runs/smoke_v2
if [ "${SKIP_PRETRAIN:-0}" = "1" ] && [ -f runs/pretrain/encoder.pt ]; then echo "[门 1] 复用已有预训练结果"; else
  rm -rf runs/pretrain; echo "[门 1] 预训练"; python3 -u visual_dev/pretrain.py > runs/pretrain_stdout_v2.txt 2>&1 || { echo "GATE1_FAILED: 预训练异常"; exit 1; }
fi
python3 - <<'PY' || { echo "GATE1_FAILED: 指标不达标"; exit 1; }
import json; r=json.load(open('runs/pretrain/pretrain_report.json'))
det=r['dev_detection']['all']; roi=r['dev_signal_metrics']['all']['roi_head']['known_acc']; z=r['dev_z_head_latest_frame']['known_acc']
rep=r['label_representability']; ok=det['hit_le2px']>=0.6 and det['cell_ok_rate']>=0.6 and roi>=0.9 and z>=0.9 and all(v['negative_components']==0 and v['over_one_components']==0 for v in rep.values())
print(f"检测 ≤2px命中 {det['hit_le2px']:.3f} 格一致 {det['cell_ok_rate']:.3f} 中心误差 {det['mean_center_err_px']:.2f}px 误检率 {r['dev_detection']['no_visible_light_frames']['false_alarm_rate']} | ROI已知 {roi:.3f} Z已知 {z:.3f} | 可表示性 {rep['dev']} -> {'通过' if ok else '不通过'}"); raise SystemExit(0 if ok else 1)
PY
echo "[门 2] 审计"; python3 -u visual_dev/run_stage.py audit > runs/audit_stdout_v2.txt 2>&1 || { echo "GATE2_FAILED: 审计未通过"; exit 1; }
echo "[门 3] 共同适配"; python3 -u visual_dev/run_stage.py adapt > runs/adapt_stdout.txt 2>&1 || { echo "GATE3_FAILED: 适配异常"; exit 1; }
echo "[门 3] 联合臂 600 子步烟测（含评估与诊断）"; python3 -u visual_dev/run_stage.py train --arm joint --tag smoke_v2 --substeps 600 --seconds 900 > runs/smoke_v2_stdout.txt 2>&1 || { echo "GATE3_FAILED: 烟测异常，见 runs/smoke_v2_stdout.txt"; exit 1; }
test -f runs/smoke_v2/joint/evaluation.json || { echo "GATE3_FAILED: 烟测无评估输出"; exit 1; }
if [ "${STOP_BEFORE_ARMS:-0}" = "1" ]; then echo "GATES_1_3_PASSED（按要求在三臂长跑前停止）"; exit 0; fi
echo "[门 4] 三臂 $TAG"; bash runs/run_pilot.sh "$TAG" 9000 > "runs/${TAG}_driver.txt" 2>&1
tail -1 "runs/${TAG}_driver.txt"; echo "CHAIN_DONE"
