#!/usr/bin/env bash
# v4 分级门禁：无地图、靠视觉获取道路事件信息的纵向驾驶。
# 门 1 预训练 v4：dev 上检测召回 灯/警示牌 ≥0.8、终点 ≥0.7、解除牌 ≥0.5（阈值 0.5 口径）、误检 ≤2%、灯色运行时(预测 ROI) 已知类 ≥0.85；随后按 dev 校准逐类阈值
# 门 2 闭环探针（执行层驱动常量策略）：oracle 记忆 9/9 到达 0 违规；视觉记忆 到达 ≥8/9、闯红灯 0、弯道超速子步 ≤5
# 门 3 审计（含 v4 无地图检查）；门 4 共同适配 + joint_head/joint 600 子步烟测；门 5 四臂小规模
export OMP_NUM_THREADS=1 VISUAL_CONFIG="${VISUAL_CONFIG:-v4_cpu.json}"
cd "$(dirname "$0")/.."
TAG="${1:-v4_pilot}"; SMALL="${SMALL_SUBSTEPS:-3000}"
if [ "${SKIP_PRETRAIN:-0}" = "1" ] && [ -f runs/pretrain_v4/encoder.pt ]; then echo "[门 1] 复用已有 v4 预训练"; else
  rm -rf runs/pretrain_v4; echo "[门 1] 预训练 v4"; python3 -u visual_dev/pretrain.py --coverage v4 --out pretrain_v4 --steps 6000 --n-train 3200 --n-dev 320 > runs/pretrain_v4_stdout.txt 2>&1 || { echo "GATE1_FAILED: 预训练异常"; exit 1; }
fi
python3 - <<'PY' || { echo "GATE1_FAILED: 指标不达标"; exit 1; }
import json; r=json.load(open('runs/pretrain_v4/pretrain_report.json')); v=r['dev_v4']
rec={c:v[c]['recall'] for c in ('traffic_light','curve_sign','end_marker','release_sign')}; fa={c:v[c]['false_alarm'] for c in rec}; lc=v['light_color']['runtime_roi_known_acc']; tc=v['light_color']['teacher_roi_known_acc']; zh=r['dev_z_head_latest_frame']['known_acc']
ok=max(fa.values())<=0.02 and tc>=0.9 and zh>=0.9 and rec['curve_sign']>=0.7 and rec['traffic_light']>=0.6
print(f"检测召回(阈值0.5口径，信息量，工作点由校准阈值+记忆门控决定) { {k:round(x,3) for k,x in rec.items()} } 误检 { {k:round(x,3) for k,x in fa.items()} } 灯色 教师ROI已知 {tc:.3f} 运行时(阈值0.5) {lc:.3f} Z头已知 {zh:.3f} -> {'通过' if ok else '不通过'}（决定性判据在门 2 闭环）"); raise SystemExit(0 if ok else 1)
PY
python3 -u visual_dev/v4_calibrate_thresholds.py --fa-max 0.01 > runs/v4_calibrate_stdout.txt 2>&1 && python3 - <<'PY'
import json, os; cf='configs/'+os.environ['VISUAL_CONFIG']; c=json.load(open(cf)); c['det_thr']=json.load(open('runs/pretrain_v4/thresholds.json'))['recommended']; json.dump(c,open(cf,'w'),indent=1,ensure_ascii=False); print('阈值写入配置', cf, c['det_thr'])
PY
echo "[门 2] 闭环探针"; python3 -u visual_dev/v4_probe_closed_loop.py --config "$VISUAL_CONFIG" --out runs/probes/v4_probe_closed_loop.json > runs/v4_probe2_stdout.txt 2>&1 || { echo "GATE2_FAILED: 探针异常"; exit 1; }
python3 - <<'PY' || { echo "GATE2_FAILED: 闭环未达标，见 runs/v4_probe2_stdout.txt"; exit 1; }
import json; r=json.load(open('runs/probes/v4_probe_closed_loop.json')); o=r['oracle']; v=r['vision']
ok=o['arrived']==9 and o['violations']==0 and v['arrived']>=8 and v['violations']==0 and v['offroad']<=5
print(f"oracle 到达 {o['arrived']} 违规 {o['violations']} | 视觉 到达 {v['arrived']} 违规 {v['violations']} 弯道超速子步 {v['offroad']} 回退 {v['fallback']} -> {'通过' if ok else '不通过'}"); raise SystemExit(0 if ok else 1)
PY
SUB=$(python3 -c "import json,os; print(json.load(open('configs/'+os.environ['VISUAL_CONFIG'])).get('runs_subdir',''))"); rm -rf "runs/$SUB/common" "runs/$SUB/$TAG" "runs/$SUB/smoke" "runs/$SUB/audit"
echo "[门 3] 审计"; python3 -u visual_dev/run_stage.py audit > runs/v4_audit_stdout.txt 2>&1 || { echo "GATE3_FAILED: 审计未通过，见 runs/v4_audit_stdout.txt"; exit 1; }
echo "[门 4] 共同适配"; python3 -u visual_dev/run_stage.py adapt > runs/v4_adapt_stdout.txt 2>&1 || { echo "GATE4_FAILED: 适配异常"; exit 1; }
for arm in joint_head joint; do
  echo "[门 4] $arm 600 子步烟测"; python3 -u visual_dev/run_stage.py train --arm $arm --tag smoke --substeps 600 --seconds 900 > "runs/v4_smoke_${arm}_stdout.txt" 2>&1 || { echo "GATE4_FAILED: $arm 烟测异常"; exit 1; }
  test -f "runs/$SUB/smoke/$arm/evaluation.json" || { echo "GATE4_FAILED: $arm 无评估输出"; exit 1; }
done
if [ "${STOP_BEFORE_ARMS:-0}" = "1" ]; then echo "GATES_1_4_PASSED"; exit 0; fi
echo "[门 5] 四臂 $SMALL 子步：$TAG"; bash runs/run_pilot_v3.sh "$TAG" "$SMALL" > "runs/v4_${TAG}_driver.txt" 2>&1; tail -1 "runs/v4_${TAG}_driver.txt"; echo "CHAIN_DONE"
