#!/usr/bin/env bash
# v4b 门禁（渲染器修复 + 编码器 v4b 覆盖后）：等预训练结束 → 门 1 指标 + 阈值校准写入三份 v4b 配置 → 停止线回归测试 → 门 2 闭环探针 → 停（人工核对后再启动种子）。
export OMP_NUM_THREADS=1 VISUAL_CONFIG=v4b_cpu.json
cd "$(dirname "$0")/.."
while pgrep -f "[p]retrain.py --coverage v4b" >/dev/null; do sleep 20; done
[ -f runs/pretrain_v4b/encoder.pt ] || { echo "GATE1_FAILED: 无 encoder.pt"; exit 1; }
python3 - <<'PY' || { echo "GATE1_FAILED: 指标不达标"; exit 1; }
import json; r=json.load(open('runs/pretrain_v4b/pretrain_report.json')); v=r['dev_v4']
rec={c:v[c]['recall'] for c in ('traffic_light','curve_sign','end_marker','release_sign')}; fa={c:v[c]['false_alarm'] for c in rec}; tc=v['light_color']['teacher_roi_known_acc']; zh=r['dev_z_head_latest_frame']['known_acc']
ok=max(fa.values())<=0.02 and tc>=0.9 and zh>=0.9 and rec['curve_sign']>=0.7 and rec['traffic_light']>=0.6
print(f"[门 1] 召回 { {k:round(x,3) for k,x in rec.items()} } 误检 { {k:round(x,3) for k,x in fa.items()} } 灯色教师ROI {tc:.3f} Z头 {zh:.3f} 灯距离MAE {v['traffic_light']['dist_mae']:.1f} 训练 {r['training']['wall_s']:.0f}s -> {'通过' if ok else '不通过'}"); raise SystemExit(0 if ok else 1)
PY
python3 -u visual_dev/v4_calibrate_thresholds.py --encoder runs/pretrain_v4b/encoder.pt --fa-max 0.01 --out runs/pretrain_v4b/thresholds.json > runs/v4b_calibrate_stdout.txt 2>&1 || { echo "GATE1_FAILED: 校准异常"; exit 1; }
python3 - <<'PY'
import json; thr=json.load(open('runs/pretrain_v4b/thresholds.json'))['recommended']
for f in ('v4b_cpu.json','v4b_cpu_s1.json','v4b_cpu_s2.json'):
    c=json.load(open('configs/'+f)); c['det_thr']=thr; json.dump(c,open('configs/'+f,'w'),indent=1,ensure_ascii=False)
print('阈值写入三份 v4b 配置', thr)
PY
echo "[回归] 停止线灯箱可见性/近线检测/闭环（冻结臂用 v4 旧网络仅作参考，硬判据为渲染标签）"
python3 -u visual_dev/v4_test_stopline_visibility.py --config v4b_cpu.json --arm frozen --tag v4_pilot --must-arrive "" > runs/v4b_stopline_test_stdout.txt 2>&1; echo "回归测试退出码 $?"; grep -E "渲染|检测|闭环|总结" runs/v4b_stopline_test_stdout.txt | cut -c1-200
echo "[门 2] 闭环探针"; python3 -u visual_dev/v4_probe_closed_loop.py --config v4b_cpu.json --encoder runs/pretrain_v4b/encoder.pt --out runs/probes/v4b_probe_closed_loop.json > runs/v4b_probe2_stdout.txt 2>&1 || { echo "GATE2_FAILED: 探针异常"; exit 1; }
python3 - <<'PY' || { echo "GATE2_FAILED: 闭环未达标，见 runs/v4b_probe2_stdout.txt"; exit 1; }
import json; r=json.load(open('runs/probes/v4b_probe_closed_loop.json')); o=r['oracle']; v=r['vision']
ok=o['arrived']==9 and o['violations']==0 and v['arrived']>=8 and v['violations']==0 and v['offroad']<=5
print(f"[门 2] oracle 到达 {o['arrived']} 违规 {o['violations']} | 视觉 到达 {v['arrived']} 违规 {v['violations']} 弯道超速子步 {v['offroad']} 回退 {v['fallback']} | 消融 隐藏警示牌 到达 {r['vision_hide_curve_sign']['arrived']} 超速 {r['vision_hide_curve_sign']['offroad']} 灯色灭 到达 {r['vision_hide_light_color']['arrived']} 闯红灯 {r['vision_hide_light_color']['violations']} 隐藏终点 到达 {r['vision_hide_end_marker']['arrived']} -> {'通过' if ok else '不通过'}"); raise SystemExit(0 if ok else 1)
PY
echo "GATES_1_2_PASSED"
