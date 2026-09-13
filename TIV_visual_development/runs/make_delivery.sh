#!/usr/bin/env bash
# 完整交付包（审计整改）：分卷 ≤30 MiB。卷 1 = 代码/配置/报告/审计/编码器/12 组最终网络/共同适配网络/源码与权重哈希/评估 JSON/轨迹 npz；
# 卷 2.. = 逐步 layer JSON（每工况约 1–3 MB，分卷）。eval_sets.pt(146 MB)/pool.pt(1.3 GB)/common.pt(224 MB) 不入包：由脚本按固定种子再生成或为训练中间态，其 sha256 记入 MANIFEST。
set -e; cd "$(dirname "$0")/.."; OUT="${1:-/tmp/delivery}"; mkdir -p "$OUT"; rm -f "$OUT"/evcRL_v4r_*.zip "$OUT"/MANIFEST.txt
{
  echo "evcRL v4r 完整交付包 $(date -u +%Y-%m-%dT%H:%MZ) git $(git -C .. rev-parse HEAD)"; echo
  echo "## sha256（不入包的大文件，可按 pretrain.py --coverage v4 / regen_pool.py 固定种子再生成）"
  for f in runs/pretrain_v4/eval_sets.pt runs/pretrain_v4/pool.pt runs/v4r/common/common.pt runs/v4r_s1/common/common.pt runs/v4r_s2/common/common.pt; do [ -f $f ] && echo "$(sha256sum $f | cut -c1-64)  $f  $(stat -c %s $f) B"; done
  echo; echo "## sha256（入包）"
  for f in runs/pretrain_v4/encoder.pt runs/pretrain_v4/audit_set.npz $(ls runs/v4r*/v4_pilot/*/final_nets.pt runs/v4r*/common/common_nets.pt 2>/dev/null) $(ls visual_dev/*.py visual_z/*.py v19_deps/code/*.py v19_deps/weights/* 2>/dev/null); do echo "$(sha256sum $f | cut -c1-64)  $f"; done
} > "$OUT/MANIFEST.txt"
zip -q -r "$OUT/evcRL_v4r_vol1_core.zip" ../README.md README.md reports configs visual_dev visual_z v19_deps runs/run_v4_gated.sh runs/run_seeds_local.sh runs/run_pilot_v3.sh runs/make_delivery.sh \
  runs/pretrain_v4/encoder.pt runs/pretrain_v4/audit_set.npz runs/pretrain_v4/pretrain_report.json runs/pretrain_v4/thresholds.json \
  runs/v4/audit runs/probes/v4_probe_closed_loop.json runs/probes/v4r_probe_closed_loop.json runs/probes/v4b_probe_closed_loop.json runs/_failed \
  $(ls -d runs/v4r*/common/common.json runs/v4r*/common/common_nets.pt runs/v4r*/v4_pilot/*/final_nets.pt runs/v4r*/v4_pilot/*/evaluation*.json runs/v4r*/v4_pilot/*/status.json runs/v4r*/v4_pilot/*/evaluation_traces/*.npz runs/v4r*/v4_pilot/*_stdout.txt runs/v4r*/v4_pilot/report.json runs/v4r*/v4_pilot/summary*.* runs/v4r*/v4_pilot/diagnostics.png runs/v4r*_adapt_stdout.txt runs/v4/v4_pilot/*/evaluation*.json 2>/dev/null) "$OUT/MANIFEST.txt" -x "*__pycache__*" -x "*.pyc" -x "runs/_failed/*.pt" -x "runs/_failed/*/*.pt" -x "runs/_failed/*/*.npz"
# 卷 2..：layer JSON 分卷（zip -s 30m）
zip -q -r -s 30m "$OUT/evcRL_v4r_vol2_layer_logs.zip" $(ls runs/v4r*/v4_pilot/*/evaluation_traces/*_layer.json) 
ls -la "$OUT" | awk '{print $5/1048576 " MiB", $9}'
