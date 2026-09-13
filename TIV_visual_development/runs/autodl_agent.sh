#!/usr/bin/env bash
# 以 git 为控制通道的执行代理：在租用服务器上常驻运行，每 60 s 拉取 main，发现 runs/jobs/*.json 中未完成的任务就执行，
# 日志与结果推到分支 autodl-results。控制方只需提交任务文件；本机需要能 push 的 git 凭据（例如 git config credential.helper store 后先手动 push 一次）。
# 任务文件格式：{"name":"seed1","cmd":"bash runs/autodl_run.sh 1 4 9000","cwd":"TIV_visual_development"}
# 用法：bash runs/autodl_agent.sh   （建议 nohup … & 或放进 tmux）
set -u
cd "$(git rev-parse --show-toplevel)"; git config user.name >/dev/null || git config user.name autodl-agent; git config user.email >/dev/null || git config user.email autodl@local
HOST=$(hostname); NPROC=$(nproc); MEM=$(free -g | awk '/Mem/{print $2}')
git checkout -q -B autodl-results 2>/dev/null || true
while true; do
  git fetch -q origin main && git merge -q --no-edit origin/main 2>/dev/null || true
  mkdir -p TIV_visual_development/runs/jobs/results
  echo "{\"host\":\"$HOST\",\"nproc\":$NPROC,\"mem_gb\":$MEM,\"time\":\"$(date -u +%FT%TZ)\"}" > TIV_visual_development/runs/jobs/results/heartbeat.json
  for job in TIV_visual_development/runs/jobs/*.json; do
    [ -f "$job" ] || continue; name=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$job")
    done_file="TIV_visual_development/runs/jobs/results/$name.done.json"; [ -f "$done_file" ] && continue
    cmd=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cmd'])" "$job"); cwd=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('cwd','.'))" "$job")
    echo "[$(date -u +%T)] 执行任务 $name: $cmd"; start=$(date -u +%FT%TZ)
    ( cd "$cwd" && bash -c "$cmd" ) > "TIV_visual_development/runs/jobs/results/$name.log" 2>&1; code=$?
    echo "{\"name\":\"$name\",\"exit\":$code,\"start\":\"$start\",\"end\":\"$(date -u +%FT%TZ)\"}" > "$done_file"
    git add -A && git commit -qm "autodl 任务 $name 完成，退出码 $code" && git push -q -u origin autodl-results || echo "推送失败"
  done
  git add -A TIV_visual_development/runs/jobs/results 2>/dev/null && git commit -qm "autodl 心跳" 2>/dev/null && git push -q -u origin autodl-results 2>/dev/null || true
  sleep 60
done
