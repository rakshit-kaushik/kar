#!/usr/bin/env bash
# Routing batch job on a schedule that never overlaps a live-call window or a training run (plan T+14-16):
# every INTERVAL seconds, if the GPU lock is free and there are unlabeled tickets, run the job (it takes the lock as
# owner R for its duration and skips itself if the lock is taken meanwhile). Logs to $R3_ARTIFACTS/logs/routing.log.
#   MODEL_DIR=$R3_ARTIFACTS/models/r3-routing-<hash> nohup scripts/routing_scheduler.sh &     (INTERVAL default 600)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"; LOG="$ART/logs/routing.log"; mkdir -p "$ART/logs"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the routing model directory}"; INTERVAL="${INTERVAL:-600}"
echo "$(date '+%F %T') scheduler start: model $MODEL_DIR every ${INTERVAL}s" >> "$LOG"
while true; do
  if scripts/gpu_lock.sh status >/dev/null 2>&1; then
    n="$(uv run python -c "import os,psycopg; from dotenv import load_dotenv; load_dotenv('.env'); print(psycopg.connect(os.environ['DATABASE_URL']).execute('select count(*) from tickets where label is null').fetchone()[0])" 2>/dev/null || echo 0)"
    if [ "${n:-0}" -gt 0 ]; then
      echo "$(date '+%F %T') $n unlabeled tickets, lock free: running" >> "$LOG"
      uv run python -m services.routing.batch --model-dir "$MODEL_DIR" --limit 500 >> "$LOG" 2>&1
    fi
  fi
  sleep "$INTERVAL"
done
