#!/usr/bin/env bash
# restart.sh [--no-agent] [--no-server]
# Brings up, in order, verifying each: Postgres (must already run) -> model server :8000 -> proxy :8001
# -> prewarm request (Metal kernels + system-prompt KV cache) -> LiveKit agent worker (dev mode).
# Logs: $R3_ARTIFACTS/logs/{server,proxy,agent}.log. MODEL_DIR selects the served model directory
# (base by default; a fused adapter dir such as models/r3-voice-<hash> for a deploy).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
ART=${R3_ARTIFACTS:-$HOME/r3-artifacts}; LOG=$ART/logs; mkdir -p "$LOG"
MODEL_DIR=${MODEL_DIR:-$ART/models/Qwen3-4B-8bit}
SERVED=${LOCAL_LLM_MODEL:-r3-voice-base}
PSQL=${PSQL:-/opt/homebrew/opt/postgresql@15/bin/psql}
NO_AGENT=0; NO_SERVER=0
for a in "$@"; do case "$a" in --no-agent) NO_AGENT=1;; --no-server) NO_SERVER=1;; esac; done

wait_for() { # url seconds label
  for _ in $(seq 1 "$2"); do curl -sf -m 2 "$1" >/dev/null 2>&1 && { echo "  $3 ok"; return 0; }; sleep 1; done
  echo "  $3 FAILED to answer at $1 within $2 s"; return 1
}

echo "[1/5] postgres"
"$PSQL" "$DATABASE_URL" -tAc "select 1" >/dev/null && echo "  postgres ok ($DATABASE_URL)"

if [ "$NO_SERVER" = 0 ]; then
  echo "[2/5] model server :8000  ($MODEL_DIR as $SERVED)"
  lsof -ti:8000 | xargs kill 2>/dev/null || true; sleep 1
  nohup .venv/bin/vllm-mlx serve "$MODEL_DIR" --host 127.0.0.1 --port 8000 --served-model-name "$SERVED" \
    --prefix-trie-cache --default-chat-template-kwargs '{"enable_thinking": false}' \
    --default-temperature 0.2 --max-tokens 256 > "$LOG/server.log" 2>&1 &
  wait_for http://127.0.0.1:8000/v1/models 180 "server"
else
  echo "[2/5] model server: skipped (--no-server)"
fi

echo "[3/5] proxy :8001"
lsof -ti:8001 | xargs kill 2>/dev/null || true; sleep 1
nohup uv run python -m services.proxy.main > "$LOG/proxy.log" 2>&1 &
wait_for http://127.0.0.1:8001/health 60 "proxy"

echo "[4/5] prewarm"
if [ "$NO_SERVER" = 0 ]; then
  uv run python scripts/smoke_tools.py --base-url http://127.0.0.1:8001/v1 >/dev/null 2>&1 && echo "  prewarm ok (tool call parsed through proxy)" || { echo "  prewarm FAILED"; exit 1; }
else
  echo "  skipped"
fi

if [ "$NO_AGENT" = 0 ]; then
  echo "[5/5] agent worker (LLM -> $LLM_MODEL @ $LLM_BASE_URL)"
  pkill -f "services.agent.main" 2>/dev/null || true; sleep 1
  nohup uv run python -m services.agent.main dev > "$LOG/agent.log" 2>&1 &
  for _ in $(seq 1 90); do grep -qE "registered worker|worker registered|registered" "$LOG/agent.log" 2>/dev/null && break; sleep 1; done
  grep -qE "registered" "$LOG/agent.log" && echo "  agent ok" || { echo "  agent did not register; tail:"; tail -5 "$LOG/agent.log"; exit 1; }
else
  echo "[5/5] agent: skipped (--no-agent)"
fi
echo "up. logs in $LOG"
