#!/usr/bin/env bash
# restart.sh [--no-agent] [--no-server] [--agent-only] [--server-only]   (--agent-only restarts the LiveKit worker alone: no server, no proxy, no lock)
# Brings up, in order, verifying each: Postgres (must already run) -> model server :8000 -> proxy :8001
# -> prewarm request (Metal kernels + system-prompt KV cache) -> LiveKit agent worker (dev mode).
# Logs: $R3_ARTIFACTS/logs/{server,proxy,agent}.log. MODEL_DIR selects the served model directory
# (base by default; a fused adapter dir such as models/r3-voice-<hash> for a deploy).
set -euo pipefail
cd "$(dirname "$0")/.."
# values given on the command line win over .env (MODEL_DIR=... LLM_MODEL=... LOCAL_LLM_MODEL=... scripts/restart.sh)
_OVR_LLM_MODEL="${LLM_MODEL:-}"; _OVR_LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-}"; _OVR_LLM_BASE_URL="${LLM_BASE_URL:-}"
set -a; source .env; set +a
[ -n "$_OVR_LLM_MODEL" ] && export LLM_MODEL="$_OVR_LLM_MODEL"
[ -n "$_OVR_LOCAL_LLM_MODEL" ] && export LOCAL_LLM_MODEL="$_OVR_LOCAL_LLM_MODEL"
[ -n "$_OVR_LLM_BASE_URL" ] && export LLM_BASE_URL="$_OVR_LLM_BASE_URL"
ART=${R3_ARTIFACTS:-$HOME/r3-artifacts}; LOG=$ART/logs; mkdir -p "$LOG"
MODEL_DIR=${MODEL_DIR:-$ART/models/Qwen3-4B-8bit}
SERVED=${LOCAL_LLM_MODEL:-r3-voice-base}
PSQL=${PSQL:-/opt/homebrew/opt/postgresql@15/bin/psql}
NO_AGENT=0; NO_SERVER=0; NO_PROXY=0
for a in "$@"; do case "$a" in --no-agent) NO_AGENT=1;; --no-server) NO_SERVER=1;; --agent-only) NO_SERVER=1; NO_PROXY=1;; --server-only) NO_PROXY=1; NO_AGENT=1;; --proxy-only) NO_SERVER=1; NO_AGENT=1;; esac; done

wait_for() { # url seconds label
  for _ in $(seq 1 "$2"); do curl -sf -m 2 "$1" >/dev/null 2>&1 && { echo "  $3 ok"; return 0; }; sleep 1; done
  echo "  $3 FAILED to answer at $1 within $2 s"; return 1
}

# --- shared-stack guard: never restart under a session that holds the GPU/server lock (scripts/gpu_lock.sh)
OWNER="${R3_OWNER:-S}"
if [ "$NO_SERVER" = 1 ] && { [ "$NO_PROXY" = 1 ] || [ "$NO_AGENT" = 1 ]; } && ! { [ "$NO_PROXY" = 1 ] && [ "$NO_AGENT" = 1 ]; }; then
  LOCK_WAS_MINE=1  # agent-only or proxy-only: the GPU/server is not touched
elif ! "$(dirname "$0")/gpu_lock.sh" status >/dev/null 2>&1; then
  holder="$("$(dirname "$0")/gpu_lock.sh" status 2>/dev/null || true)"
  if ! echo "$holder" | grep -q "held by $OWNER "; then
    if [ "${FORCE:-0}" != "1" ]; then echo "restart refused: $holder (set R3_OWNER to your lock owner, or FORCE=1 as the human)"; exit 3; fi
    echo "FORCE=1: restarting despite lock: $holder"
  fi
  LOCK_WAS_MINE=1
else
  "$(dirname "$0")/gpu_lock.sh" acquire "$OWNER" 10 "restart" >/dev/null; LOCK_WAS_MINE=0
fi
trap '[ "${LOCK_WAS_MINE:-1}" = 0 ] && "$(dirname "$0")/gpu_lock.sh" release "$OWNER" >/dev/null 2>&1 || true' EXIT

echo "[1/5] postgres"
"$PSQL" "$DATABASE_URL" -tAc "select 1" >/dev/null && echo "  postgres ok ($DATABASE_URL)"

if [ "$NO_SERVER" = 0 ]; then
  echo "[2/5] model server :8000  ($MODEL_DIR as $SERVED)"
  lsof -ti:8000 -sTCP:LISTEN | xargs kill 2>/dev/null || true; sleep 1
  # scripts/vllm_mlx_serve.py = vllm-mlx CLI + services/server/trie_keyfix.py (vllm-mlx 0.5.0 keys the prompt trie by the
  # unhashable MLX module: without the fix the trie never hits and every turn prefills the whole prompt). Trie capped at
  # 8 entries / 2 GB: one entry is the KV of a whole call (~150 KB per token on Qwen3-4B, ~300 MB at 2K tokens).
  VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=wait nohup .venv/bin/python scripts/vllm_mlx_serve.py serve "$MODEL_DIR" --host 127.0.0.1 --port 8000 --served-model-name "$SERVED" \
    --prefix-trie-cache --prefix-trie-cache-size 8 --prefix-trie-cache-memory-mb 2048 --default-chat-template-kwargs '{"enable_thinking": false}' \
    --default-temperature 0.2 --max-tokens 256 > "$LOG/server.log" 2>&1 &
  wait_for http://127.0.0.1:8000/v1/models 180 "server"
else
  echo "[2/5] model server: skipped (--no-server)"
fi

if [ "$NO_PROXY" = 0 ]; then
echo "[3/5] proxy :8001"
lsof -ti:8001 -sTCP:LISTEN | xargs kill 2>/dev/null || true; sleep 1
nohup uv run python -m services.proxy.main > "$LOG/proxy.log" 2>&1 &
wait_for http://127.0.0.1:8001/health 60 "proxy"
else
  echo "[3/5] proxy: skipped (--agent-only / --server-only)"
fi

echo "[4/5] prewarm (always: a changed prompt or tool set means a cold system cache, and the first cold request exceeds the proxy's fallback timeout)"
if curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null; then
  # straight at the server: the first request after start takes ~19 s (engine warmup) and would otherwise trip the proxy's hosted fallback
  uv run python scripts/smoke_tools.py --base-url http://127.0.0.1:8000/v1 >/dev/null 2>&1 || true
  echo "  server warm, prefix cached"
  if [ "$NO_PROXY" = 0 ] || curl -s -m 2 http://127.0.0.1:8001/health >/dev/null; then
    uv run python scripts/smoke_tools.py --base-url http://127.0.0.1:8001/v1 >/dev/null 2>&1 && echo "  proxy ok (tool call parsed through proxy, served locally)" || { echo "  proxy check FAILED"; exit 1; }
  fi
else
  echo "  skipped (server not up)"
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
