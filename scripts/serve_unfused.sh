#!/usr/bin/env bash
# Gate-fail fallback (plan: "if the fused artifact fails the precision gate, serve the unfused adapter on the 8-bit base and
# accept the slower tokens"). Replaces the vllm-mlx server on :8000 with mlx_lm.server (the exact configuration the gate run
# used) and restarts proxy + agent with the label <id>-unfused. Hold lock S first. mlx_lm.server answers only to the model
# name `default_model` and would try to download any other name as a HF repo, so the proxy's upstream name is pinned; it is
# one request at a time with no prefix cache: expect base-like TTFT (~0.7-1.0 s on tool-call turns).
# Usage: scripts/serve_unfused.sh $R3_ARTIFACTS/models/r3-voice-<hash>   (uses <dir>/adapter and $R3_ARTIFACTS/models/Qwen3-4B-8bit)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"; DIR="${1:?model dir}"; ID="$(basename "$DIR")-unfused"; ADAPTER="$DIR/adapter"; BASE="${BASE_DIR:-$ART/models/Qwen3-4B-8bit}"
[ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter_config.json in $ADAPTER"; exit 2; }
scripts/gpu_lock.sh status | grep -q "held by S " || { echo "hold lock S first (scripts/gpu_lock.sh acquire S 20 serve-unfused)"; exit 3; }
echo "[1/3] stop the vllm-mlx server"; pkill -f "scripts/vllm_mlx_serve.py" 2>/dev/null || true; sleep 2
echo "[2/3] mlx_lm.server: $BASE + $ADAPTER as default_model on :8000"
nohup uv run python -m mlx_lm.server --model "$BASE" --adapter-path "$ADAPTER" --host 127.0.0.1 --port 8000 \
  --chat-template-args '{"enable_thinking": false}' --temp 0.2 --max-tokens 256 > "$ART/logs/server.log" 2>&1 &
for _ in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null || { echo "server did not come up; tail:"; tail -5 "$ART/logs/server.log"; exit 1; }
echo "[3/3] proxy + agent labeled $ID, upstream name pinned to default_model"
LOCAL_LLM_UPSTREAM_NAME=default_model LLM_MODEL="$ID" LOCAL_LLM_MODEL="$ID" R3_OWNER=S scripts/restart.sh --no-server
echo "serving $ID (unfused adapter on $BASE). Rollback: scripts/deploy_model.sh --rollback (or scripts/restart.sh with MODEL_DIR)"
