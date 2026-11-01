#!/usr/bin/env bash
# The GPU block for lane S, scripted so it runs unattended under the lock (lane S owner "S"):
#   1. G2 serve step (optional, SMOKE_MODEL_DIR set): serve the fused adapter under its versioned id, probe, one simulated call, report
#   2. G1: base model (restart skipped when it is already served), 10 sequential simulated calls -> cost_ledger run g1-sim-base,
#      then 3 calls with the LLM caller (hosted model plays Dana) -> g1-sim-base-llm
#   3. admission pair: 2 concurrent calls; with PROXY_ADMISSION_LIMIT=1 the second call's turns go to the hosted model -> load-c2
#      (the 1/2/4/8 audio-level sweep was replaced by scripts/load_text.py: the SimpleEngine serializes, admission limit 1)
#   4. turn-taking A/B: agent restarted with tuned knobs (+ preemptive generation), 4 calls -> g1-sim-tuned; agent restored
#   5. interruption: one call where the caller cuts into the availability read-out (turn 4 after 2.5 s of agent audio) -> g1-interrupt
# Usage: [SMOKE_MODEL_DIR=$R3_ARTIFACTS/models/r3-voice-<hash>] [G1_CALLS=10] [LLM_CALLS=3] [LOCK_MINUTES=40] scripts/g1_block.sh
# Output: $R3_ARTIFACTS/runs/<run id>/{block.log,network.txt,*.json,*_report.txt}
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"
RUN="${RUN_ID:-g1-block-$(date +%m%d-%H%M)}"; OUT="$ART/runs/$RUN"; mkdir -p "$OUT"
BATCH="$(date +%H%M)"
SMOKE_DIR="${SMOKE_MODEL_DIR:-}"
G1_CALLS="${G1_CALLS:-10}"; LLM_CALLS="${LLM_CALLS:-3}"
LOG="$OUT/block.log"
step() { echo; echo "=== $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; }
served() { curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | uv run python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true; }

scripts/gpu_lock.sh wait S "${LOCK_MINUTES:-40}" "$RUN" | tee -a "$LOG"
trap 'scripts/gpu_lock.sh release S | tee -a "$LOG"' EXIT

step "network condition (TCP connect RTT to the LiveKit host)"
uv run python - > "$OUT/network.txt" <<'PY'
import os, socket, statistics, time, urllib.parse
host = urllib.parse.urlparse(os.environ["LIVEKIT_URL"]).hostname; r = []
for _ in range(10):
    t = time.perf_counter()
    try: socket.create_connection((host, 443), timeout=5).close(); r.append((time.perf_counter() - t) * 1000)
    except Exception: pass
    time.sleep(0.2)
print(f"tcp rtt to {host}: median {statistics.median(r):.0f} ms, max {max(r):.0f} ms, stdev {statistics.pstdev(r):.0f} ms, n={len(r)}" if r else "no connection")
PY
NET="$(cat "$OUT/network.txt")"; echo "$NET" | tee -a "$LOG"

if [ -n "$SMOKE_DIR" ] && [ -d "$SMOKE_DIR" ]; then
  SMOKE_ID="$(basename "$SMOKE_DIR")"
  step "G2: serve $SMOKE_ID (fused adapter), probe, one simulated call"
  MODEL_DIR="$SMOKE_DIR" LOCAL_LLM_MODEL="$SMOKE_ID" LLM_MODEL="$SMOKE_ID" R3_OWNER=S run scripts/restart.sh
  run uv run python scripts/probe_dispatch.py --timeout 40
  LLM_MODEL="$SMOKE_ID" run uv run python scripts/sim_caller.py --calls 1 --prefix "r3-g2-$BATCH" --json "$OUT/g2.json"
  run uv run python scripts/g1_report.py --like "r3-g2-$BATCH-%" --ledger --run-id "g2-$SMOKE_ID" --note "G2 serve chain: $SMOKE_ID, one simulated call; $NET" | tee "$OUT/g2_report.txt"
fi

step "G1: base model, $G1_CALLS sequential simulated calls (fixed Dana script)"
if [ "$(served)" = "${LOCAL_LLM_MODEL:-r3-voice-base}" ] && curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null; then
  echo "  base already served as $(served); restart skipped" | tee -a "$LOG"
else
  R3_OWNER=S run scripts/restart.sh
fi
run uv run python scripts/probe_dispatch.py --timeout 40
run uv run python scripts/sim_caller.py --calls "$G1_CALLS" --concurrency 1 --prefix "r3-g1-$BATCH" --json "$OUT/g1.json"
run uv run python scripts/g1_report.py --like "r3-g1-$BATCH-%" --ledger --run-id g1-sim-base --note "G1 simulated audio calls, base model, sequential; $NET" | tee "$OUT/g1_report.txt"

if [ "$LLM_CALLS" -gt 0 ]; then
  step "G1 with the LLM caller: $LLM_CALLS calls (hosted model plays the caller, improvised)"
  run uv run python scripts/sim_caller.py --calls "$LLM_CALLS" --concurrency 1 --script llm --prefix "r3-g1llm-$BATCH" --json "$OUT/g1_llm.json"
  run uv run python scripts/g1_report.py --like "r3-g1llm-$BATCH-%" --ledger --run-id g1-sim-base-llm --note "G1 simulated calls with the LLM caller, base model; $NET" | tee "$OUT/g1_llm_report.txt"
fi

step "admission pair: 2 concurrent calls (PROXY_ADMISSION_LIMIT=${PROXY_ADMISSION_LIMIT:-1}: the second call's turns are routed to the hosted model)"
for I in 1 2; do
  uv run python scripts/sim_caller.py --calls 1 --prefix "r3-load-c2-$BATCH-p$I" --json "$OUT/load_c2_p$I.json" >> "$LOG" 2>&1 &
  sleep 0.5
done
wait
grep -h "^\[r3-load-c2-$BATCH" "$LOG" | tail -n 2
run uv run python scripts/g1_report.py --like "r3-load-c2-$BATCH-%" --ledger --run-id load-c2 --note "2 concurrent simulated calls, base model, admission routing to hosted above 1; $NET" | tee "$OUT/load_c2_report.txt"

step "turn-taking A/B: VAD 0.4 s, endpointing 0.25-2.0 s, preemptive generation on; 4 sequential calls"
AGENT_VAD_MIN_SILENCE=0.4 AGENT_ENDPOINT_MIN=0.25 AGENT_ENDPOINT_MAX=2.0 AGENT_PREEMPTIVE=1 run scripts/restart.sh --agent-only
run uv run python scripts/probe_dispatch.py --timeout 40
run uv run python scripts/sim_caller.py --calls 4 --concurrency 1 --prefix "r3-tuned-$BATCH" --json "$OUT/tuned.json"
run uv run python scripts/g1_report.py --like "r3-tuned-$BATCH-%" --ledger --run-id g1-sim-tuned --note "tuned turn-taking: VAD 0.4, endpoint 0.25/2.0, preemptive; $NET" | tee "$OUT/tuned_report.txt"
step "restore the agent with .env defaults"
run scripts/restart.sh --agent-only
run uv run python scripts/probe_dispatch.py --timeout 40

step "interruption: the caller cuts into the availability read-out (turn 4) after 2.5 s of agent audio"
run uv run python scripts/sim_caller.py --calls 1 --interrupt-turn 4 --interrupt-after-ms 2500 --prefix "r3-int-$BATCH" --json "$OUT/interrupt.json"
run uv run python scripts/g1_report.py --like "r3-int-$BATCH-%" --ledger --run-id g1-interrupt --note "interruption test: caller cuts in at turn 4 after 2.5 s; $NET" | tee "$OUT/interrupt_report.txt"

step "done: $OUT"
