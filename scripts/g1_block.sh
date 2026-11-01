#!/usr/bin/env bash
# The GPU block for lane S, scripted so it runs unattended under the lock:
#   1. G2 serve step: serve T's fused smoke adapter under its versioned id, probe, one simulated call, report
#   2. G1: base model, 10 simulated calls (2 at a time), report -> cost_ledger run g1-sim-base
#   3. load tests: 1 / 2 / 4 / 8 concurrent simulated calls, report per level -> cost_ledger run load-cN
#   4. turn-taking A/B: agent restarted with tuned knobs (+ preemptive generation), 4 calls, report; agent restored
# Usage: SMOKE_MODEL_DIR=$R3_ARTIFACTS/models/r3-voice-<hash> scripts/g1_block.sh   (SMOKE_MODEL_DIR optional)
# Output: $R3_ARTIFACTS/runs/<run id>/{block.log,network.txt,*.json,*_report.txt}
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"
RUN="${RUN_ID:-g1-block-$(date +%m%d-%H%M)}"; OUT="$ART/runs/$RUN"; mkdir -p "$OUT"
BATCH="$(date +%H%M)"
SMOKE_DIR="${SMOKE_MODEL_DIR:-}"
LOG="$OUT/block.log"
step() { echo; echo "=== $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; }

scripts/gpu_lock.sh wait S "${LOCK_MINUTES:-55}" "$RUN" | tee -a "$LOG"
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
  step "G2: serve $SMOKE_ID (fused smoke adapter), probe, one simulated call"
  MODEL_DIR="$SMOKE_DIR" LOCAL_LLM_MODEL="$SMOKE_ID" LLM_MODEL="$SMOKE_ID" R3_OWNER=S run scripts/restart.sh
  run uv run python scripts/probe_dispatch.py --timeout 40
  LLM_MODEL="$SMOKE_ID" run uv run python scripts/sim_caller.py --calls 1 --prefix "r3-g2-$BATCH" --json "$OUT/g2.json"
  run uv run python scripts/g1_report.py --like "r3-g2-$BATCH-%" --ledger --run-id "g2-$SMOKE_ID" --note "G2 serve chain: $SMOKE_ID, one simulated call; $NET" | tee "$OUT/g2_report.txt"
fi

step "G1: base model, 10 simulated calls, 2 concurrent"
R3_OWNER=S run scripts/restart.sh
run uv run python scripts/probe_dispatch.py --timeout 40
run uv run python scripts/sim_caller.py --calls 10 --concurrency 2 --prefix "r3-g1-$BATCH" --json "$OUT/g1.json"
run uv run python scripts/g1_report.py --like "r3-g1-$BATCH-%" --ledger --run-id g1-sim-base --note "G1 simulated audio calls, base model, 2 concurrent; $NET" | tee "$OUT/g1_report.txt"

for C in 1 2 4 8; do
  N=$C; [ "$C" -le 2 ] && N=$((C * 2))
  step "load test: $C concurrent calls ($N calls), one caller process per concurrent call so the 10 ms audio pumps do not share an event loop"
  WAVES=$((N / C))
  for W in $(seq 1 "$WAVES"); do
    for I in $(seq 1 "$C"); do
      uv run python scripts/sim_caller.py --calls 1 --prefix "r3-load-c$C-$BATCH-w${W}p$I" --json "$OUT/load_c${C}_w${W}_p$I.json" >> "$LOG" 2>&1 &
      sleep 0.5
    done
    wait
  done
  grep -h "^\[r3-load-c$C-$BATCH" "$LOG" | tail -n "$N"
  run uv run python scripts/g1_report.py --like "r3-load-c$C-$BATCH-%" --ledger --run-id "load-c$C" --note "load test $C concurrent simulated calls, base model; $NET" | tee "$OUT/load_c${C}_report.txt"
done

step "turn-taking A/B: VAD 0.4 s, endpointing 0.25-2.0 s, preemptive generation on; 4 calls, 2 concurrent"
AGENT_VAD_MIN_SILENCE=0.4 AGENT_ENDPOINT_MIN=0.25 AGENT_ENDPOINT_MAX=2.0 AGENT_PREEMPTIVE=1 run scripts/restart.sh --agent-only
run uv run python scripts/probe_dispatch.py --timeout 40
run uv run python scripts/sim_caller.py --calls 4 --concurrency 2 --prefix "r3-tuned-$BATCH" --json "$OUT/tuned.json"
run uv run python scripts/g1_report.py --like "r3-tuned-$BATCH-%" --ledger --run-id g1-sim-tuned --note "tuned turn-taking: VAD 0.4, endpoint 0.25/2.0, preemptive; $NET" | tee "$OUT/tuned_report.txt"
step "restore the agent with .env defaults"
run scripts/restart.sh --agent-only
run uv run python scripts/probe_dispatch.py --timeout 40

step "done: $OUT"
