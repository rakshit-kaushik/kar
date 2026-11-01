#!/usr/bin/env bash
# Lane S measurement block (≤12 min under lock S), no redeploy: probe, trie check lines, N sequential simulated calls + 1
# interruption call on whatever :8000 serves, g1_report + cost into the ledger, release. For the clean latency numbers of
# a deployed candidate once the GPU is free (training and eval batches pollute TTFT).
# Usage: [CALLS=3] [TAG=clean] scripts/measure_block.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"; CALLS="${CALLS:-3}"; TAG="${TAG:-clean}"
ID="$(curl -s -m 5 http://127.0.0.1:8000/v1/models | uv run python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo unknown)"
RUN="measure-$ID-$(date +%m%d-%H%M)"; OUT="$ART/runs/$RUN"; mkdir -p "$OUT"; LOG="$OUT/block.log"; BATCH="$(date +%H%M)"
step() { echo; echo "=== $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; }
scripts/gpu_lock.sh wait S 12 "$RUN" | tee -a "$LOG"
trap 'scripts/gpu_lock.sh release S | tee -a "$LOG"' EXIT
step "serving $ID; TTS=${AGENT_TTS:-cartesia}; probe"
run uv run python scripts/probe_dispatch.py --timeout 40
step "$CALLS sequential simulated calls + 1 interruption call"
run uv run python scripts/sim_caller.py --calls "$CALLS" --concurrency 1 --prefix "r3-$TAG-$BATCH" --json "$OUT/calls.json"
run uv run python scripts/sim_caller.py --calls 1 --interrupt-turn 4 --interrupt-after-ms 2500 --prefix "r3-${TAG}int-$BATCH" --json "$OUT/interrupt.json"
run uv run python scripts/g1_report.py --like "r3-$TAG-$BATCH-%" --ledger --run-id "g1-sim-$ID" --note "$ID, $CALLS sequential simulated calls, GPU free, TTS ${AGENT_TTS:-cartesia}" | tee "$OUT/g1_report.txt"
run uv run python scripts/g1_report.py --like "r3-${TAG}int-$BATCH-%" --ledger --run-id "g1-interrupt-$ID" --note "$ID interruption test" | tee "$OUT/interrupt_report.txt"
run uv run python scripts/cost_report.py --like "r3-$TAG-$BATCH-%" --ledger --run-id "cost-$ID"
grep -E "Prefix trie cache HIT" "$ART/logs/server.log" | tail -3 | tee -a "$LOG"
step "done: $OUT"
