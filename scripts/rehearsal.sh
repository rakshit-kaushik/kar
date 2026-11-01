#!/usr/bin/env bash
# Unattended rehearsal of the five demo scenarios (demo/scenarios.md) on whatever :8000 serves, under lock S:
# each scenario RUNS times through the simulated caller (audio level, two-channel wav per call), scored by
# scripts/demo_check.py against its expected final state, per-stage latency by g1_report, cost by cost_report,
# all into the ledger under rehearsal-<model>-<stamp>. The plan's T+22-24 "each scenario five times on the frozen build".
# Usage: [RUNS=5] [INT_TURN=5] [SCENARIOS="ordinary correction timeout interruption handoff"] scripts/rehearsal.sh
#   INT_TURN: the caller turn that cuts into the agent's reply in the interruption scenario (5 = the read-back; the run-1
#   candidate answers the slot question in one sentence, too short to cut into at turn 4).
set -uo pipefail
export SIM_TTS="${SIM_TTS:-deepgram}"   # one caller voice across all five scenarios while Cartesia is out of credits (cached per voice)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"; RUNS="${RUNS:-5}"; INT_TURN="${INT_TURN:-5}"
SCENARIOS="${SCENARIOS:-ordinary correction timeout interruption handoff handoff_person}"
ID="$(curl -s -m 5 http://127.0.0.1:8000/v1/models | uv run python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo unknown)"
STAMP="$(date +%m%d-%H%M)"; RUN="rehearsal-$ID-$STAMP"; OUT="$ART/runs/$RUN"; mkdir -p "$OUT/audio"; LOG="$OUT/block.log"
N_SC=$(echo $SCENARIOS | wc -w | tr -d ' '); MINUTES=$(( RUNS * N_SC * 2 + 8 ))   # ~1.5-2 min per call + reports
step() { echo; echo "=== $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; }
scripts/gpu_lock.sh wait S "$MINUTES" "$RUN" | tee -a "$LOG"
trap 'scripts/gpu_lock.sh release S | tee -a "$LOG"' EXIT
step "rehearsal on $ID: $N_SC scenarios x $RUNS, TTS=${AGENT_TTS:-cartesia}, filler=${AGENT_FILLER:-}, loop guard=${AGENT_LOOP_GUARD:-1}; probe"
run uv run python scripts/probe_dispatch.py --timeout 40
for SC in $SCENARIOS; do
  PREFIX="demo-$SC-$STAMP"; EXTRA=""
  case "$SC" in
    timeout) PREFIX="$PREFIX--fault-book_appointment-timeout";;          # the first book_appointment times out (services/agent/faults.py; the final model books directly)
    interruption) EXTRA="--interrupt-turn $INT_TURN --interrupt-after-ms 2500";;
  esac
  step "$SC x$RUNS (rooms $PREFIX-*)"
  run uv run python scripts/sim_caller.py --calls "$RUNS" --concurrency 1 --lines-file "demo/turns/$SC.json" --prefix "$PREFIX" --record-dir "$OUT/audio" $EXTRA --json "$OUT/$SC.json"
  run uv run python scripts/demo_check.py --like "demo-$SC-$STAMP%" --scenario "$SC" --ledger --run-id "$RUN" | tee "$OUT/$SC.check.txt"
  run uv run python scripts/g1_report.py --like "demo-$SC-$STAMP%" --ledger --run-id "$RUN-$SC" --note "rehearsal $SC x$RUNS on $ID, TTS ${AGENT_TTS:-cartesia}" | tee "$OUT/$SC.latency.txt"
done
step "cost"
run uv run python scripts/cost_report.py --like "demo-%-$STAMP%" --ledger --run-id "cost-$RUN"
step "summary"
grep -hE "^[a-z_]+: [0-9]+/[0-9]+ passed" "$OUT"/*.check.txt | tee -a "$LOG"
ls "$OUT/audio" | wc -l | xargs -I{} echo "recordings: {} wav files in $OUT/audio" | tee -a "$LOG"
step "done: $OUT"
