#!/usr/bin/env bash
# Lane S hand-back block (≤15 min under lock S): deploy a fused candidate, probe, verify the prefix trie hits on a 2-turn
# request pair, 3 simulated calls + 1 interruption call, reports + cost into the ledger, release.
# Usage: scripts/deploy_block.sh $R3_ARTIFACTS/models/r3-voice-<hash> [CALLS=3]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"; DIR="${1:?model dir}"; ID="$(basename "$DIR")"; CALLS="${CALLS:-3}"
RUN="deploy-$ID-$(date +%m%d-%H%M)"; OUT="$ART/runs/$RUN"; mkdir -p "$OUT"; LOG="$OUT/block.log"; BATCH="$(date +%H%M)"
step() { echo; echo "=== $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; }
scripts/gpu_lock.sh wait S 15 "$RUN" | tee -a "$LOG"
trap 'scripts/gpu_lock.sh release S | tee -a "$LOG"' EXIT
step "deploy $ID"
R3_OWNER=S run scripts/deploy_model.sh "$DIR"
run uv run python scripts/probe_dispatch.py --timeout 40
step "prefix trie: 2-turn request pair straight to :8000 (second request must log a trie HIT)"
uv run python - "$ID" "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, time, urllib.request, glob
model, out = sys.argv[1], sys.argv[2]
tools = [json.load(open(p)) for p in sorted(glob.glob("schemas/tools/*.json"))]
tools = [t if t.get("type") == "function" else {"type": "function", "function": t} for t in tools]
sysmsg = {"role": "system", "content": open("services/agent/system_prompt.md").read() if glob.glob("services/agent/system_prompt.md") else "You are the phone agent of Marlow Mutual Bank. " * 40}
def ask(msgs):
    body = json.dumps({"model": model, "messages": msgs, "tools": tools, "stream": False, "max_tokens": 40, "temperature": 0.0}).encode()
    t0 = time.perf_counter(); r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=60)
    d = json.loads(r.read()); dt = time.perf_counter() - t0
    m = d["choices"][0]["message"]; return dt, d.get("usage", {}), (m.get("content") or "")[:80], m.get("tool_calls")
m1 = [sysmsg, {"role": "assistant", "content": "Thanks for calling Marlow Mutual Bank. Can I get the phone number on your account?"}, {"role": "user", "content": "Five five five zero one four two two one nine."}]
dt1, u1, c1, tc1 = ask(m1)
m2 = m1 + [{"role": "assistant", "content": c1 or "Can you please provide your date of birth?"}, {"role": "user", "content": "March twelfth nineteen eighty eight."}]
dt2, u2, c2, tc2 = ask(m2)
print(f"turn1 {dt1*1000:.0f} ms prompt {u1.get('prompt_tokens')} tok -> {c1!r} {('tool_calls' if tc1 else '')}")
print(f"turn2 {dt2*1000:.0f} ms prompt {u2.get('prompt_tokens')} tok -> {c2!r} {('tool_calls' if tc2 else '')}")
json.dump({"turn1_ms": dt1 * 1000, "turn2_ms": dt2 * 1000, "usage1": u1, "usage2": u2}, open(f"{out}/trie_probe.json", "w"))
PY
grep -E "Prefix trie cache HIT|System KV cache" "$ART/logs/server.log" | tail -3 | tee -a "$LOG"
step "$CALLS sequential simulated calls + 1 interruption call on $ID"
run uv run python scripts/sim_caller.py --calls "$CALLS" --concurrency 1 --prefix "r3-dep-$BATCH" --json "$OUT/calls.json"
run uv run python scripts/sim_caller.py --calls 1 --interrupt-turn 4 --interrupt-after-ms 2500 --prefix "r3-depint-$BATCH" --json "$OUT/interrupt.json"
run uv run python scripts/g1_report.py --like "r3-dep-$BATCH-%" --ledger --run-id "g1-sim-$ID" --note "candidate $ID, sequential simulated calls, trie fix live" | tee "$OUT/g1_report.txt"
run uv run python scripts/g1_report.py --like "r3-depint-$BATCH-%" --ledger --run-id "g1-interrupt-$ID" --note "candidate $ID interruption test" | tee "$OUT/interrupt_report.txt"
run uv run python scripts/cost_report.py --like "r3-dep-$BATCH-%" --ledger --run-id "cost-$ID"
step "done: $OUT"
