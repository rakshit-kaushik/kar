#!/usr/bin/env bash
# Deploy a fused model directory as the live voice model under its versioned id, with rollback.
#   scripts/deploy_model.sh $R3_ARTIFACTS/models/r3-voice-<hash> [--owner S]   -> serve it (server + proxy + agent), probe, ledger row
#   scripts/deploy_model.sh --rollback                                           -> serve the previous model again
#   scripts/deploy_model.sh --status                                             -> current and previous
# State: $R3_ARTIFACTS/deploy/current (model dir), $R3_ARTIFACTS/deploy/previous. Hold the GPU lock (scripts/gpu_lock.sh)
# when other sessions may be using the server; the restart itself takes ~25 s and the proxy falls back to the
# hosted model for live calls meanwhile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"; ST="$ART/deploy"; mkdir -p "$ST"
OWNER="${R3_OWNER:-S}"
cur() { [ -f "$ST/current" ] && cat "$ST/current" || echo "$ART/models/Qwen3-4B-8bit"; }
case "${1:-}" in
  --status) echo "current:  $(cur)"; echo "previous: $([ -f "$ST/previous" ] && cat "$ST/previous" || echo '-')"; exit 0 ;;
  --rollback) [ -f "$ST/previous" ] || { echo "no previous model recorded"; exit 1; }; TARGET="$(cat "$ST/previous")"; NOTE="rollback" ;;
  "") echo "usage: $0 <model dir> | --rollback | --status"; exit 2 ;;
  *) TARGET="$1"; NOTE="deploy" ;;
esac
[ -d "$TARGET" ] || { echo "not a directory: $TARGET"; exit 1; }
ID="$(basename "$TARGET")"; [ "$ID" = "Qwen3-4B-8bit" ] && ID="r3-voice-base"
PREV="$(cur)"
echo "$NOTE: $ID from $TARGET (previous: $PREV)"
MODEL_DIR="$TARGET" LOCAL_LLM_MODEL="$ID" LLM_MODEL="$ID" R3_OWNER="$OWNER" scripts/restart.sh
# Prove the weights, not the name: the server process must be serving exactly $TARGET (Nov 1 08:02 incident: the name said
# run 2, the process served run 1). /v1/models only echoes --served-model-name, so it cannot catch this.
SERVING="$(ps -o command= -p "$(lsof -ti:8000 -sTCP:LISTEN | head -1)" 2>/dev/null | sed -n 's/.*vllm_mlx_serve.py serve \([^ ]*\) .*/\1/p')"
if [ "$SERVING" != "$TARGET" ]; then
  echo "DEPLOY FAILED: :8000 is serving '$SERVING', not '$TARGET'. State files and .env left unchanged." >&2
  uv run python - "$ID" "$TARGET" "$SERVING" <<'PY'
import os, sys, psycopg
mid, mdir, serving = sys.argv[1:4]
with psycopg.connect(os.environ["DATABASE_URL"]) as con:
    con.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values ('deploy', %s, 'deploy.failed_path_check', 1, 'model', true, %s)", (mid, f"wanted={mdir}; serving={serving}"))
PY
  exit 1
fi
echo "  serving path verified: $SERVING"
uv run python scripts/probe_dispatch.py --timeout 40 || uv run python scripts/probe_dispatch.py --timeout 40
# make the choice durable: .env carries the live model id so later plain restarts keep serving it
python3 - "$ID" "$TARGET" <<'PY'
import pathlib, re, sys
mid, mdir = sys.argv[1], sys.argv[2]
p = pathlib.Path(".env"); t = p.read_text()
for k, v in (("LLM_MODEL", mid), ("LOCAL_LLM_MODEL", mid), ("MODEL_DIR", mdir)):
    t = re.sub(rf"^{k}=.*$", f"{k}={v}", t, flags=re.M) if re.search(rf"^{k}=", t, re.M) else t.rstrip("\n") + f"\n{k}={v}\n"
p.write_text(t)
PY
[ "$PREV" != "$TARGET" ] && echo "$PREV" > "$ST/previous"
echo "$TARGET" > "$ST/current"
PROV="$([ -f "$TARGET/r3.json" ] && tr -d '\n' < "$TARGET/r3.json" | cut -c1-400 || echo '{}')"
uv run python - "$ID" "$TARGET" "$NOTE" "$PREV" "$PROV" <<'PY'
import os, sys, psycopg
mid, mdir, note, prev, prov = sys.argv[1:6]
with psycopg.connect(os.environ["DATABASE_URL"]) as con:
    con.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values ('deploy', %s, %s, 1, 'model', true, %s)",
                (mid, f"deploy.{note}", f"dir={mdir}; previous={prev}; provenance={prov}"))
print("ledger: deploy row written")
PY
echo "live model: $ID ($TARGET); rollback: scripts/deploy_model.sh --rollback"
