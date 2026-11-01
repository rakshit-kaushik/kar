#!/usr/bin/env bash
# GPU lock: the schedule when no human is arbitrating. One Mac GPU trains or serves load tests, never both.
#   scripts/gpu_lock.sh acquire <owner> <minutes> [purpose]   -> exit 0 if acquired (or already yours), 1 if held by someone else
#   scripts/gpu_lock.sh release <owner>
#   scripts/gpu_lock.sh status                                 -> prints holder, purpose, expiry; exit 0 free / 1 held
#   scripts/gpu_lock.sh wait <owner> <minutes> [purpose]       -> polls every 30 s until acquired; gives up if orphaned
# Lock = $R3_ARTIFACTS/gpu.lock (JSON; GPU_LOCK_FILE overrides, for tests). An expired lock is free. Serving live calls on
# the base model does NOT need the lock; training runs, benchmarks and load tests do. Each session sends a cross-session
# message when it takes/returns it.
# Fail closed (Nov 1 07:14 overlap): a lock file that exists but cannot be read is HELD, never free; acquire and release
# refuse it. Writes go to a temp file and are renamed into place, so a reader never sees a truncated file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; [ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
LOCK="${GPU_LOCK_FILE:-${R3_ARTIFACTS:-$HOME/r3-artifacts}/gpu.lock}"
PY="${GPU_LOCK_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3 || echo python3)"; fi
fi
now() { date +%s; }
# prints "owner until purpose", or "UNREADABLE <now+1h> lock-file-unreadable" when the file exists but cannot be parsed
read_lock() {
  [ -f "$LOCK" ] || return 0
  local out
  if out="$("$PY" -c "import json; d=json.load(open('$LOCK')); print(d['owner'], d['until'], d.get('purpose',''))" 2>/dev/null)" && [ -n "$out" ]; then
    echo "$out"
  else
    echo "UNREADABLE $(( $(now) + 3600 )) lock-file-unreadable"
  fi
}
holder() { set -- $(read_lock); if [ -n "${1:-}" ] && [ "$(now)" -lt "${2:-0}" ]; then echo "$1"; fi; }
describe() {
  set -- $(read_lock)
  if [ "${1:-}" = "UNREADABLE" ]; then echo "gpu: lock file unreadable ($LOCK); treated as held, fix or remove it by hand"; else echo "gpu: held by $1 for another $(( ($2 - $(now)) / 60 )) min ($3)"; fi
}
write_lock() {  # atomic: temp file + rename
  local owner="$1" mins="$2" purpose="$3" tmp="$LOCK.tmp.$$"
  printf '{"owner":"%s","since":%s,"until":%s,"purpose":"%s"}\n' "$owner" "$(now)" "$(( $(now) + mins*60 ))" "$purpose" > "$tmp"
  mv -f "$tmp" "$LOCK"
}
case "${1:-status}" in
  status)
    h=$(holder); if [ -z "$h" ]; then echo "gpu: free"; exit 0; fi
    describe; exit 1 ;;
  acquire)
    owner="${2:?owner}"; mins="${3:?minutes}"; purpose="${4:-}"; h=$(holder)
    if [ -n "$h" ] && [ "$h" != "$owner" ]; then describe; exit 1; fi
    write_lock "$owner" "$mins" "$purpose"
    echo "gpu: acquired by $owner for ${mins} min ($purpose)"; exit 0 ;;
  release)
    owner="${2:?owner}"; h=$(holder)
    if [ -n "$h" ] && [ "$h" != "$owner" ]; then if [ "$h" = "UNREADABLE" ]; then describe; else echo "gpu: held by $h, not released"; fi; exit 1; fi
    rm -f "$LOCK"; echo "gpu: released by $owner"; exit 0 ;;
  wait)
    owner="${2:?owner}"; mins="${3:?minutes}"; purpose="${4:-}"
    until "$0" acquire "$owner" "$mins" "$purpose"; do
      sleep 30
      # an orphaned waiter (its block was killed) must not grab the lock for nobody: Nov 1 07:11, purpose measure-unknown-1101-0513
      if [ "$(ps -o ppid= -p $$ 2>/dev/null | tr -d ' ')" = "1" ]; then echo "gpu: waiter orphaned, giving up"; exit 1; fi
    done ;;
  *) echo "usage: $0 status|acquire <owner> <minutes> [purpose]|release <owner>|wait <owner> <minutes> [purpose]"; exit 2 ;;
esac
