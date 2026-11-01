#!/usr/bin/env bash
# GPU lock: the schedule when no human is arbitrating. One Mac GPU trains or serves load tests, never both.
#   scripts/gpu_lock.sh acquire <owner> <minutes> [purpose]   -> exit 0 if acquired (or already yours), 1 if held by someone else
#   scripts/gpu_lock.sh release <owner>
#   scripts/gpu_lock.sh status                                 -> prints holder, purpose, expiry; exit 0 free / 1 held
#   scripts/gpu_lock.sh wait <owner> <minutes> [purpose]       -> polls every 30 s until acquired
# Lock = $R3_ARTIFACTS/gpu.lock (JSON). An expired lock is free. Serving live calls on the base model does NOT need the
# lock; training runs, benchmarks and load tests do. Each session sends a cross-session message when it takes/returns it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; [ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
LOCK="${R3_ARTIFACTS:-$HOME/r3-artifacts}/gpu.lock"
now() { date +%s; }
read_lock() { [ -f "$LOCK" ] && python3 -c "import json,sys; d=json.load(open('$LOCK')); print(d['owner'], d['until'], d.get('purpose','')); " 2>/dev/null || true; }
holder() { set -- $(read_lock); if [ -n "${1:-}" ] && [ "$(now)" -lt "${2:-0}" ]; then echo "$1"; fi; }
case "${1:-status}" in
  status)
    h=$(holder); if [ -z "$h" ]; then echo "gpu: free"; exit 0; fi
    set -- $(read_lock); echo "gpu: held by $1 until $(date -r "$2" '+%H:%M:%S') ($3)"; exit 1 ;;
  acquire)
    owner="${2:?owner}"; mins="${3:?minutes}"; purpose="${4:-}"; h=$(holder)
    if [ -n "$h" ] && [ "$h" != "$owner" ]; then set -- $(read_lock); echo "gpu: held by $1 until $(date -r "$2" '+%H:%M:%S') ($3)"; exit 1; fi
    printf '{"owner":"%s","since":%s,"until":%s,"purpose":"%s"}\n' "$owner" "$(now)" "$(( $(now) + mins*60 ))" "$purpose" > "$LOCK"
    echo "gpu: acquired by $owner for ${mins} min ($purpose)"; exit 0 ;;
  release)
    owner="${2:?owner}"; h=$(holder)
    if [ -n "$h" ] && [ "$h" != "$owner" ]; then echo "gpu: held by $h, not released"; exit 1; fi
    rm -f "$LOCK"; echo "gpu: released by $owner"; exit 0 ;;
  wait)
    owner="${2:?owner}"; mins="${3:?minutes}"; purpose="${4:-}"
    until "$0" acquire "$owner" "$mins" "$purpose"; do sleep 30; done ;;
  *) echo "usage: $0 status|acquire <owner> <minutes> [purpose]|release <owner>|wait <owner> <minutes> [purpose]"; exit 2 ;;
esac
