#!/usr/bin/env bash
# Agent watchdog: every INTERVAL seconds, make sure the LiveKit worker is alive and actually gets dispatched.
# Restarts the worker alone (scripts/restart.sh --agent-only: no lock, nothing shared) when the process is gone,
# or when two consecutive dispatch probes fail while no caller is in any room. Log: $R3_ARTIFACTS/logs/watchdog.log
#   nohup scripts/agent_watchdog.sh >/dev/null 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; [ -f .env ] && set -a && . ./.env && set +a
LOG="${R3_ARTIFACTS:-$HOME/r3-artifacts}/logs/watchdog.log"; PIDFILE="${R3_ARTIFACTS:-$HOME/r3-artifacts}/watchdog.pid"
INTERVAL="${WATCHDOG_INTERVAL:-300}"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "watchdog already running (pid $(cat "$PIDFILE"))"; exit 0; fi
echo $$ > "$PIDFILE"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $(date +%s) $*" >> "$LOG"; }
log "watchdog start (interval ${INTERVAL}s)"
fails=0
while true; do
  if ! pgrep -f "services.agent.main" >/dev/null; then
    log "worker process gone -> restart --agent-only"; scripts/restart.sh --agent-only >>"$LOG" 2>&1; fails=0
  elif uv run python scripts/probe_dispatch.py --timeout 30 >/dev/null 2>&1; then
    fails=0; log "probe ok"
  else
    fails=$((fails + 1)); log "probe FAILED ($fails)"
    if [ "$fails" -ge 2 ]; then
      if uv run python scripts/rooms.py --busy >/dev/null 2>&1; then log "caller present, restart deferred"
      else log "restart --agent-only"; scripts/restart.sh --agent-only >>"$LOG" 2>&1; fails=0; fi
    fi
  fi
  sleep "$INTERVAL"
done
