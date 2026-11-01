#!/usr/bin/env bash
# Stop the model server on :8000 (listener only, never connected clients): TERM, wait up to 10 s, then KILL.
# For T's launcher (R3_SERVER_STOP_CMD) and anything else that must free the GPU. Hold scripts/gpu_lock.sh first.
set -uo pipefail
pids="$(lsof -ti:8000 -sTCP:LISTEN 2>/dev/null || true)"
[ -z "$pids" ] && { echo "server: nothing listening on :8000"; exit 0; }
kill $pids 2>/dev/null || true
for i in $(seq 1 20); do sleep 0.5; [ -z "$(lsof -ti:8000 -sTCP:LISTEN 2>/dev/null || true)" ] && { echo "server: stopped (pid $pids)"; exit 0; }; done
kill -9 $pids 2>/dev/null || true; sleep 0.5; echo "server: killed (pid $pids)"
