#!/usr/bin/env bash
# gpu_flip.sh hosted|local — point every service's LLM_BASE_URL/LLM_MODEL at the hosted or local endpoint.
# The training launcher calls `hosted` before a run and `local` after; the agent re-reads .env per call.
set -euo pipefail
cd "$(dirname "$0")/.."; set -a; source .env; set +a
case "${1:-}" in
  hosted) U=$HOSTED_LLM_BASE_URL; M=$HOSTED_LLM_MODEL ;;
  local)  U=$LOCAL_LLM_BASE_URL;  M=$LOCAL_LLM_MODEL ;;
  *) echo "usage: $0 hosted|local"; exit 2 ;;
esac
sed -i '' -e "s|^LLM_BASE_URL=.*|LLM_BASE_URL=$U|" -e "s|^LLM_MODEL=.*|LLM_MODEL=$M|" .env
echo "LLM endpoint -> $1 ($M @ $U)"
