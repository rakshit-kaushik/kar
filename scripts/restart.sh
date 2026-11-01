#!/usr/bin/env bash
# Bring up Postgres check, model server (8000), proxy (8001), agent, UI (3000) in order; verify each. Filled in by session S.
set -euo pipefail
cd "$(dirname "$0")/.."; set -a; source .env; set +a
echo "TODO(S): server, proxy, agent, ui"; exit 1
