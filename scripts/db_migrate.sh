#!/usr/bin/env bash
# Apply every migration in order against $DATABASE_URL (idempotent SQL).
set -euo pipefail
cd "$(dirname "$0")/.."; set -a; source .env; set +a
PSQL=${PSQL:-/opt/homebrew/opt/postgresql@15/bin/psql}
for f in db/migrations/*.sql; do echo "== $f"; "$PSQL" "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f "$f"; done
