#!/usr/bin/env bash
# Backup for the sprint (plan T+16-18: "the model directory and ledger copied to a second disk"). This Mac has no
# second disk (only app images under /Volumes), so everything goes to $R3_ARTIFACTS/backups/<stamp>/ on the same
# disk; copy that directory elsewhere when one exists.
#   git bundle of every local branch (origin is far behind and no push was asked for), pg_dump of the r3 database
#   (turn_events, cost_ledger, eval_results, sessions, cost_rates, tickets ...), a snapshot of .env (secrets: keep the
#   backups directory private), the deploy state files, and (BACKUP_MODELS=1) the currently deployed model directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; set -a; . ./.env; set +a
ART="${R3_ARTIFACTS:-$HOME/r3-artifacts}"; STAMP="$(date +%Y%m%d-%H%M)"; DEST="$ART/backups/$STAMP"; mkdir -p "$DEST"
PG_DUMP="${PG_DUMP:-$(ls /opt/homebrew/opt/postgresql@15/bin/pg_dump /opt/homebrew/bin/pg_dump 2>/dev/null | head -1)}"
git bundle create "$DEST/kar-all.bundle" --all >/dev/null 2>&1 && git bundle verify "$DEST/kar-all.bundle" >/dev/null 2>&1 && echo "git: $DEST/kar-all.bundle ($(git rev-parse --short main) main)"
if [ -n "$PG_DUMP" ]; then "$PG_DUMP" "$DATABASE_URL" | gzip > "$DEST/r3.sql.gz" && echo "postgres: $DEST/r3.sql.gz ($(du -h "$DEST/r3.sql.gz" | cut -f1))"; else echo "postgres: pg_dump not found (set PG_DUMP)"; fi
cp .env "$DEST/env.snapshot" && chmod 600 "$DEST/env.snapshot" && echo "env: $DEST/env.snapshot"
[ -d "$ART/deploy" ] && cp -R "$ART/deploy" "$DEST/deploy" && echo "deploy state: $(cat "$ART/deploy/current" 2>/dev/null || echo none)"
if [ "${BACKUP_MODELS:-0}" = 1 ]; then
  CUR="$(cat "$ART/deploy/current" 2>/dev/null || echo "$ART/models/Qwen3-4B-8bit")"
  rsync -a "$CUR" "$DEST/models/" && echo "model: $CUR -> $DEST/models/ ($(du -sh "$DEST/models" | cut -f1))"
fi
ls -la "$DEST" | tail -n +2 | awk '{print "  " $5 "\t" $9}'
echo "backup: $DEST"
