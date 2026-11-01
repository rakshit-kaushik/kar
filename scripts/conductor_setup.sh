#!/usr/bin/env bash
# Conductor workspace setup: copy .env from the root checkout, sync the venv from the shared uv cache.
set -euo pipefail
ROOT="${CONDUCTOR_ROOT_PATH:-/Users/abgup/Documents/Code/kar/kar}"
[ -f .env ] || cp "$ROOT/.env" .env
uv sync --extra dev
echo "workspace ready: $(uv run python -c 'import mlx_lm, livekit.agents; print(mlx_lm.__version__, livekit.agents.__version__)')"
