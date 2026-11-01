"""Run the proxy: `uv run python -m services.proxy.main`. Config from .env:
PROXY_UPSTREAM (default http://127.0.0.1:8000), PROXY_PORT (default 8001), DATABASE_URL (optional),
PROXY_FALLBACK (default 1; 0 disables), PROXY_FIRST_TOKEN_TIMEOUT_MS (default 2000), PROXY_ADMISSION_LIMIT (default 1; 0 disables),
HOSTED_LLM_BASE_URL / HOSTED_LLM_API_KEY / HOSTED_LLM_MODEL (the fallback target; all three required),
LOCAL_LLM_UPSTREAM_NAME (optional: the name the local server answers to; discovered from its /v1/models when unset)."""
from __future__ import annotations

import logging
import os
import pathlib

import httpx
import uvicorn
from dotenv import load_dotenv

from services.proxy.app import create_app
from services.proxy.events import PostgresSink

ROOT = pathlib.Path(__file__).resolve().parents[2]


def build_app():
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    upstream = os.environ.get("PROXY_UPSTREAM", "http://127.0.0.1:8000")
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(300.0, connect=5.0))
    dsn = os.environ.get("DATABASE_URL")
    sink = PostgresSink(dsn) if dsn else None

    fallback_client = fallback_model = None
    hosted_url, hosted_key, hosted_model = (os.environ.get(k) for k in ("HOSTED_LLM_BASE_URL", "HOSTED_LLM_API_KEY", "HOSTED_LLM_MODEL"))
    if os.environ.get("PROXY_FALLBACK", "1") != "0" and hosted_url and hosted_key and hosted_model:
        # the proxy posts to /v1/chat/completions itself, so the base is the host without /v1
        base = hosted_url.rstrip("/").removesuffix("/v1")
        fallback_client = httpx.AsyncClient(base_url=base, headers={"Authorization": f"Bearer {hosted_key}"}, timeout=httpx.Timeout(120.0, connect=5.0))
        fallback_model = hosted_model
    timeout_s = float(os.environ.get("PROXY_FIRST_TOKEN_TIMEOUT_MS", "2000")) / 1000

    logging.getLogger("r3.proxy").info("upstream=%s turn_events=%s fallback=%s first_token_timeout=%.1fs", upstream, "postgres" if sink else "log only", fallback_model or "off", timeout_s)
    admission = int(os.environ.get("PROXY_ADMISSION_LIMIT", "1"))  # measured Nov 1: the SimpleEngine serves one call within budget
    logging.getLogger("r3.proxy").info("admission_limit=%d (turns above it go hosted when available)", admission)
    return create_app(upstream_client=client, event_sink=sink, fallback_client=fallback_client, fallback_model=fallback_model, first_token_timeout=timeout_s, admission_limit=admission,
                      upstream_model=os.environ.get("LOCAL_LLM_UPSTREAM_NAME") or None)


def main() -> None:
    load_dotenv(ROOT / ".env")
    uvicorn.run(build_app(), host="127.0.0.1", port=int(os.environ.get("PROXY_PORT", "8001")), log_level="warning")


if __name__ == "__main__":
    main()
