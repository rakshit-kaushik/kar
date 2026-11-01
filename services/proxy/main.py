"""Run the proxy: `uv run python -m services.proxy.main`. Config from .env:
PROXY_UPSTREAM (default http://127.0.0.1:8000), PROXY_PORT (default 8001), DATABASE_URL (optional)."""
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
    logging.getLogger("r3.proxy").info("upstream=%s turn_events=%s", upstream, "postgres" if sink else "log only")
    return create_app(upstream_client=client, event_sink=sink)


def main() -> None:
    load_dotenv(ROOT / ".env")
    uvicorn.run(build_app(), host="127.0.0.1", port=int(os.environ.get("PROXY_PORT", "8001")), log_level="warning")


if __name__ == "__main__":
    main()
