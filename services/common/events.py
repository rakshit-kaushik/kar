"""Turn-event sink shared by the proxy and the agent: appends rows to Postgres `turn_events` (CONTRACT.md §6)."""
from __future__ import annotations

import logging

import psycopg
from psycopg.types.json import Jsonb

log = logging.getLogger("r3.events")


class PostgresSink:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: psycopg.AsyncConnection | None = None

    async def _connection(self) -> psycopg.AsyncConnection:
        if self._conn is None or self._conn.closed:
            self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        return self._conn

    async def __call__(self, ev: dict) -> None:
        try:
            conn = await self._connection()
            await conn.execute(
                "insert into turn_events(call_id, turn_idx, model_id, event, ts_ms, payload) values (%s, %s, %s, %s, %s, %s)",
                (ev["call_id"], ev["turn_idx"], ev["model_id"], ev["event"], ev["ts_ms"], Jsonb(ev.get("payload") or {})),
            )
        except Exception as e:  # never let logging take down a call
            log.warning("turn_events insert failed: %r", e)
            self._conn = None
