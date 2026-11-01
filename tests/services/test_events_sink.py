"""The turn-event sink is shared by concurrent tasks (the proxy's hedge, the agent's callbacks). One psycopg AsyncConnection
refuses a second command while one is in flight ("another command is already in progress"): 29 proxy rows were lost on Nov 1."""
import asyncio

from services.common import events


class OneCommandAtATime:
    """Behaves like psycopg.AsyncConnection under concurrency: a second execute while one is in flight raises."""

    def __init__(self) -> None:
        self.busy = False
        self.closed = False
        self.rows: list[tuple] = []

    async def execute(self, sql: str, params: tuple) -> None:
        if self.busy:
            raise RuntimeError("sending query and params failed: another command is already in progress")
        self.busy = True
        await asyncio.sleep(0.005)
        self.rows.append(params)
        self.busy = False


async def test_concurrent_events_all_reach_the_table(monkeypatch):
    conn = OneCommandAtATime()
    sink = events.PostgresSink("postgresql://unused")

    async def same_connection():
        return conn

    monkeypatch.setattr(sink, "_connection", same_connection)
    evs = [{"call_id": "c1", "turn_idx": i, "model_id": "m", "event": "llm_first_token", "ts_ms": 1000 + i, "payload": {}} for i in range(6)]
    await asyncio.gather(*(sink(ev) for ev in evs))
    assert sorted(r[1] for r in conn.rows) == list(range(6))
