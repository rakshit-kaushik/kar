"""scripts/cost_report.py against the real database, inside a transaction that is rolled back."""
import importlib.util
import os
import pathlib
import time

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")
spec = importlib.util.spec_from_file_location("cost_report", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "cost_report.py")
cost_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost_report)


@pytest.fixture
def con():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    with psycopg.connect(url) as c:
        yield c
        c.rollback()


def _call(con, call_id: str, model_id: str) -> None:
    now = int(time.time() * 1000)
    con.execute("insert into sessions(call_id, model_id, started_at, ended_at, outcome) values (%s, %s, now(), now(), 'completed')", (call_id, model_id))
    for i, ev in enumerate(("llm_request", "llm_done", "call_end")):
        payload = '{"duration_s": 60}' if ev == "call_end" else '{"ms": 500, "prompt_tokens": 100, "completion_tokens": 10}'
        con.execute("insert into turn_events(call_id, turn_idx, model_id, event, ts_ms, payload) values (%s, 0, %s, %s, %s, %s::jsonb)", (call_id, model_id, ev, now + i, payload))


def test_live_calls_like_filter_applies_to_every_ended_call(con):
    _call(con, "zz-test-cost-a-1", "zz-test-model")
    _call(con, "zz-test-cost-b-1", "zz-test-model")
    groups = cost_report.live_calls(con, "zz-test-cost-a-%")
    assert [u["call_id"] for u in groups.get("zz-test-model", [])] == ["zz-test-cost-a-1"]
