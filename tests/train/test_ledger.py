import json
import os

import pytest

from train.ledger import write_ledger, write_run_json

ROWS = [
    {"item": "train_wall_time", "qty": 12.5, "unit": "s"},
    {"item": "train_tokens_per_sec_total", "qty": 812.0, "unit": "tokens/s", "note": "smoke"},
]


def test_write_ledger_falls_back_to_jsonl_without_a_database(tmp_path):
    res = write_ledger(ROWS, run_id="r1", model_id="r3-voice-abc", database_url="", jsonl_path=tmp_path / "ledger.jsonl")
    assert res["db"] is False and res["n"] == 2
    rows = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert rows[0]["run_id"] == "r1" and rows[0]["model_id"] == "r3-voice-abc"
    assert rows[1]["item"] == "train_tokens_per_sec_total" and rows[1]["measured"] is True


def test_write_ledger_reaches_postgres_when_available(tmp_path):
    from train.env import load_env

    load_env()
    url = os.environ.get("DATABASE_URL")
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = psycopg.connect(url, connect_timeout=2)
    except Exception as e:  # pragma: no cover - depends on the machine
        pytest.skip(f"postgres not reachable: {e}")
    run_id = "selftest-ledger"
    try:
        res = write_ledger(ROWS, run_id=run_id, model_id="selftest", database_url=url, jsonl_path=tmp_path / "l.jsonl")
        assert res["db"] is True and res["error"] is None
        with conn.cursor() as cur:
            cur.execute("select item, qty, unit, measured from cost_ledger where run_id = %s order by id", (run_id,))
            got = cur.fetchall()
        assert [g[0] for g in got] == ["train_wall_time", "train_tokens_per_sec_total"]
        assert float(got[0][1]) == 12.5 and got[0][3] is True
    finally:
        with conn.cursor() as cur:
            cur.execute("delete from cost_ledger where run_id = %s", (run_id,))
        conn.commit()
        conn.close()


def test_write_run_json_roundtrip(tmp_path):
    p = write_run_json(tmp_path / "run", {"run_id": "x", "status": "planned"})
    assert p.name == "run.json" and json.loads(p.read_text())["status"] == "planned"
    write_run_json(tmp_path / "run", {"run_id": "x", "status": "done"})
    assert json.loads(p.read_text())["status"] == "done"
