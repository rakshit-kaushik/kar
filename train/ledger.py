"""cost_ledger rows and runs/<id>/run.json for training runs (CONTRACT.md §7 tables)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

COLUMNS = ("run_id", "model_id", "item", "qty", "unit", "unit_cost", "measured", "note")


def write_run_json(run_dir: str | Path, record: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n")
    os.replace(tmp, path)
    return path


def write_ledger(rows: list[dict], run_id: str, model_id: str | None, database_url: str | None, jsonl_path: str | Path) -> dict:
    """Append rows to runs/<id>/ledger.jsonl always, and to Postgres `cost_ledger` when a
    DATABASE_URL is given and reachable. Row keys: item, qty, [unit, unit_cost, measured, note]."""
    full = []
    for r in rows:
        note = r.get("note")
        full.append({
            "run_id": run_id,
            "model_id": model_id,
            "item": r["item"],
            "qty": float(r["qty"]),
            "unit": r.get("unit"),
            "unit_cost": r.get("unit_cost"),
            "measured": bool(r.get("measured", True)),
            "note": json.dumps(note, default=str) if isinstance(note, (dict, list)) else note,
        })
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        for r in full:
            f.write(json.dumps({**r, "ts": time.time()}) + "\n")
    result = {"db": False, "jsonl": str(jsonl_path), "n": len(full), "error": None}
    if database_url:
        try:
            import psycopg

            with psycopg.connect(database_url, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        "insert into cost_ledger (run_id, model_id, item, qty, unit, unit_cost, measured, note) "
                        "values (%s, %s, %s, %s, %s, %s, %s, %s)",
                        [tuple(r[c] for c in COLUMNS) for r in full],
                    )
                conn.commit()
            result["db"] = True
        except Exception as e:  # the JSONL copy is the fallback of record
            result["error"] = f"{type(e).__name__}: {e}"
    return result
