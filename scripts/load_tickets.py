"""Load E's synthetic tickets (JSONL: ticket_id, text, label, split ...) into the Postgres `tickets` table for the
routing batch job: category = gold label, summary = text, label NULL (the job fills it). Idempotent on ticket_id.

  uv run python scripts/load_tickets.py $R3_ARTIFACTS/datasets/tickets-v1/tickets.jsonl --split test
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--split", default="all", help="test | train | all")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
    rows = [json.loads(line) for line in open(args.path) if line.strip()]
    if args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con, con.cursor() as cur:
        cur.executemany(
            "insert into tickets (ticket_id, customer_id, category, summary, label, label_source, created_at, idempotency_key) "
            "values (%s, null, %s, %s, null, null, now(), %s) on conflict (ticket_id) do nothing",
            [(r["ticket_id"], r["label"], r["text"], f"load:{r['ticket_id']}") for r in rows])
        n = con.execute("select count(*), count(label) from tickets").fetchone()
    print(f"loaded {len(rows)} rows from {os.path.basename(args.path)} (split={args.split}); tickets table now {n[0]} rows, {n[1]} labeled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
