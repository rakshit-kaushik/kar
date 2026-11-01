#!/usr/bin/env python3
"""Relabel the model id on one block's rows (sessions, turn_events, cost_ledger) when the stack was restarted under the wrong
served name (Nov 1 07:11: run 1's weights served and logged as r3-voice-base). Restricted to call-id prefixes and ledger run ids,
so genuine rows of the other model keep their label. Dry run by default; --apply writes in one transaction.

  uv run python scripts/relabel_model.py --from r3-voice-base --to r3-voice-206506525c --call-like 'r3-filler-0711-%' --call-like 'demo-%-filler-0711%' \
      --run-id g1-sim-r3-voice-base --run-id cost-r3-voice-base [--since '2026-11-01 07:11'] [--apply]
"""
from __future__ import annotations

import argparse
import os

import psycopg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--call-like", action="append", default=[], help="call_id patterns (sessions + turn_events)")
    ap.add_argument("--run-id", action="append", default=[], help="cost_ledger run ids to relabel; the model name inside run_id and note is rewritten too")
    ap.add_argument("--since", default=None, help="only rows created at or after this timestamp (ledger created_at, sessions started_at, turn_events via their session)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not a.call_like and not a.run_id:
        ap.error("give at least one --call-like or --run-id")
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        cur = con.cursor()
        total = 0
        for like in a.call_like:
            since_s = " and started_at >= %s" if a.since else ""
            params = [like, a.src] + ([a.since] if a.since else [])
            calls = [r[0] for r in cur.execute(f"select call_id from sessions where call_id like %s and model_id = %s{since_s}", params).fetchall()]
            n_ev = cur.execute("select count(*) from turn_events where call_id = any(%s) and model_id = %s", (calls, a.src)).fetchone()[0] if calls else 0
            print(f"{like}: {len(calls)} sessions, {n_ev} turn_events labeled {a.src}")
            if a.apply and calls:
                cur.execute("update sessions set model_id = %s where call_id = any(%s) and model_id = %s", (a.dst, calls, a.src))
                cur.execute("update turn_events set model_id = %s where call_id = any(%s) and model_id = %s", (a.dst, calls, a.src))
            total += len(calls) + n_ev
        for run_id in a.run_id:
            since_s = " and created_at >= %s" if a.since else ""
            params = [run_id] + ([a.since] if a.since else [])
            n = cur.execute(f"select count(*) from cost_ledger where run_id = %s{since_s}", params).fetchone()[0]
            new_run = run_id.replace(a.src, a.dst)
            print(f"{run_id}: {n} ledger rows -> run_id {new_run}, model_id {a.dst}")
            if a.apply and n:
                cur.execute(f"update cost_ledger set run_id = %s, model_id = %s, note = replace(coalesce(note, ''), %s, %s) where run_id = %s{since_s}",
                            [new_run, a.dst, a.src, a.dst, run_id] + ([a.since] if a.since else []))
            total += n
        if a.apply:
            con.commit()
            print(f"applied: {total} rows relabeled {a.src} -> {a.dst}")
        else:
            con.rollback()
            print(f"dry run: {total} rows would be relabeled; add --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
