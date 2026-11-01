#!/usr/bin/env python
"""Score demo calls against their scenario (demo/scenarios.md) from Postgres: sessions.state + turn_events per call.

Usage: uv run python scripts/demo_check.py --like 'demo-ordinary-1101-0530%' --scenario ordinary [--ledger --run-id rehearsal-<model>-<stamp>]
Prints PASS/FAIL with reasons per call and the pass count; --ledger writes rehearsal.<scenario>.pass / .runs rows."""
from __future__ import annotations

import argparse
import os
import time
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import psycopg  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from services.demo.check import SCENARIOS, call_summary, check_call, unfinished  # noqa: E402

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, help="call_id pattern, e.g. 'demo-ordinary-1101-0530%'")
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        # the final state lands at call end: wait for every matched call to end before scoring (services.demo.check.unfinished)
        for _ in range(int(os.environ.get("DEMO_CHECK_WAIT_S", "45"))):
            pending = unfinished(con.execute("select call_id, ended_at from sessions where call_id like %s", (args.like,)).fetchall())
            if not pending:
                break
            time.sleep(1)
        else:
            print(f"warning: {len(pending)} call(s) still open after the wait, scored on their opening checkpoint: {', '.join(pending)}")
        calls = con.execute("select call_id, model_id, outcome, state from sessions where call_id like %s order by started_at", (args.like,)).fetchall()
        if not calls:
            print(f"no sessions match {args.like}")
            return 1
        passed, guarded, claimed, retried, policied, reprompted, vendors, model = 0, 0, 0, 0, 0, 0, set(), calls[0][1]
        print(f"scenario {args.scenario}   model {model}   calls {len(calls)}")
        for call_id, _, outcome, state in calls:
            evs = con.execute("select call_id, turn_idx, model_id, event, ts_ms, payload from turn_events where call_id = %s order by ts_ms, id", (call_id,)).fetchall()
            reasons = check_call(args.scenario, state or {}, evs)
            info = call_summary(evs)
            passed += not reasons
            guarded += bool(info["guard"])
            claimed += bool(info["claim_guard"])
            retried += bool(info["retry"])
            policied += bool(info["policy"])
            reprompted += bool(info["reprompt"])
            vendors.add(info["tts"])
            print(f"  {call_id:<42} {str(outcome):<10} guard={info['guard']} claim={info['claim_guard']} retry={info['retry']} policy={info['policy']} reprompt={info['reprompt']} tts={info['tts']:<9} {'PASS' if not reasons else 'FAIL: ' + '; '.join(reasons)}")
        print(f"{args.scenario}: {passed}/{len(calls)} passed; loop guard intervened in {guarded} call(s); claim guard in {claimed}; executor retried in {retried}; policy handoff in {policied}; re-prompt after untranscribed speech in {reprompted}; tts {'/'.join(sorted(vendors))}")
        if args.ledger:
            run_id = args.run_id or f"rehearsal-{model}"
            note = f"tts={'/'.join(sorted(vendors))}; scripts/demo_check.py --like {args.like!r}"
            rows = ((f"rehearsal.{args.scenario}.pass", passed), (f"rehearsal.{args.scenario}.runs", len(calls)), (f"rehearsal.{args.scenario}.guarded", guarded), (f"rehearsal.{args.scenario}.claim_guarded", claimed), (f"rehearsal.{args.scenario}.retried", retried), (f"rehearsal.{args.scenario}.policy", policied), (f"rehearsal.{args.scenario}.reprompted", reprompted))
            for item, qty in rows:
                con.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,%s,%s,%s,true,%s)",
                            (run_id, model, item, qty, "calls", note))
            con.commit()
            print(f"ledger: {len(rows)} rows under {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
