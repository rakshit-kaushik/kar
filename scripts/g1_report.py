#!/usr/bin/env python
"""G1 report: per-stage latency p50/p95 and per-call outcomes from turn_events (CONTRACT.md §6).

Per user turn (anchored on the agent's speech_end event):
  eos->stt_final            agent stt_final ts - speech_end ts
  eos->first_audio          agent_speaking ts - speech_end ts   (playback start = first audible audio)
  llm_ttft (proxy)          first llm_first_token payload.ms after speech_end
  llm_ttft (livekit)        llm_metrics payload.ttft
  tts_ttfb                  tts_first_audio payload.ttfb
  prompt/completion tokens  proxy llm_done payload
Per call: lookup verified, structural leaks, turns, outcome.

Usage: uv run python scripts/g1_report.py [--like 'r3-call-%'] [--since '2026-11-01 00:00'] [--ledger]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import statistics
from collections import defaultdict

import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt(xs, unit="ms"):
    if not xs:
        return "n/a"
    return f"p50 {pct(xs, 0.5):.0f} {unit} · p95 {pct(xs, 0.95):.0f} {unit} · n={len(xs)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", default="r3-call-%")
    ap.add_argument("--since", default=None)
    ap.add_argument("--ledger", action="store_true", help="write baseline rows to cost_ledger")
    ap.add_argument("--run-id", default="g1-base")
    args = ap.parse_args()

    q = "select call_id, turn_idx, model_id, event, ts_ms, payload from turn_events where call_id like %s"
    params: list = [args.like]
    if args.since:
        q += " and wall_ts >= %s"
        params.append(args.since)
    q += " order by call_id, ts_ms, id"
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        rows = con.execute(q, params).fetchall()
        sessions = {r[0]: r for r in con.execute("select call_id, outcome, started_at, ended_at from sessions where call_id like %s", [args.like]).fetchall()}

    by_call: dict[str, list] = defaultdict(list)
    for r in rows:
        by_call[r[0]].append(r)
    if not by_call:
        print("no turn_events match"); return

    eos_stt, eos_audio, ttft_proxy, ttft_lk, tts_ttfb, eou_delay, prompt_tok, compl_tok = ([] for _ in range(8))
    per_call = []
    for call_id, evs in by_call.items():
        model = evs[0][2]
        leaks = sum(1 for e in evs if e[3] == "transcript" and e[5].get("role") == "assistant" and e[5].get("structural"))
        lookup_ok = any(e[3] == "tool_end" and e[5].get("name") == "lookup_account" and e[5].get("ok") for e in evs)
        tools = [e[5].get("name") for e in evs if e[3] == "tool_end" and e[5].get("ok")]
        n_turns = sum(1 for e in evs if e[3] == "speech_end")
        speech_ends = [e[4] for e in evs if e[3] == "speech_end"]
        for i, t in enumerate(speech_ends):
            t_next = speech_ends[i + 1] if i + 1 < len(speech_ends) else t + 30_000
            window = [e for e in evs if t <= e[4] < t_next]
            stt = next((e for e in window if e[3] == "stt_final"), None)
            spk = next((e for e in window if e[3] == "agent_speaking"), None)
            ft = next((e for e in window if e[3] == "llm_first_token"), None)
            if stt:
                eos_stt.append(stt[4] - t)
            if spk:
                eos_audio.append(spk[4] - t)
            if ft:
                ttft_proxy.append(ft[5].get("ms"))
            for e in window:
                if e[3] == "llm_metrics" and e[5].get("ttft") is not None:
                    ttft_lk.append(e[5]["ttft"] * 1000)
                if e[3] == "tts_first_audio" and e[5].get("ttfb") is not None:
                    tts_ttfb.append(e[5]["ttfb"] * 1000)
                if e[3] == "eou_metrics":
                    eou_delay.append(e[5].get("end_of_utterance_delay", 0) * 1000)
                if e[3] == "llm_done":
                    if e[5].get("prompt_tokens"):
                        prompt_tok.append(e[5]["prompt_tokens"])
                    if e[5].get("completion_tokens"):
                        compl_tok.append(e[5]["completion_tokens"])
        s = sessions.get(call_id)
        per_call.append((call_id, model, n_turns, lookup_ok, leaks, ",".join(tools) or "-", s[1] if s else "?"))

    print(f"calls: {len(per_call)}   model: {per_call[0][1]}")
    print(f"{'call_id':<16} {'turns':>5} {'lookup':>6} {'leaks':>5}  outcome    tools")
    for c in per_call:
        print(f"{c[0]:<16} {c[2]:>5} {'ok' if c[3] else 'NO':>6} {c[4]:>5}  {c[6]:<10} {c[5]}")
    print()
    print(f"end of speech -> STT final      {fmt(eos_stt)}")
    print(f"end of speech -> first audio    {fmt(eos_audio)}   (target p50 ≤ 1000, p95 ≤ 1600)")
    print(f"EOU delay (LiveKit)             {fmt(eou_delay)}")
    print(f"LLM TTFT (proxy, per request)   {fmt(ttft_proxy)}   (target p50 ≤ 350 at 4 calls)")
    print(f"LLM TTFT (LiveKit metrics)      {fmt(ttft_lk)}")
    print(f"TTS first byte                  {fmt(tts_ttfb)}   (target ≤ 200)")
    print(f"prompt tokens per request       {fmt(prompt_tok, 'tok')}")
    print(f"completion tokens per request   {fmt(compl_tok, 'tok')}")
    ok_calls = sum(1 for c in per_call if c[3])
    leak_calls = sum(1 for c in per_call if c[4])
    print(f"\nG1: lookups completed {ok_calls}/{len(per_call)}; calls with structural speech {leak_calls}; "
          f"{'PASS' if ok_calls >= 10 and leak_calls == 0 else 'not yet'}")

    if args.ledger:
        items = [("g1.calls", len(per_call), "calls"), ("g1.lookups_ok", ok_calls, "calls"), ("g1.structural_leak_calls", leak_calls, "calls")]
        for name, xs, unit in (("eos_to_stt_final", eos_stt, "ms"), ("eos_to_first_audio", eos_audio, "ms"), ("llm_ttft_proxy", ttft_proxy, "ms"), ("tts_ttfb", tts_ttfb, "ms"), ("prompt_tokens", prompt_tok, "tok"), ("completion_tokens", compl_tok, "tok")):
            if xs:
                items += [(f"g1.{name}.p50", pct(xs, 0.5), unit), (f"g1.{name}.p95", pct(xs, 0.95), unit)]
        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
            con.cursor().executemany(
                "insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,%s,%s,%s,true,%s)",
                [(args.run_id, per_call[0][1], item, qty, unit, f"live browser calls matching {args.like}") for item, qty, unit in items],
            )
        print(f"ledger: {len(items)} rows under {args.run_id}")


if __name__ == "__main__":
    main()
