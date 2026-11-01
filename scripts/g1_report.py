#!/usr/bin/env python
"""G1 / load-test report: per-stage latency p50/p95 and per-call outcomes from turn_events (CONTRACT.md §6).

Turns are anchored on the caller's end of speech: `caller_speech_end` for simulated calls (scripts/sim_caller.py),
the agent's `speech_end` for real-mic calls. Per turn, measured from that anchor:
  turn-taking overhead      first proxy llm_request          (VAD silence + STT final + endpointing)
  LLM TTFT / total          proxy llm_first_token.ms, llm_done.ms; source local|hosted
  first audible TTS frame   agent tts_first_audible          (agent clock: audio handed to the room)
  first audio heard         caller_first_audio               (caller clock: the number the plan budgets)
  transport                 heard - audible                  (room, SFU, jitter buffer)
Per call: lookup verified, structural leaks, turns, fallbacks, outcome.

Usage: uv run python scripts/g1_report.py [--like 'r3-sim-%'] [--since '2026-11-01 00:00'] [--ledger --run-id g1-sim-base]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re

from services.common.claims import CLAIM_TOOLS, unsupported_claims  # noqa: E402,F401  (shared with services/demo/check.py)
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


def filler_timing(turn_events, anchor):
    """For a turn that spoke a filler before a tool call: (eos -> filler audible, eos -> first audible reply) in ms on the agent
    clock, each None when that audio never came; (None, None) for turns without a filler_start event."""
    if not any(e[3] == "filler_start" for e in turn_events):
        return (None, None)
    audible = sorted(e[4] for e in turn_events if e[3] == "tts_first_audible")
    return (audible[0] - anchor if audible else None, audible[1] - anchor if len(audible) > 1 else None)


def turn_kind(llm_done_payload: dict) -> str:
    """'tool' when the model's reply was a tool call (vllm-mlx streams it only when complete, so its first token is the whole
    turn and the caller hears nothing until the tool ran and a second turn spoke), else 'spoken'."""
    return "tool" if (llm_done_payload or {}).get("tool_calls") else "spoken"


def decode_rate(completion_tokens, first_ms, done_ms, min_window_ms: int = 250) -> float | None:
    """Tokens after the first over the streaming window, tok/s. None when the window is too short to measure: a
    tool-call turn (vllm-mlx emits the call only when complete, so first token = done) or a one-token reply."""
    if not completion_tokens or completion_tokens < 2 or first_ms is None or done_ms is None:
        return None
    window = done_ms - first_ms
    if window < min_window_ms:
        return None
    return (completion_tokens - 1) / (window / 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", default="r3-call-%")
    ap.add_argument("--since", default=None, help="wall_ts lower bound, e.g. '2026-11-01 02:00'")
    ap.add_argument("--ledger", action="store_true", help="write summary rows to cost_ledger")
    ap.add_argument("--run-id", default="g1-base")
    ap.add_argument("--note", default=None, help="ledger note (default describes the call filter)")
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

    M: dict[str, list] = defaultdict(list)  # metric name -> samples
    per_call = []
    simulated = 0
    filler_turns = 0
    tts_vendors: set[str] = set()
    for call_id, evs in by_call.items():
        model = evs[0][2]
        leaks = sum(1 for e in evs if e[3] == "transcript" and e[5].get("role") == "assistant" and e[5].get("structural"))
        lookup_ok = any(e[3] == "tool_end" and e[5].get("name") == "lookup_account" and e[5].get("ok") for e in evs)
        tools = [e[5].get("name") for e in evs if e[3] == "tool_end" and e[5].get("ok")]
        fallbacks = sum(1 for e in evs if e[3] == "llm_first_token" and e[5].get("source") == "hosted")
        claims = unsupported_claims(evs)
        tts_vendors.update(str(e[5].get("tts_vendor")) for e in evs if e[3] == "call_end" and e[5].get("tts_vendor"))  # turns served by the hosted model (a hedge the local side won is not one)
        joined = next((e[5].get("ms") for e in evs if e[3] == "caller_agent_joined"), None)
        if joined is not None:
            M["agent_dispatch"].append(joined); simulated += 1
        anchors = [e[4] for e in evs if e[3] == "caller_speech_end"] or [e[4] for e in evs if e[3] == "speech_end"]
        n_turns = len(anchors)
        for i, t in enumerate(anchors):
            t_next = anchors[i + 1] if i + 1 < len(anchors) else t + 60_000
            window = [e for e in evs if t <= e[4] < t_next]
            def first(name, key=None):
                for e in window:
                    if e[3] == name:
                        return e[4] if key is None else e[5].get(key)
                return None
            req, ft, done = first("llm_request"), first("llm_first_token"), first("llm_done")
            audible, heard, spk = first("tts_first_audible"), first("caller_first_audio"), first("agent_speaking")
            if req: M["turn_taking_overhead"].append(req - t)
            if ft:
                M["llm_ttft"].append(first("llm_first_token", "ms")); M[f"llm_ttft_{first('llm_first_token', 'source') or 'local'}"].append(first("llm_first_token", "ms"))
                done_payload = next((e[5] for e in window if e[3] == "llm_done"), {})
                M[f"llm_ttft_{turn_kind(done_payload)}"].append(first("llm_first_token", "ms"))
            if done and first("llm_done", "completion_tokens") and first("llm_first_token", "ms") is not None:
                ct = first("llm_done", "completion_tokens"); total = first("llm_done", "ms"); f0 = first("llm_first_token", "ms")
                rate = decode_rate(ct, f0, total)
                if rate is not None: M["decode_tok_s"].append(rate)
            if first("llm_done", "prompt_tokens"): M["prompt_tokens"].append(first("llm_done", "prompt_tokens"))
            if first("llm_done", "completion_tokens"): M["completion_tokens"].append(first("llm_done", "completion_tokens"))
            if audible: M["eos_to_tts_audible"].append(audible - t)
            if heard: M["eos_to_first_audio_heard"].append(heard - t)
            if heard and audible: M["transport"].append(heard - audible)
            if spk and not heard: M["eos_to_agent_speaking"].append(spk - t)
            if any(e[3] == "filler_start" for e in window):
                filler_turns += 1
                f_aud, r_aud = filler_timing(window, t)
                if f_aud is not None: M["eos_to_filler_audible"].append(f_aud)
                if r_aud is not None: M["eos_to_reply_audible"].append(r_aud)
            for e in window:
                if e[3] == "eou_metrics":
                    if e[5].get("end_of_utterance_delay") is not None: M["eou_delay"].append(e[5]["end_of_utterance_delay"] * 1000)
                    if e[5].get("transcription_delay") is not None: M["transcription_delay"].append(e[5]["transcription_delay"] * 1000)
                if e[3] == "tts_first_audio" and e[5].get("ttfb") is not None: M["tts_ttfb"].append(e[5]["ttfb"] * 1000)
                if e[3] == "stt_final" and not first("caller_speech_end"): pass
        s = sessions.get(call_id)
        per_call.append((call_id, model, n_turns, lookup_ok, leaks, fallbacks, ",".join(tools) or "-", s[1] if s else "?", claims))

    kind = "simulated audio calls" if simulated == len(per_call) else ("live browser calls" if simulated == 0 else "mixed live/simulated calls")
    print(f"calls: {len(per_call)} ({kind})   model: {per_call[0][1]}")
    print(f"{'call_id':<24} {'turns':>5} {'lookup':>6} {'leaks':>5} {'fallbk':>6}  outcome    tools")
    for c in per_call:
        print(f"{c[0]:<24} {c[2]:>5} {'ok' if c[3] else 'NO':>6} {c[4]:>5} {c[5]:>6}  {str(c[7]):<10} {c[6]}")
    print()
    print(f"end of speech -> first audio heard   {fmt(M['eos_to_first_audio_heard'])}   (target p50 ≤ 1000, p95 ≤ 1600)")
    print(f"end of speech -> first audible TTS   {fmt(M['eos_to_tts_audible'])}   (agent clock)")
    if filler_turns:
        print(f"  filler before tool calls           {filler_turns} turns · filler audible {fmt(M['eos_to_filler_audible'])} · first reply audible {fmt(M['eos_to_reply_audible'])}   (agent clock; 'first audio heard' includes the filler)")
    print(f"transport: audible -> heard          {fmt(M['transport'])}")
    print(f"turn-taking overhead -> LLM request  {fmt(M['turn_taking_overhead'])}   (VAD + STT final + endpointing)")
    print(f"  EOU delay (LiveKit)                {fmt(M['eou_delay'])}   STT final after VAD end {fmt(M['transcription_delay'])}")
    print(f"LLM TTFT (proxy)                     {fmt(M['llm_ttft'])}   local {fmt(M['llm_ttft_local'])}   hosted {fmt(M['llm_ttft_hosted'])}")
    print(f"  by turn type                       tool-call turns {fmt(M['llm_ttft_tool'])}   spoken turns {fmt(M['llm_ttft_spoken'])}   (tool-call turns: first token = whole call)")
    print(f"LLM decode                           {fmt(M['decode_tok_s'], 'tok/s')}")
    print(f"TTS first byte ({'/'.join(sorted(tts_vendors)) or 'vendor n/a':<9})   {fmt(M['tts_ttfb'])}   (target ≤ 200)")
    print(f"prompt / completion tokens           {fmt(M['prompt_tokens'], 'tok')} / {fmt(M['completion_tokens'], 'tok')}")
    if M["agent_dispatch"]:
        print(f"agent joined after caller            {fmt(M['agent_dispatch'])}")
    ok_calls = sum(1 for c in per_call if c[3]); leak_calls = sum(1 for c in per_call if c[4]); fb_total = sum(c[5] for c in per_call)
    claim_calls = sum(1 for c in per_call if c[8])
    print(f"\nG1: lookups completed {ok_calls}/{len(per_call)}; calls with structural speech {leak_calls}; hosted fallbacks {fb_total}; "
          f"unsupported success claims {claim_calls} calls; {'PASS' if ok_calls >= 10 and leak_calls == 0 and claim_calls == 0 else 'not yet'}")

    if args.ledger:
        note = args.note or f"{kind} matching {args.like}"
        items = [("calls", len(per_call), "calls"), ("lookups_ok", ok_calls, "calls"), ("structural_leak_calls", leak_calls, "calls"), ("unsupported_claim_calls", claim_calls, "calls"), ("filler_turns", filler_turns, "turns"), ("fallbacks", fb_total, "requests")]
        for name in ("eos_to_first_audio_heard", "eos_to_tts_audible", "eos_to_filler_audible", "eos_to_reply_audible", "transport", "turn_taking_overhead", "eou_delay", "transcription_delay", "llm_ttft_local", "llm_ttft_hosted", "llm_ttft_tool", "llm_ttft_spoken", "decode_tok_s", "tts_ttfb", "prompt_tokens", "completion_tokens", "agent_dispatch"):
            xs = M.get(name) or []
            unit = "tok/s" if name == "decode_tok_s" else ("tok" if "tokens" in name else "ms")
            if xs:
                items += [(f"{name}.p50", pct(xs, 0.5), unit), (f"{name}.p95", pct(xs, 0.95), unit), (f"{name}.n", len(xs), "samples")]
        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
            con.cursor().executemany(
                "insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,%s,%s,%s,true,%s)",
                [(args.run_id, per_call[0][1], f"g1.{item}", qty, unit, note) for item, qty, unit in items],
            )
        print(f"ledger: {len(items)} rows under {args.run_id}")


if __name__ == "__main__":
    main()
