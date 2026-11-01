"""Scores one demo call (simulated or live) against its scenario in demo/scenarios.md: the final state from sessions.state
(services/agent/state.py: completed writes, handoff) plus the call's turn events (assistant transcripts, tool results,
interruptions). Returns the reasons it failed; an empty list is a pass. Used by scripts/demo_check.py in the rehearsal."""
from __future__ import annotations

import re
from typing import Any

from services.common.claims import unsupported_claims

SCENARIOS = ("ordinary", "correction", "timeout", "interruption", "handoff", "handoff_person")
TIMEOUT_TOOL = "book_appointment"  # the write the final model actually calls on the booking path (it books directly when the caller names the time)
DANA, MARCUS = "C-1001", "C-1002"
DANA_BOOKING = {"customer_id": DANA, "branch_id": "market_st", "service_type": "mortgage_consultation", "slot_start": "2026-11-03T09:30"}
ID_RE = re.compile(r"\b[CAT]-\d{3,}\b")  # customer, appointment and ticket ids are never spoken
FALSE_PROMISE_RE = re.compile(r"\b(reversed|blocked|refund(?:ed)?|money back|cancell?ed (?:your|the) card|disputed? (?:the|that) charge)\b", re.I)


def _assistant(events) -> list[str]:
    return [e[5].get("text") or "" for e in events if e[3] == "transcript" and (e[5] or {}).get("role") == "assistant"]


def _writes(state: dict[str, Any], tool: str, status: str) -> list[dict[str, Any]]:
    return [c for c in state.get("completed") or [] if c.get("tool") == tool and (c.get("result") or {}).get("status") == status]


def check_call(scenario: str, state: dict[str, Any], events) -> list[str]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; one of {SCENARIOS}")
    reasons: list[str] = []
    texts = _assistant(events)
    if any(e[3] == "transcript" and (e[5] or {}).get("role") == "assistant" and e[5].get("structural") for e in events):
        reasons.append("structural text spoken")
    n = unsupported_claims(events)
    if n:
        reasons.append(f"{n} success claim(s) with no matching tool result")
    ids = sorted({m.group(0) for t in texts for m in ID_RE.finditer(t)})
    if ids:
        reasons.append("spoke ids: " + ", ".join(ids))
    bookings = _writes(state, "book_appointment", "booked")
    tickets = _writes(state, "create_ticket", "open")
    handoff = state.get("handoff") or {}

    if scenario in ("ordinary", "interruption", "timeout"):
        if scenario == "timeout":
            checks = [e[5] for e in sorted(events, key=lambda e: e[4]) if e[3] == "tool_end" and (e[5] or {}).get("name") == TIMEOUT_TOOL]
            if not checks or checks[0].get("ok"):
                reasons.append(f"the first {TIMEOUT_TOOL} did not time out (fault not injected)")
            elif len(checks) < 2:  # one failed attempt and no retry is not recovery, whatever the call ended in
                reasons.append(f"no retry of {TIMEOUT_TOOL} after the timeout")
                return reasons
            elif not any(c.get("ok") for c in checks[1:]):
                # both attempts failed: the contract's path is a handoff with no booking
                if not handoff.get("requested") or bookings:
                    reasons.append("no successful retry and no clean handoff")
                return reasons
        if len(bookings) != 1:
            reasons.append(f"{len(bookings)} bookings, expected 1")
        else:
            got = {k: bookings[0].get("args", {}).get(k) for k in DANA_BOOKING}
            if got != DANA_BOOKING:
                reasons.append(f"booking {got} != {DANA_BOOKING}")
        if tickets:
            reasons.append("a ticket was created")
        if handoff.get("requested"):
            reasons.append("handoff requested")
        if scenario == "interruption" and not any(e[3] in ("interruption", "caller_interrupted") for e in events):
            reasons.append("no interruption happened (the agent's reply was too short to cut into?)")
    elif scenario == "correction":
        if len(bookings) != 1:
            reasons.append(f"{len(bookings)} bookings, expected 1")
        else:
            args = bookings[0].get("args", {})
            slot = str(args.get("slot_start") or "")
            if not slot.startswith("2026-11-04"):
                reasons.append(f"booked {slot or 'nothing'}: stale detail, expected Wednesday 2026-11-04 after the correction")
            elif slot != "2026-11-04T09:30" or args.get("branch_id") != "market_st" or args.get("service_type") != "mortgage_consultation":
                reasons.append(f"booking {slot} at {args.get('branch_id')} for {args.get('service_type')}, expected 2026-11-04T09:30 at market_st, mortgage_consultation")
        if tickets:
            reasons.append("a ticket was created")
        if handoff.get("requested"):
            reasons.append("handoff requested")
    elif scenario == "handoff":
        good = [t for t in tickets if t.get("args", {}).get("category") == "unauthorized_transaction" and t.get("args", {}).get("customer_id") == MARCUS]
        if len(tickets) != 1 or len(good) != 1:
            reasons.append(f"tickets {[(t.get('args', {}).get('category'), t.get('args', {}).get('customer_id')) for t in tickets]}: expected one unauthorized_transaction for {MARCUS}")
        if not handoff.get("requested"):
            reasons.append("no handoff requested")
        elif handoff.get("reason") != "suspected_fraud":
            reasons.append(f"handoff reason {handoff.get('reason')!r}, expected suspected_fraud")
        if bookings:
            reasons.append("an appointment was booked")
        bad = sorted({m.group(0).lower() for t in texts for m in FALSE_PROMISE_RE.finditer(t)})
        if bad:
            reasons.append("promised what no tool does: " + ", ".join(bad))
    elif scenario == "handoff_person":  # the caller asks for a person twice: transfer for that reason, nothing written
        if not handoff.get("requested"):
            reasons.append("no handoff requested")
        elif handoff.get("reason") != "caller_requested":
            reasons.append(f"handoff reason {handoff.get('reason')!r}, expected caller_requested")
        if tickets:
            reasons.append("a ticket was created")
        if bookings:
            reasons.append("an appointment was booked")
    return reasons


def unfinished(rows) -> list[str]:
    """call ids whose session has no ended_at yet: the agent writes the final state once, at call end, so a checker that reads
    earlier scores the opening checkpoint (Nov 1 08:26: "0 bookings" for a booked call, read 3 s before the state landed)."""
    return [r[0] for r in rows if r[1] is None]


def call_summary(events) -> dict[str, Any]:
    """Per-call facts reported beside the verdict: loop-guard and claim-guard interventions and executor retries (the agent's, not the
    model's) and the TTS vendor."""
    end = next((e[5] for e in events if e[3] == "call_end"), {}) or {}
    return {"guard": sum(1 for e in events if e[3] == "loop_guard"), "claim_guard": sum(1 for e in events if e[3] == "claim_guard"),
            "retry": sum(1 for e in events if e[3] == "tool_retry"), "policy": sum(1 for e in events if e[3] == "policy_handoff"), "reprompt": sum(1 for e in events if e[3] == "stt_timeout_reprompt"),
            "tts": str(end.get("tts_vendor") or "?")}
