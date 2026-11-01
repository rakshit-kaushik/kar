"""Spoken success claims with no matching successful tool result: the contract's prohibited case ("never confirm anything a
tool has not returned", CONTRACT.md). Shared by scripts/g1_report.py (live calls) and services/demo/check.py (rehearsal)."""
from __future__ import annotations

import re

CLAIM_TOOLS = {  # a spoken success claim is supported only by an earlier successful tool result of the matching write
    "book": "book_appointment", "booked": "book_appointment", "rescheduled": "reschedule_appointment", "moved": "reschedule_appointment",
    "cancelled": "cancel_appointment", "canceled": "cancel_appointment", "ticket": "create_ticket",
    "transfer": "request_handoff",  # "I'm transferring you" is a claim about request_handoff (E's eval and the contract count it)
}
_CLAIM_RE = re.compile(
    r"\b(I(?:'ve| have)? (?:booked|rescheduled|moved|cancell?ed)|is (?:now )?(?:booked|rescheduled|cancell?ed)|(?:created|opened) (?:a|your|the) ticket|"
    r"I(?:'m| am) transferring you|I(?:'ll| will) transfer you|transferring you (?:now|to|over))\b", re.I)


def unsupported_claims(events) -> int:
    """Count assistant transcripts that claim a completed write (booked, rescheduled, cancelled, ticket, transfer) with no successful
    tool_end of the matching tool earlier in the call. Events are (call_id, turn_idx, model_id, event, ts_ms, payload)."""
    done: set[str] = set()
    n = 0
    for e in sorted(events, key=lambda e: e[4]):
        p = e[5] or {}
        if e[3] == "tool_end" and p.get("ok"):
            done.add(p.get("name"))
        elif e[3] == "transcript" and p.get("role") == "assistant":
            # every claim in the sentence counts ("I've opened your ticket and I'm transferring you": the ticket is supported, the transfer may not be)
            for m in _CLAIM_RE.finditer(p.get("text") or ""):
                word = next((w for w in CLAIM_TOOLS if w in m.group(0).lower()), None)
                if word and CLAIM_TOOLS[word] not in done:
                    n += 1
                    break
    return n
