"""Fault injection for live demo calls (CONTRACT §exceptions: tool timeout, unavailable, conflict, not_found).

The caller's room name carries the spec, so no code or config changes between demo scenarios:
  <room>--fault-<tool>-<error>[-<call_index>]   (repeatable)
  e.g. demo-timeout--fault-check_availability-timeout      -> the first check_availability call times out
       demo-x--fault-book_appointment-unavailable-2        -> the second book_appointment call fails
"""
from __future__ import annotations

import re

TOOLS = ("lookup_account", "check_availability", "book_appointment", "reschedule_appointment", "cancel_appointment", "create_ticket", "request_handoff")
ERRORS = ("timeout", "unavailable", "conflict", "not_found", "not_verified")
_SPEC = re.compile(r"--fault-([a-z_]+)-([a-z_]+?)(?:-(\d+))?(?=--fault-|$)")


def faults_from_room_name(room_name: str) -> list[dict]:
    out = []
    for tool, error, idx in _SPEC.findall(room_name):
        if tool in TOOLS and error in ERRORS:
            out.append({"tool": tool, "call_index": int(idx) if idx else 1, "error": error})
    return out
