"""Argument normalization so scorers compare meaning, not formatting."""

from __future__ import annotations

import re
from typing import Any

from eval.sandbox.store import digits, parse_date, parse_slot

DATE_KEYS = {"date", "date_of_birth"}
SLOT_KEYS = {"slot_start", "new_slot_start"}
ID_KEYS = {"customer_id", "appointment_id", "ticket_id"}
ENUM_KEYS = {"branch_id", "service_type", "reason", "category", "replacement"}
WRITE_TOOLS = (
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "block_card",
    "create_ticket",
)


def norm_time(s: str) -> str:
    """'14:30', '2:30 pm', '2:30pm', '14:30:00' -> 'HH:MM'."""
    t = s.strip().lower()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?", t)
    if not m:
        return t
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return f"{h:02d}:{mi:02d}"


def norm_value(key: str, val: Any) -> Any:
    if val is None:
        return None
    s = str(val).strip()
    if key == "phone":
        return digits(s)
    if key == "card_last4":
        return digits(s)[-4:]
    if key in DATE_KEYS:
        d = parse_date(s)
        return d.isoformat() if d else s.lower()
    if key in SLOT_KEYS:
        dt = parse_slot(s)
        return dt.strftime("%Y-%m-%dT%H:%M") if dt else s.lower()
    if key == "time":
        return norm_time(s)
    if key in ID_KEYS:
        return s.upper()
    if key in ENUM_KEYS:
        return s.lower()
    return re.sub(r"\s+", " ", s.lower())


def normalize_args(args: Any) -> dict:
    if not isinstance(args, dict):
        return {}
    return {k: norm_value(k, v) for k, v in args.items() if k != "idempotency_key"}


def slot_of(args: dict) -> str | None:
    for k in ("slot_start", "new_slot_start"):
        if args.get(k):
            return norm_value(k, args[k])
    return None


def args_match(expected: dict, actual: Any) -> list[str]:
    """Keys in `expected` whose normalized value differs from `actual`. [] means match.

    `date` / `time` in expected may stand for the date / time part of the call's slot.
    """
    if not isinstance(actual, dict):
        return list(expected)
    bad = []
    for k, want in expected.items():
        if k in ("date", "time") and k not in actual and slot_of(actual):
            slot = slot_of(actual)
            got = slot[:10] if k == "date" else slot[11:16]
            if norm_value(k, want) != got:
                bad.append(k)
            continue
        if norm_value(k, want) != norm_value(k, actual.get(k)):
            bad.append(k)
    return bad
