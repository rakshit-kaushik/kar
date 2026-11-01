"""The contract tools, implemented per eval/sandbox/seed.json["rules"] against a `Sandbox`.

Business errors use the codes in CONTRACT §4; malformed calls (missing required argument, bad
enum, unparseable date) return `invalid_argument`. Idempotency replay and fault injection live in
`Sandbox.call`. Interpretations match services/tools/sandbox.py (session S) so the live agent and
the eval see identical behaviour: ids A-3001+/T-5001+, cancelled appointments are `not_found`,
an unknown customer on a write is `not_found`, a bad handoff reason becomes `out_of_scope`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from . import seed as S
from .store import (
    HANDOFF_REASONS,
    HandoffRecord,
    digits,
    err,
    parse_date,
    parse_slot,
    validate_args,
)

if TYPE_CHECKING:
    from .store import Sandbox

HANDLERS: dict[str, Callable] = {}


def tool(name: str):
    def deco(fn):
        HANDLERS[name] = fn
        return fn

    return deco


def _appointment_view(a: S.Appointment) -> dict:
    return {
        "appointment_id": a.appointment_id,
        "branch_id": a.branch_id,
        "service_type": a.service_type,
        "slot_start": a.slot,
    }


@tool("lookup_account")
def lookup_account(sb: Sandbox, args: dict) -> dict:
    cust = sb.find_customer_by_phone(args.get("phone"))
    if cust is None:
        return err("not_found", "no account matches that phone number")
    dob = parse_date(args.get("date_of_birth"))
    if dob is None or dob != cust.date_of_birth:
        return err("not_verified", "date of birth does not match")
    return {
        "verified": True,
        "customer_id": cust.customer_id,
        "first_name": cust.first_name,
        "branch_id": cust.branch_id,
        "appointments": [_appointment_view(a) for a in sb.customer_appointments(cust.customer_id)],
        "cards": [
            {"card_last4": k.card_last4, "status": k.status}
            for k in sb.customer_cards(cust.customer_id)
        ],
    }


@tool("check_availability")
def check_availability(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("check_availability", args)
    if bad:
        return bad
    day = parse_date(args.get("date"))
    if (
        args["branch_id"] not in sb.branches
        or args["service_type"] not in sb.services
        or day is None
    ):
        return err("invalid_argument", "branch_id, service_type and date (YYYY-MM-DD) are required")
    if day < sb.today or sb.closed(day):
        return err("unavailable", "the branch is closed that day")
    return {
        "branch_id": args["branch_id"],
        "date": day.isoformat(),
        "slots": sb.open_slots(args["branch_id"], day),
    }


@tool("book_appointment")
def book_appointment(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("book_appointment", args)
    if bad:
        return bad
    cust = sb.find_customer(args["customer_id"])
    if cust is None:
        return err("not_found", "customer not found")
    when = parse_slot(args["slot_start"])
    if (
        args["branch_id"] not in sb.branches
        or args["service_type"] not in sb.services
        or when is None
    ):
        return err(
            "invalid_argument",
            "branch_id, service_type and slot_start (YYYY-MM-DDTHH:MM) are required",
        )
    dup = next(
        (
            a
            for a in sb.customer_appointments(cust.customer_id)
            if a.service_type == args["service_type"] and a.slot_start.date() == when.date()
        ),
        None,
    )
    if dup:
        return err(
            "conflict",
            f"already has {args['service_type']} on {dup.slot[:10]} at {dup.slot[11:]} ({dup.appointment_id})",
        )
    if not sb.slot_is_open(args["branch_id"], when):
        return err("unavailable", "that slot is not open")
    ap = S.Appointment(
        sb.next_id("A"),
        cust.customer_id,
        args["branch_id"],
        args["service_type"],
        when,
        "booked",
        args.get("idempotency_key"),
    )
    sb.appointments.append(ap)
    return {**_appointment_view(ap), "status": "booked"}


@tool("reschedule_appointment")
def reschedule_appointment(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("reschedule_appointment", args)
    if bad:
        return bad
    ap = sb.find_appointment(args["appointment_id"])
    if ap is None or ap.status != "booked":
        return err("not_found", "no booked appointment with that id")
    when = parse_slot(args["new_slot_start"])
    if when is None:
        return err("invalid_argument", "new_slot_start must be YYYY-MM-DDTHH:MM")
    if when == ap.slot_start:
        return err("conflict", "the appointment is already at that time")
    dup = next(
        (
            a
            for a in sb.customer_appointments(ap.customer_id)
            if a.appointment_id != ap.appointment_id
            and a.service_type == ap.service_type
            and a.slot_start.date() == when.date()
        ),
        None,
    )
    if dup:
        return err(
            "conflict", f"already has {ap.service_type} on {dup.slot[:10]} ({dup.appointment_id})"
        )
    if not sb.slot_is_open(ap.branch_id, when, ignore_appointment=ap.appointment_id):
        return err("unavailable", "that slot is not open")
    ap.slot_start = when
    return {**_appointment_view(ap), "status": "booked"}


@tool("cancel_appointment")
def cancel_appointment(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("cancel_appointment", args)
    if bad:
        return bad
    ap = sb.find_appointment(args["appointment_id"])
    if ap is None or ap.status != "booked":
        return err("not_found", "no booked appointment with that id")
    ap.status = "cancelled"
    return {"appointment_id": ap.appointment_id, "status": "cancelled"}


@tool("block_card")
def block_card(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("block_card", args)
    if bad:
        return bad
    cust = sb.find_customer(args["customer_id"])
    if cust is None:
        return err("not_found", "customer not found")
    card = next(
        (
            k
            for k in sb.customer_cards(cust.customer_id)
            if k.card_last4 == digits(args["card_last4"])
        ),
        None,
    )
    if card is None:
        return err("not_found", "no card ending in those digits on this account")
    if card.status == "blocked":
        return err("conflict", "that card is already blocked")
    card.status = "blocked"
    card.replacement = args.get("replacement") or "none"
    return {"card_last4": card.card_last4, "status": "blocked", "replacement": card.replacement}


@tool("create_ticket")
def create_ticket(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("create_ticket", args)
    if bad:
        return bad
    cust = sb.find_customer(args["customer_id"])
    if cust is None:
        return err("not_found", "customer not found")
    t = S.Ticket(
        sb.next_id("T"),
        cust.customer_id,
        args["category"],
        str(args.get("summary") or "").strip(),
        "open",
        args.get("idempotency_key"),
    )
    sb.tickets.append(t)
    return {"ticket_id": t.ticket_id, "category": t.category, "status": "open"}


@tool("request_handoff")
def request_handoff(sb: Sandbox, args: dict) -> dict:
    reason = args.get("reason")
    valid = reason in HANDOFF_REASONS
    reason = reason if valid else "out_of_scope"
    sb.handoff = HandoffRecord(
        reason=reason, summary=str(args.get("summary") or ""), reason_valid=valid
    )
    return {"status": "transferring", "reason": reason}
