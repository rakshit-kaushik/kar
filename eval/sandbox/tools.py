"""The eight contract tools (CONTRACT.md §4) against a `Sandbox`.

Each handler takes (sandbox, args) and returns a JSON-able dict: either the tool's result
or {"error": <code>, "message": <short text>}. Idempotency replay and timeout injection
happen in `Sandbox.call`, not here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from . import seed as S
from .store import digits, err, parse_date, parse_slot, validate_args

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
        "service_type": a.service_type,
        "branch_id": a.branch_id,
        "slot_start": a.slot_start.strftime("%Y-%m-%dT%H:%M"),
    }


# ----- read tools ------------------------------------------------------------
@tool("lookup_account")
def lookup_account(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("lookup_account", args, fallback="not_found")
    if bad:
        return bad
    phone = digits(args["phone"])
    if len(phone) != 10:
        return err("not_found", "No account matches that phone number.")
    cust = sb.find_customer_by_phone(phone)
    if cust is None:
        return err("not_found", "No account matches that phone number.")
    dob = parse_date(args["date_of_birth"])
    if dob is None or dob != cust.date_of_birth:
        return err("not_verified", "Date of birth does not match the account.")
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
    bad = validate_args("check_availability", args, fallback="invalid_argument")
    if bad:
        return bad
    d = parse_date(args["date"])
    if d is None:
        return err("invalid_argument", "date must be YYYY-MM-DD.")
    problem = sb.date_problem(d)
    if problem:
        return err("unavailable", problem)
    return {
        "slots": sb.open_slots(args["branch_id"], d),
        "branch_id": args["branch_id"],
        "date": d.isoformat(),
    }


# ----- write tools -----------------------------------------------------------
def _slot_problem(sb: Sandbox, dt: datetime) -> str | None:
    """Why a slot cannot be booked (unavailable), or None."""
    problem = sb.date_problem(dt.date())
    if problem:
        return problem
    if dt.strftime("%H:%M") not in S.slots_for(dt.date()):
        hours = S.hours_for(dt.date())
        return f"Outside opening hours ({hours[0]}–{hours[1]})."
    if dt <= sb.now:
        return "That time has already passed."
    return None


@tool("book_appointment")
def book_appointment(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("book_appointment", args, fallback="invalid_argument")
    if bad:
        return bad
    cust = sb.find_customer(args["customer_id"])
    if cust is None:
        return err("invalid_argument", "Unknown customer_id.")
    dt = parse_slot(args["slot_start"])
    if dt is None:
        return err("invalid_argument", "slot_start must be YYYY-MM-DDTHH:MM.")
    if dt.minute not in (0, 30) or dt.second:
        return err("invalid_argument", "Slots start on the hour or half hour.")
    problem = _slot_problem(sb, dt)
    if problem:
        return err("unavailable", problem)
    dup = next(
        (
            a
            for a in sb.customer_appointments(cust.customer_id)
            if a.service_type == args["service_type"] and a.slot_start.date() == dt.date()
        ),
        None,
    )
    if dup:
        return err(
            "conflict",
            f"Customer already has a {args['service_type']} appointment that day "
            f"({dup.appointment_id} at {dup.slot_start.strftime('%H:%M')}).",
        )
    if not sb.slot_is_open(args["branch_id"], dt):
        return err("unavailable", "That slot was just taken; check availability again.")
    ap = S.Appointment(
        sb.new_appointment_id(),
        cust.customer_id,
        args["branch_id"],
        args["service_type"],
        dt,
        "booked",
        args["idempotency_key"],
    )
    sb.appointments.append(ap)
    sb.touch(ap.appointment_id)
    return {
        "appointment_id": ap.appointment_id,
        "branch_id": ap.branch_id,
        "service_type": ap.service_type,
        "slot_start": dt.strftime("%Y-%m-%dT%H:%M"),
    }


@tool("reschedule_appointment")
def reschedule_appointment(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("reschedule_appointment", args, fallback="not_found")
    if bad:
        return bad
    ap = sb.find_appointment(args["appointment_id"])
    if ap is None:
        return err("not_found", "No appointment with that id.")
    if ap.status != "booked":
        return err("conflict", "That appointment is already cancelled.")
    dt = parse_slot(args["new_slot_start"])
    if dt is None or dt.minute not in (0, 30) or dt.second:
        return err("unavailable", "new_slot_start must be YYYY-MM-DDTHH:MM on a half hour.")
    if dt == ap.slot_start:
        return err("conflict", "The appointment is already at that time.")
    problem = _slot_problem(sb, dt)
    if problem:
        return err("unavailable", problem)
    if not sb.slot_is_open(ap.branch_id, dt):
        return err("unavailable", "That slot is taken; check availability again.")
    ap.slot_start = dt
    sb.touch(ap.appointment_id)
    return {
        "appointment_id": ap.appointment_id,
        "branch_id": ap.branch_id,
        "service_type": ap.service_type,
        "slot_start": dt.strftime("%Y-%m-%dT%H:%M"),
    }


@tool("cancel_appointment")
def cancel_appointment(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("cancel_appointment", args, fallback="not_found")
    if bad:
        return bad
    ap = sb.find_appointment(args["appointment_id"])
    if ap is None:
        return err("not_found", "No appointment with that id.")
    if ap.status == "cancelled":
        return err("not_found", "That appointment was already cancelled.")
    ap.status = "cancelled"
    sb.touch(ap.appointment_id)
    return {"appointment_id": ap.appointment_id, "status": "cancelled"}


@tool("block_card")
def block_card(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("block_card", args, fallback="not_found")
    if bad:
        return bad
    cust = sb.find_customer(args["customer_id"])
    if cust is None:
        return err("not_found", "Unknown customer_id.")
    card = next(
        (
            k
            for k in sb.customer_cards(cust.customer_id)
            if k.card_last4 == digits(args["card_last4"])
        ),
        None,
    )
    if card is None:
        return err("not_found", "No card ending in those digits on this account.")
    if card.status == "blocked":
        return err("conflict", f"Card ending {card.card_last4} is already blocked.")
    card.status = "blocked"
    replacement = args.get("replacement") or "none"
    card.replacement = replacement
    out = {"blocked": True, "card_last4": card.card_last4, "replacement": replacement}
    if replacement in S.CARD_REPLACEMENT:
        out["replacement_eta"] = S.CARD_REPLACEMENT[replacement]["eta"]
        out["replacement_fee_usd"] = S.CARD_REPLACEMENT[replacement]["fee_usd"]
    return out


@tool("create_ticket")
def create_ticket(sb: Sandbox, args: dict) -> dict:
    bad = validate_args("create_ticket", args, fallback="invalid_argument")
    if bad:
        return bad
    cust = sb.find_customer(args["customer_id"])
    if cust is None:
        return err("invalid_argument", "Unknown customer_id.")
    if not args["summary"].strip():
        return err("invalid_argument", "summary must not be empty.")
    t = S.Ticket(
        sb.new_ticket_id(),
        cust.customer_id,
        args["category"],
        args["summary"].strip(),
        args["idempotency_key"],
    )
    sb.tickets.append(t)
    return {"ticket_id": t.ticket_id, "category": t.category}


@tool("request_handoff")
def request_handoff(sb: Sandbox, args: dict) -> dict:
    from .store import HandoffRecord  # local import keeps module import order simple

    args = args if isinstance(args, dict) else {}
    reason = str(args.get("reason") or "unspecified")
    allowed = TOOL_SCHEMAS_REASONS
    sb.handoff = HandoffRecord(
        reason=reason,
        summary=str(args.get("summary") or ""),
        ts=datetime.now().isoformat(timespec="seconds"),
        reason_valid=reason in allowed,
    )
    return {"handoff": True, "reason": reason}


from .store import TOOL_SCHEMAS

TOOL_SCHEMAS_REASONS = TOOL_SCHEMAS["request_handoff"]["function"]["parameters"]["properties"][
    "reason"
]["enum"]
