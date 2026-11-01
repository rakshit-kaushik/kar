"""Write tools: conflict/unavailable/invalid_argument/not_found/timeout + idempotency."""

from datetime import date

import pytest

from eval.sandbox import Sandbox

TODAY = date(2026, 11, 2)
DANA = "C-1042"
OPEN_DAY = "2026-11-04"


@pytest.fixture
def sb():
    return Sandbox(seed=42, today=TODAY)


def first_open(sb, branch="market_st", day=OPEN_DAY, service="notary"):
    return sb.call(
        "check_availability", {"branch_id": branch, "service_type": service, "date": day}
    )["slots"][0]


def book_args(sb, **over):
    slot = over.pop("slot", None) or first_open(sb)
    a = {
        "customer_id": DANA,
        "branch_id": "market_st",
        "service_type": "notary",
        "slot_start": f"{OPEN_DAY}T{slot}",
        "idempotency_key": "k-1",
    }
    a.update(over)
    return a


# ----- book_appointment ------------------------------------------------------
def test_book_ok_creates_appointment_and_removes_slot(sb):
    args = book_args(sb)
    r = sb.call("book_appointment", args)
    assert set(r) == {"appointment_id", "branch_id", "service_type", "slot_start"}
    assert r["appointment_id"].startswith("A-")
    assert r["slot_start"] == args["slot_start"]
    slots = sb.call(
        "check_availability", {"branch_id": "market_st", "service_type": "notary", "date": OPEN_DAY}
    )["slots"]
    assert args["slot_start"][-5:] not in slots
    look = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    assert look["appointments"][0]["appointment_id"] == r["appointment_id"]


def test_book_repeat_idempotency_key_returns_original(sb):
    args = book_args(sb)
    r1 = sb.call("book_appointment", args)
    r2 = sb.call("book_appointment", args)
    assert r1 == r2
    assert sb.tool_log[-1].replay is True
    assert len([a for a in sb.appointments if a.customer_id == DANA]) == 1


def test_book_same_service_same_day_is_conflict(sb):
    slots = sb.call(
        "check_availability", {"branch_id": "market_st", "service_type": "notary", "date": OPEN_DAY}
    )["slots"]
    sb.call("book_appointment", book_args(sb, slot=slots[0], idempotency_key="k-1"))
    r = sb.call("book_appointment", book_args(sb, slot=slots[1], idempotency_key="k-2"))
    assert r["error"] == "conflict"
    assert "A-" in r["message"]  # names the existing appointment


def test_book_taken_slot_is_unavailable(sb):
    taken = next(
        a
        for a in sb.appointments
        if a.branch_id == "marlow_st" and a.slot_start.date() == date(2026, 11, 4)
    )
    r = sb.call(
        "book_appointment",
        book_args(
            sb, branch_id="marlow_st", slot_start=taken.slot_start.strftime("%Y-%m-%dT%H:%M")
        ),
    )
    assert r["error"] == "unavailable"


def test_book_outside_hours_is_unavailable(sb):
    r = sb.call("book_appointment", book_args(sb, slot_start=f"{OPEN_DAY}T18:00"))
    assert r["error"] == "unavailable"


def test_book_off_grid_time_is_invalid_argument(sb):
    r = sb.call("book_appointment", book_args(sb, slot_start=f"{OPEN_DAY}T09:15"))
    assert r["error"] == "invalid_argument"


def test_book_in_the_past_is_unavailable(sb):
    r = sb.call("book_appointment", book_args(sb, slot_start="2026-10-28T10:00"))
    assert r["error"] == "unavailable"


def test_book_sunday_is_unavailable(sb):
    r = sb.call("book_appointment", book_args(sb, slot_start="2026-11-08T10:00"))
    assert r["error"] == "unavailable"


def test_book_bad_enum_is_invalid_argument(sb):
    assert (
        sb.call("book_appointment", book_args(sb, branch_id="main_st"))["error"]
        == "invalid_argument"
    )
    assert (
        sb.call("book_appointment", book_args(sb, service_type="haircut"))["error"]
        == "invalid_argument"
    )


def test_book_malformed_slot_is_invalid_argument(sb):
    r = sb.call("book_appointment", book_args(sb, slot_start="tuesday 9:30"))
    assert r["error"] == "invalid_argument"


def test_book_unknown_customer_is_invalid_argument(sb):
    r = sb.call("book_appointment", book_args(sb, customer_id="C-9"))
    assert r["error"] == "invalid_argument"


def test_book_missing_idempotency_key_is_invalid_argument(sb):
    args = book_args(sb)
    del args["idempotency_key"]
    assert sb.call("book_appointment", args)["error"] == "invalid_argument"


def test_book_injected_timeout_does_not_write_and_retry_succeeds(sb):
    sb.inject_timeout("book_appointment", nth_call=1)
    args = book_args(sb)
    assert sb.call("book_appointment", args)["error"] == "timeout"
    assert not any(a.customer_id == DANA for a in sb.appointments)
    r = sb.call("book_appointment", args)  # same key, retry
    assert "appointment_id" in r
    assert sb.tool_log[-1].replay is False


# ----- reschedule_appointment -------------------------------------------------
@pytest.fixture
def booked(sb):
    r = sb.call("book_appointment", book_args(sb))
    return r["appointment_id"]


def test_reschedule_ok_moves_slot(sb, booked):
    new_slot = first_open(sb, day="2026-11-05")
    r = sb.call(
        "reschedule_appointment",
        {
            "appointment_id": booked,
            "new_slot_start": f"2026-11-05T{new_slot}",
            "idempotency_key": "r-1",
        },
    )
    assert r == {
        "appointment_id": booked,
        "branch_id": "market_st",
        "service_type": "notary",
        "slot_start": f"2026-11-05T{new_slot}",
    }
    assert (
        sb.find_appointment(booked).slot_start.strftime("%Y-%m-%dT%H:%M")
        == f"2026-11-05T{new_slot}"
    )


def test_reschedule_not_found(sb):
    r = sb.call(
        "reschedule_appointment",
        {"appointment_id": "A-1", "new_slot_start": f"{OPEN_DAY}T10:00", "idempotency_key": "r-1"},
    )
    assert r["error"] == "not_found"


def test_reschedule_cancelled_appointment_is_conflict(sb, booked):
    sb.call("cancel_appointment", {"appointment_id": booked, "idempotency_key": "c-1"})
    r = sb.call(
        "reschedule_appointment",
        {
            "appointment_id": booked,
            "new_slot_start": f"2026-11-05T{first_open(sb, day='2026-11-05')}",
            "idempotency_key": "r-1",
        },
    )
    assert r["error"] == "conflict"


def test_reschedule_to_taken_slot_is_unavailable(sb, booked):
    taken = next(
        a
        for a in sb.appointments
        if a.branch_id == "market_st" and a.slot_start.date() == date(2026, 11, 5)
    )
    r = sb.call(
        "reschedule_appointment",
        {
            "appointment_id": booked,
            "new_slot_start": taken.slot_start.strftime("%Y-%m-%dT%H:%M"),
            "idempotency_key": "r-1",
        },
    )
    assert r["error"] == "unavailable"


def test_reschedule_to_same_slot_is_conflict(sb, booked):
    cur = sb.find_appointment(booked).slot_start.strftime("%Y-%m-%dT%H:%M")
    r = sb.call(
        "reschedule_appointment",
        {"appointment_id": booked, "new_slot_start": cur, "idempotency_key": "r-1"},
    )
    assert r["error"] == "conflict"


def test_reschedule_malformed_slot_is_not_found_fallback(sb, booked):
    # reschedule has no invalid_argument code in the contract; malformed input maps to unavailable
    r = sb.call(
        "reschedule_appointment",
        {"appointment_id": booked, "new_slot_start": "thursday", "idempotency_key": "r-1"},
    )
    assert r["error"] == "unavailable"


def test_reschedule_idempotent_replay(sb, booked):
    args = {
        "appointment_id": booked,
        "new_slot_start": f"2026-11-05T{first_open(sb, day='2026-11-05')}",
        "idempotency_key": "r-1",
    }
    r1 = sb.call("reschedule_appointment", args)
    r2 = sb.call("reschedule_appointment", args)
    assert r1 == r2 and sb.tool_log[-1].replay is True


def test_reschedule_timeout_injection(sb, booked):
    sb.inject_timeout("reschedule_appointment", 1)
    args = {
        "appointment_id": booked,
        "new_slot_start": f"2026-11-05T{first_open(sb, day='2026-11-05')}",
        "idempotency_key": "r-1",
    }
    assert sb.call("reschedule_appointment", args)["error"] == "timeout"
    assert sb.find_appointment(booked).slot_start.date() == date(2026, 11, 4)


# ----- cancel_appointment -----------------------------------------------------
def test_cancel_ok(sb, booked):
    r = sb.call("cancel_appointment", {"appointment_id": booked, "idempotency_key": "c-1"})
    assert r == {"appointment_id": booked, "status": "cancelled"}
    assert sb.find_appointment(booked).status == "cancelled"
    look = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    assert look["appointments"] == []


def test_cancel_frees_the_slot(sb, booked):
    slot = sb.find_appointment(booked).slot_start
    sb.call("cancel_appointment", {"appointment_id": booked, "idempotency_key": "c-1"})
    assert (
        slot.strftime("%H:%M")
        in sb.call(
            "check_availability",
            {"branch_id": "market_st", "service_type": "notary", "date": OPEN_DAY},
        )["slots"]
    )


def test_cancel_not_found(sb):
    r = sb.call("cancel_appointment", {"appointment_id": "A-1", "idempotency_key": "c-1"})
    assert r["error"] == "not_found"


def test_cancel_already_cancelled_is_not_found(sb, booked):
    sb.call("cancel_appointment", {"appointment_id": booked, "idempotency_key": "c-1"})
    r = sb.call("cancel_appointment", {"appointment_id": booked, "idempotency_key": "c-2"})
    assert r["error"] == "not_found"
    assert "already" in r["message"]


def test_cancel_missing_key_is_not_found(sb, booked):
    assert sb.call("cancel_appointment", {"appointment_id": booked})["error"] == "not_found"


def test_cancel_timeout_injection(sb, booked):
    sb.inject_timeout("cancel_appointment", 1)
    assert (
        sb.call("cancel_appointment", {"appointment_id": booked, "idempotency_key": "c-1"})["error"]
        == "timeout"
    )
    assert sb.find_appointment(booked).status == "booked"


# ----- block_card -------------------------------------------------------------
def test_block_card_ok(sb):
    r = sb.call(
        "block_card",
        {"customer_id": DANA, "card_last4": "4219", "reason": "lost", "idempotency_key": "b-1"},
    )
    assert r == {"blocked": True, "card_last4": "4219", "replacement": "none"}
    assert sb.customer_cards(DANA)[0].status == "blocked"


def test_block_card_with_replacement_echoes_eta(sb):
    r = sb.call(
        "block_card",
        {
            "customer_id": DANA,
            "card_last4": "4219",
            "reason": "stolen",
            "replacement": "express",
            "idempotency_key": "b-1",
        },
    )
    assert r == {
        "blocked": True,
        "card_last4": "4219",
        "replacement": "express",
        "replacement_eta": "2 business days",
        "replacement_fee_usd": 25,
    }


def test_block_card_already_blocked_is_conflict(sb):
    sb.call(
        "block_card",
        {"customer_id": DANA, "card_last4": "4219", "reason": "lost", "idempotency_key": "b-1"},
    )
    r = sb.call(
        "block_card",
        {"customer_id": DANA, "card_last4": "4219", "reason": "lost", "idempotency_key": "b-2"},
    )
    assert r["error"] == "conflict"
    assert "already blocked" in r["message"]


def test_block_card_wrong_last4_is_not_found(sb):
    r = sb.call(
        "block_card",
        {"customer_id": DANA, "card_last4": "0000", "reason": "lost", "idempotency_key": "b-1"},
    )
    assert r["error"] == "not_found"


def test_block_card_unknown_customer_is_not_found(sb):
    r = sb.call(
        "block_card",
        {"customer_id": "C-9", "card_last4": "4219", "reason": "lost", "idempotency_key": "b-1"},
    )
    assert r["error"] == "not_found"


def test_block_card_bad_reason_is_not_found_fallback(sb):
    r = sb.call(
        "block_card",
        {
            "customer_id": DANA,
            "card_last4": "4219",
            "reason": "misplaced",
            "idempotency_key": "b-1",
        },
    )
    assert r["error"] == "not_found"


def test_block_card_idempotent_and_timeout(sb):
    sb.inject_timeout("block_card", 1)
    args = {"customer_id": DANA, "card_last4": "4219", "reason": "lost", "idempotency_key": "b-1"}
    assert sb.call("block_card", args)["error"] == "timeout"
    assert sb.customer_cards(DANA)[0].status == "active"
    r1 = sb.call("block_card", args)
    assert r1["blocked"] is True
    assert sb.call("block_card", args) == r1 and sb.tool_log[-1].replay is True


# ----- create_ticket ----------------------------------------------------------
def test_create_ticket_ok(sb):
    r = sb.call(
        "create_ticket",
        {
            "customer_id": DANA,
            "category": "unauthorized_transaction",
            "summary": "Lost card 4219; caller reports a charge they did not make",
            "idempotency_key": "t-1",
        },
    )
    assert r == {"ticket_id": "T-5520", "category": "unauthorized_transaction"}
    assert sb.tickets[0].customer_id == DANA


def test_create_ticket_bad_category(sb):
    r = sb.call(
        "create_ticket",
        {"customer_id": DANA, "category": "fraud", "summary": "x", "idempotency_key": "t-1"},
    )
    assert r["error"] == "invalid_argument"


def test_create_ticket_unknown_customer(sb):
    r = sb.call(
        "create_ticket",
        {"customer_id": "C-9", "category": "complaint", "summary": "x", "idempotency_key": "t-1"},
    )
    assert r["error"] == "invalid_argument"


def test_create_ticket_empty_summary(sb):
    r = sb.call(
        "create_ticket",
        {"customer_id": DANA, "category": "complaint", "summary": "", "idempotency_key": "t-1"},
    )
    assert r["error"] == "invalid_argument"


def test_create_ticket_idempotent_and_timeout(sb):
    sb.inject_timeout("create_ticket", 1)
    args = {
        "customer_id": DANA,
        "category": "complaint",
        "summary": "Waited an hour",
        "idempotency_key": "t-1",
    }
    assert sb.call("create_ticket", args)["error"] == "timeout"
    assert sb.tickets == []
    r1 = sb.call("create_ticket", args)
    assert sb.call("create_ticket", args) == r1
    assert len(sb.tickets) == 1


# ----- request_handoff --------------------------------------------------------
def test_handoff_records_reason_and_summary(sb):
    r = sb.call("request_handoff", {"reason": "suspected_fraud", "summary": "Card 4219 blocked"})
    assert r == {"handoff": True, "reason": "suspected_fraud"}
    assert sb.handoff.reason == "suspected_fraud" and sb.handoff.reason_valid is True
    assert sb.dump_state()["handoff"]["requested"] is True


def test_handoff_never_errors_but_flags_bad_reason(sb):
    r = sb.call("request_handoff", {"reason": "because", "summary": "x"})
    assert r == {"handoff": True, "reason": "because"}
    assert sb.handoff.reason_valid is False


def test_handoff_without_summary_still_succeeds(sb):
    r = sb.call("request_handoff", {"reason": "caller_requested"})
    assert r["handoff"] is True and sb.handoff.summary == ""


def test_dump_state_includes_episode_writes(sb, booked):
    sb.call(
        "block_card",
        {"customer_id": DANA, "card_last4": "4219", "reason": "lost", "idempotency_key": "b-1"},
    )
    st = sb.dump_state()
    assert any(a["appointment_id"] == booked for a in st["appointments"])
    assert {
        "customer_id": DANA,
        "card_last4": "4219",
        "status": "blocked",
        "replacement": "none",
    } in st["cards"]
    assert st["handoff"] is None and st["tool_calls"] == len(sb.tool_log)
