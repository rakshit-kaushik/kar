"""eval.sandbox on seed.json: data, clock, blocked slots and every rule in seed.json["rules"]."""

from datetime import date, datetime

import pytest

from eval.sandbox import Sandbox, load_seed

DANA = {"phone": "5550142219", "date_of_birth": "1988-03-12"}


@pytest.fixture
def sb():
    return Sandbox()


def test_seed_loads_and_clock_is_monday_ten():
    seed = load_seed()
    assert seed["clock"] == "2026-11-02T10:00"
    sb = Sandbox()
    assert sb.clock == datetime(2026, 11, 2, 10, 0) and sb.today == date(2026, 11, 2)
    assert sb.clock_spoken.startswith("Monday, November second")
    assert len(sb.customers) == 10 and "C-1001" in sb.customers


def test_unknown_tool_and_bad_args_are_invalid_argument(sb):
    assert sb.call("make_coffee", {}) == {
        "error": "invalid_argument",
        "message": "unknown tool make_coffee",
    }
    r = sb.call("check_availability", "not json")
    assert r["error"] == "invalid_argument"
    assert sb.tool_log[-1].ok is False


# ----- lookup_account -------------------------------------------------------------------------
def test_lookup_verified_profile(sb):
    r = sb.call("lookup_account", DANA)
    assert r == {
        "verified": True,
        "customer_id": "C-1001",
        "first_name": "Dana",
        "branch_id": "market_st",
        "appointments": [],
        "cards": [{"card_last4": "4219", "status": "active"}],
    }


def test_lookup_lists_booked_appointments_sorted(sb):
    r = sb.call("lookup_account", {"phone": "5550198823", "date_of_birth": "1975-07-04"})
    assert r["appointments"] == [
        {
            "appointment_id": "A-2001",
            "branch_id": "marlow_st",
            "service_type": "notary",
            "slot_start": "2026-11-05T14:00",
        }
    ]
    assert [c["card_last4"] for c in r["cards"]] == ["7781", "0034"]


def test_lookup_normalizes_phone_formats(sb):
    assert sb.call("lookup_account", {"phone": "(555) 014-2219", "date_of_birth": "1988-03-12"})[
        "verified"
    ]
    assert sb.call("lookup_account", {"phone": "1-555-014-2219", "date_of_birth": "1988-03-12"})[
        "verified"
    ]


def test_lookup_not_verified_vs_not_found(sb):
    assert (
        sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-13"})["error"]
        == "not_verified"
    )
    assert (
        sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "march twelfth"})[
            "error"
        ]
        == "not_verified"
    )
    assert sb.call("lookup_account", {"phone": "5550142219"})["error"] == "not_verified"
    assert (
        sb.call("lookup_account", {"phone": "5550900001", "date_of_birth": "1988-03-12"})["error"]
        == "not_found"
    )
    assert sb.call("lookup_account", {"date_of_birth": "1988-03-12"})["error"] == "not_found"


# ----- check_availability -----------------------------------------------------------------------
def avail(sb, branch, day, service="notary"):
    return sb.call(
        "check_availability", {"branch_id": branch, "service_type": service, "date": day}
    )


def test_availability_excludes_booked_and_blocked_slots(sb):
    slots = avail(sb, "marlow_st", "2026-11-05")["slots"]
    assert "14:00" not in slots and "09:00" in slots and "16:30" in slots and len(slots) == 15
    assert avail(sb, "marker_st", "2026-11-03")["slots"][0] == "12:00"
    assert avail(sb, "mill_rd", "2026-11-04") == {
        "branch_id": "mill_rd",
        "date": "2026-11-04",
        "slots": [],
    }


def test_availability_today_hides_slots_at_or_before_the_clock(sb):
    slots = avail(sb, "market_st", "2026-11-02")["slots"]
    assert slots[0] == "10:30" and "10:00" not in slots


def test_availability_saturday_has_eight_slots_and_no_horizon(sb):
    assert len(avail(sb, "market_st", "2026-11-07")["slots"]) == 8
    assert len(avail(sb, "market_st", "2027-01-05")["slots"]) == 16


def test_availability_errors(sb):
    assert avail(sb, "mill_rd", "2026-11-08")["error"] == "unavailable"  # Sunday
    assert avail(sb, "mill_rd", "2026-11-01")["error"] == "unavailable"  # past
    assert avail(sb, "main_st", "2026-11-03")["error"] == "invalid_argument"
    assert avail(sb, "mill_rd", "2026-11-03", service="haircut")["error"] == "invalid_argument"
    assert avail(sb, "mill_rd", "tuesday")["error"] == "invalid_argument"
    assert (
        sb.call("check_availability", {"branch_id": "mill_rd", "service_type": "notary"})["error"]
        == "invalid_argument"
    )


# ----- book_appointment -------------------------------------------------------------------------
def book(sb, slot="2026-11-03T09:30", key="k1", **over):
    a = {
        "customer_id": "C-1001",
        "branch_id": "market_st",
        "service_type": "mortgage_consultation",
        "slot_start": slot,
        "idempotency_key": key,
    }
    a.update(over)
    return sb.call("book_appointment", a)


def test_book_creates_a3xxx_and_takes_the_slot(sb):
    r = book(sb)
    assert r == {
        "appointment_id": "A-3001",
        "branch_id": "market_st",
        "service_type": "mortgage_consultation",
        "slot_start": "2026-11-03T09:30",
        "status": "booked",
    }
    assert "09:30" not in avail(sb, "market_st", "2026-11-03")["slots"]
    assert sb.call("lookup_account", DANA)["appointments"][0]["appointment_id"] == "A-3001"
    assert book(sb, slot="2026-11-04T10:00", key="k2")["appointment_id"] == "A-3002"


def test_book_same_service_same_day_is_conflict_and_names_the_existing_one(sb):
    r = sb.call(
        "book_appointment",
        {
            "customer_id": "C-1002",
            "branch_id": "marlow_st",
            "service_type": "notary",
            "slot_start": "2026-11-05T16:00",
            "idempotency_key": "k",
        },
    )
    assert r["error"] == "conflict" and "A-2001" in r["message"]


def test_book_taken_blocked_closed_or_past_slot_is_unavailable(sb):
    assert (
        book(sb, slot="2026-11-05T14:00", branch_id="marlow_st", service_type="notary")["error"]
        == "unavailable"
    )
    assert (
        book(sb, slot="2026-11-04T10:00", branch_id="mill_rd")["error"] == "unavailable"
    )  # blocked all day
    assert book(sb, slot="2026-11-03T17:00")["error"] == "unavailable"  # after hours
    assert book(sb, slot="2026-11-08T10:00")["error"] == "unavailable"  # Sunday
    assert book(sb, slot="2026-11-02T09:30")["error"] == "unavailable"  # before the clock
    assert book(sb, slot="2026-11-03T09:15")["error"] == "unavailable"  # off the half-hour grid


def test_book_bad_arguments(sb):
    assert book(sb, customer_id="C-9999")["error"] == "not_found"
    assert book(sb, branch_id="main_st")["error"] == "invalid_argument"
    assert book(sb, service_type="haircut")["error"] == "invalid_argument"
    assert book(sb, slot="tuesday at nine")["error"] == "invalid_argument"
    args = {
        "customer_id": "C-1001",
        "branch_id": "market_st",
        "service_type": "notary",
        "slot_start": "2026-11-03T09:30",
    }
    assert sb.call("book_appointment", args)["error"] == "invalid_argument"  # no idempotency_key


def test_idempotent_write_returns_original_without_second_write(sb):
    a = book(sb)
    b = book(sb)
    assert a == b and sb.tool_log[-1].replay is True
    assert len(sb.call("lookup_account", DANA)["appointments"]) == 1
    # a failed call is not cached: retrying the same key after fixing the slot works
    bad = book(sb, slot="2026-11-08T10:00", key="k9")
    assert bad["error"] == "unavailable"
    assert "appointment_id" in book(sb, slot="2026-11-06T09:00", key="k9")


# ----- reschedule_appointment ---------------------------------------------------------------------
def test_reschedule_moves_and_frees_old_slot(sb):
    r = sb.call(
        "reschedule_appointment",
        {"appointment_id": "A-2001", "new_slot_start": "2026-11-06T10:00", "idempotency_key": "r1"},
    )
    assert r == {
        "appointment_id": "A-2001",
        "branch_id": "marlow_st",
        "service_type": "notary",
        "slot_start": "2026-11-06T10:00",
        "status": "booked",
    }
    assert "14:00" in avail(sb, "marlow_st", "2026-11-05")["slots"]


def test_reschedule_errors(sb):
    assert (
        sb.call(
            "reschedule_appointment",
            {
                "appointment_id": "A-9999",
                "new_slot_start": "2026-11-06T10:00",
                "idempotency_key": "r",
            },
        )["error"]
        == "not_found"
    )
    sb.call("cancel_appointment", {"appointment_id": "A-2002", "idempotency_key": "c"})
    assert (
        sb.call(
            "reschedule_appointment",
            {
                "appointment_id": "A-2002",
                "new_slot_start": "2026-11-11T10:00",
                "idempotency_key": "r",
            },
        )["error"]
        == "not_found"
    )
    assert (
        sb.call(
            "reschedule_appointment",
            {
                "appointment_id": "A-2001",
                "new_slot_start": "2026-11-05T14:00",
                "idempotency_key": "r",
            },
        )["error"]
        == "conflict"
    )  # same slot
    assert (
        sb.call(
            "reschedule_appointment",
            {
                "appointment_id": "A-2003",
                "new_slot_start": "2026-11-05T14:00",
                "idempotency_key": "r",
            },
        )["error"]
        == "unavailable"
    )  # taken by A-2001
    assert (
        sb.call(
            "reschedule_appointment",
            {"appointment_id": "A-2001", "new_slot_start": "friday", "idempotency_key": "r"},
        )["error"]
        == "invalid_argument"
    )


# ----- cancel_appointment ---------------------------------------------------------------------------
def test_cancel_sets_status_and_second_cancel_is_not_found(sb):
    assert sb.call("cancel_appointment", {"appointment_id": "A-2002", "idempotency_key": "c1"}) == {
        "appointment_id": "A-2002",
        "status": "cancelled",
    }
    assert (
        sb.call("cancel_appointment", {"appointment_id": "A-2002", "idempotency_key": "c2"})[
            "error"
        ]
        == "not_found"
    )
    assert (
        sb.call("lookup_account", {"phone": "5550177310", "date_of_birth": "1992-11-30"})[
            "appointments"
        ]
        == []
    )
    assert (
        sb.call("cancel_appointment", {"appointment_id": "A-2001"})["error"] == "invalid_argument"
    )


# ----- block_card -------------------------------------------------------------------------------------
def test_block_card_rules(sb):
    r = sb.call(
        "block_card",
        {
            "customer_id": "C-1005",
            "card_last4": "3307",
            "reason": "damaged",
            "replacement": "standard",
            "idempotency_key": "b1",
        },
    )
    assert r == {"card_last4": "3307", "status": "blocked", "replacement": "standard"}
    assert (
        sb.call(
            "block_card",
            {
                "customer_id": "C-1004",
                "card_last4": "9912",
                "reason": "lost",
                "idempotency_key": "b2",
            },
        )["error"]
        == "conflict"
    )
    assert (
        sb.call(
            "block_card",
            {
                "customer_id": "C-1005",
                "card_last4": "0000",
                "reason": "lost",
                "idempotency_key": "b3",
            },
        )["error"]
        == "not_found"
    )
    assert (
        sb.call(
            "block_card",
            {
                "customer_id": "C-9999",
                "card_last4": "3307",
                "reason": "lost",
                "idempotency_key": "b4",
            },
        )["error"]
        == "not_found"
    )
    assert (
        sb.call(
            "block_card",
            {
                "customer_id": "C-1001",
                "card_last4": "4219",
                "reason": "misplaced",
                "idempotency_key": "b5",
            },
        )["error"]
        == "invalid_argument"
    )
    r = sb.call(
        "block_card",
        {"customer_id": "C-1001", "card_last4": "4219", "reason": "lost", "idempotency_key": "b6"},
    )
    assert r["replacement"] == "none"


# ----- create_ticket ----------------------------------------------------------------------------------
def test_create_ticket_rules(sb):
    r = sb.call(
        "create_ticket",
        {
            "customer_id": "C-1009",
            "category": "fee_dispute",
            "summary": "overdraft fee",
            "idempotency_key": "t1",
        },
    )
    assert r == {"ticket_id": "T-5001", "category": "fee_dispute", "status": "open"}
    assert (
        sb.call(
            "create_ticket",
            {
                "customer_id": "C-1009",
                "category": "refund",
                "summary": "x",
                "idempotency_key": "t2",
            },
        )["error"]
        == "invalid_argument"
    )
    assert (
        sb.call(
            "create_ticket",
            {
                "customer_id": "C-9999",
                "category": "complaint",
                "summary": "x",
                "idempotency_key": "t3",
            },
        )["error"]
        == "not_found"
    )
    assert sb.tickets[0].customer_id == "C-1009" and sb.tickets[0].summary == "overdraft fee"


# ----- request_handoff ----------------------------------------------------------------------------------
def test_handoff_always_succeeds_and_coerces_bad_reason(sb):
    assert sb.call("request_handoff", {"reason": "suspected_fraud", "summary": "card 5560"}) == {
        "status": "transferring",
        "reason": "suspected_fraud",
    }
    assert sb.dump_state()["handoff"] == {
        "reason": "suspected_fraud",
        "summary": "card 5560",
        "reason_valid": True,
    }
    r = sb.call("request_handoff", {"reason": "because"})
    assert r == {"status": "transferring", "reason": "out_of_scope"}
    assert sb.handoff.reason_valid is False


# ----- fault injection ------------------------------------------------------------------------------------
def test_inject_hits_only_the_nth_call_and_writes_nothing(sb):
    sb.inject(
        [
            {"tool": "cancel_appointment", "call_index": 1, "error": "timeout"},
            {"tool": "lookup_account", "call_index": 2, "error": "not_found"},
        ]
    )
    first = sb.call("cancel_appointment", {"appointment_id": "A-2001", "idempotency_key": "x1"})
    assert first == {"error": "timeout", "message": "the request timed out"}
    assert sb.tool_log[-1].injected is True
    assert (
        sb.call("lookup_account", {"phone": "5550198823", "date_of_birth": "1975-07-04"})[
            "appointments"
        ][0]["appointment_id"]
        == "A-2001"
    )
    assert (
        sb.call("lookup_account", {"phone": "5550198823", "date_of_birth": "1975-07-04"})["error"]
        == "not_found"
    )
    assert (
        sb.call("cancel_appointment", {"appointment_id": "A-2001", "idempotency_key": "x1"})[
            "status"
        ]
        == "cancelled"
    )


def test_inject_timeout_shortcut_counts_replays(sb):
    sb.inject_timeout("book_appointment", 2)
    assert "appointment_id" in book(sb)
    assert book(sb)["error"] == "timeout"  # second call (a replay) is the injected one
    assert book(sb) == book(sb)


# ----- state, diff, reset, reject --------------------------------------------------------------------------
def test_diff_against_seed_and_dump_state(sb):
    book(sb)
    sb.call("cancel_appointment", {"appointment_id": "A-2002", "idempotency_key": "c"})
    sb.call(
        "block_card",
        {"customer_id": "C-1005", "card_last4": "3307", "reason": "lost", "idempotency_key": "b"},
    )
    sb.call(
        "create_ticket",
        {
            "customer_id": "C-1005",
            "category": "card_lost_stolen",
            "summary": "lost",
            "idempotency_key": "t",
        },
    )
    d = sb.diff()
    assert d == {
        "appointments_new": [
            {
                "customer_id": "C-1001",
                "branch_id": "market_st",
                "service_type": "mortgage_consultation",
                "slot_start": "2026-11-03T09:30",
            }
        ],
        "appointments_changed": [
            {"appointment_id": "A-2002", "status": "cancelled", "slot_start": "2026-11-10T11:00"}
        ],
        "cards": [{"card_last4": "3307", "status": "blocked", "replacement": "none"}],
        "tickets": [{"customer_id": "C-1005", "category": "card_lost_stolen"}],
    }
    st = sb.dump_state()
    assert st["diff"] == d and st["handoff"] is None and st["tool_calls"] == len(sb.tool_log)
    assert st["clock"] == "2026-11-02T10:00"
    assert any(a["appointment_id"] == "A-3001" for a in st["appointments"])


def test_reset_restores_seed_and_clears_log_and_counters(sb):
    sb.inject_timeout("lookup_account", 1)
    book(sb)
    sb.call("lookup_account", DANA)
    sb.reset()
    assert sb.tool_log == [] and sb.diff() == {
        "appointments_new": [],
        "appointments_changed": [],
        "cards": [],
        "tickets": [],
    }
    assert (
        sb.call("lookup_account", DANA)["error"] == "timeout"
    )  # faults stay armed, counters restart
    assert sb.call("lookup_account", DANA)["appointments"] == []


def test_reject_logs_an_invalid_call(sb):
    r = sb.reject("lookup_account", "{bad json", "Tool arguments were not valid JSON.")
    assert r == {"error": "invalid_argument", "message": "Tool arguments were not valid JSON."}
    assert sb.tool_log[0].ok is False and sb.tool_log[0].tool == "lookup_account"


def test_tools_follow_the_schema_files_present(sb):
    from eval.sandbox import TOOL_NAMES

    assert set(sb.tools) == set(TOOL_NAMES)
