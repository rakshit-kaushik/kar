"""In-memory sandbox implementing the eight tools exactly per eval/sandbox/seed.json["rules"].
Fresh per call/episode; deterministic; supports fault injection."""

import json
import pathlib

import pytest

from services.tools.sandbox import Sandbox

SEED = pathlib.Path(__file__).resolve().parents[2] / "eval/sandbox/seed.json"


@pytest.fixture
def sb():
    return Sandbox.from_seed(SEED)


def test_lookup_verifies_with_matching_phone_and_dob(sb):
    r = sb.lookup_account(phone="5550142219", date_of_birth="1988-03-12")
    assert r["verified"] is True and r["customer_id"] == "C-1001" and r["first_name"] == "Dana"
    assert r["appointments"] == [] and r["cards"] == [{"card_last4": "4219", "status": "active"}]


def test_lookup_returns_existing_appointments(sb):
    r = sb.lookup_account(phone="5550198823", date_of_birth="1975-07-04")
    assert r["appointments"][0]["appointment_id"] == "A-2001"
    assert r["appointments"][0]["slot_start"] == "2026-11-05T14:00"


def test_lookup_wrong_dob_is_not_verified_and_unknown_phone_is_not_found(sb):
    assert sb.lookup_account(phone="5550142219", date_of_birth="1988-03-13") == {
        "error": "not_verified",
        "message": "date of birth does not match",
    }
    assert sb.lookup_account(phone="5550900001", date_of_birth="1990-04-01")["error"] == "not_found"


def test_availability_excludes_booked_and_blocked_slots(sb):
    slots = sb.check_availability(branch_id="marlow_st", service_type="notary", date="2026-11-05")[
        "slots"
    ]
    assert "14:00" not in slots and "09:00" in slots and "16:30" in slots and len(slots) == 15
    assert (
        sb.check_availability(branch_id="marker_st", service_type="notary", date="2026-11-03")[
            "slots"
        ][0]
        == "12:00"
    )
    assert (
        sb.check_availability(branch_id="mill_rd", service_type="notary", date="2026-11-04")[
            "slots"
        ]
        == []
    )


def test_availability_errors(sb):
    assert (
        sb.check_availability(branch_id="mill_rd", service_type="notary", date="2026-11-08")[
            "error"
        ]
        == "unavailable"
    )  # Sunday
    assert (
        sb.check_availability(branch_id="mill_rd", service_type="notary", date="2026-11-01")[
            "error"
        ]
        == "unavailable"
    )  # past
    assert (
        sb.check_availability(branch_id="main_st", service_type="notary", date="2026-11-03")[
            "error"
        ]
        == "invalid_argument"
    )
    assert (
        sb.check_availability(branch_id="mill_rd", service_type="haircut", date="2026-11-03")[
            "error"
        ]
        == "invalid_argument"
    )
    assert (
        sb.check_availability(branch_id="mill_rd", service_type="notary", date="tuesday")["error"]
        == "invalid_argument"
    )


def test_saturday_has_eight_slots(sb):
    assert (
        len(
            sb.check_availability(branch_id="market_st", service_type="notary", date="2026-11-07")[
                "slots"
            ]
        )
        == 8
    )


def test_book_creates_appointment_and_takes_the_slot(sb):
    r = sb.book_appointment(
        customer_id="C-1001",
        branch_id="market_st",
        service_type="mortgage_consultation",
        slot_start="2026-11-03T09:30",
        idempotency_key="k1",
    )
    assert r["appointment_id"].startswith("A-3") and r["slot_start"] == "2026-11-03T09:30"
    assert (
        "09:30"
        not in sb.check_availability(
            branch_id="market_st", service_type="notary", date="2026-11-03"
        )["slots"]
    )
    assert (
        sb.lookup_account(phone="5550142219", date_of_birth="1988-03-12")["appointments"][0][
            "appointment_id"
        ]
        == r["appointment_id"]
    )


def test_book_same_service_same_day_is_conflict(sb):
    r = sb.book_appointment(
        customer_id="C-1002",
        branch_id="marlow_st",
        service_type="notary",
        slot_start="2026-11-05T15:00",
        idempotency_key="k2",
    )
    assert r["error"] == "conflict" and "A-2001" in r["message"]


def test_book_taken_or_closed_slot_is_unavailable(sb):
    assert (
        sb.book_appointment(
            customer_id="C-1001",
            branch_id="marlow_st",
            service_type="notary",
            slot_start="2026-11-05T14:00",
            idempotency_key="k3",
        )["error"]
        == "unavailable"
    )
    assert (
        sb.book_appointment(
            customer_id="C-1001",
            branch_id="mill_rd",
            service_type="notary",
            slot_start="2026-11-04T10:00",
            idempotency_key="k4",
        )["error"]
        == "unavailable"
    )
    assert (
        sb.book_appointment(
            customer_id="C-1001",
            branch_id="mill_rd",
            service_type="notary",
            slot_start="2026-11-03T17:00",
            idempotency_key="k5",
        )["error"]
        == "unavailable"
    )


def test_book_unknown_customer_is_not_found(sb):
    assert (
        sb.book_appointment(
            customer_id="C-9999",
            branch_id="mill_rd",
            service_type="notary",
            slot_start="2026-11-03T10:00",
            idempotency_key="k6",
        )["error"]
        == "not_found"
    )


def test_idempotent_write_returns_original_without_second_write(sb):
    a = sb.book_appointment(
        customer_id="C-1001",
        branch_id="market_st",
        service_type="notary",
        slot_start="2026-11-03T10:00",
        idempotency_key="same",
    )
    b = sb.book_appointment(
        customer_id="C-1001",
        branch_id="market_st",
        service_type="notary",
        slot_start="2026-11-03T10:00",
        idempotency_key="same",
    )
    assert a == b
    assert (
        len(sb.lookup_account(phone="5550142219", date_of_birth="1988-03-12")["appointments"]) == 1
    )


def test_reschedule_moves_and_frees_old_slot(sb):
    r = sb.reschedule_appointment(
        appointment_id="A-2001", new_slot_start="2026-11-06T10:00", idempotency_key="r1"
    )
    assert r["appointment_id"] == "A-2001" and r["slot_start"] == "2026-11-06T10:00"
    assert (
        "14:00"
        in sb.check_availability(branch_id="marlow_st", service_type="notary", date="2026-11-05")[
            "slots"
        ]
    )
    assert (
        sb.reschedule_appointment(
            appointment_id="A-9999", new_slot_start="2026-11-06T10:00", idempotency_key="r2"
        )["error"]
        == "not_found"
    )


def test_cancel_sets_status_and_second_cancel_is_not_found(sb):
    assert sb.cancel_appointment(appointment_id="A-2002", idempotency_key="c1") == {
        "appointment_id": "A-2002",
        "status": "cancelled",
    }
    assert (
        sb.cancel_appointment(appointment_id="A-2002", idempotency_key="c2")["error"] == "not_found"
    )
    assert sb.lookup_account(phone="5550177310", date_of_birth="1992-11-30")["appointments"] == []


def test_block_card_blocks_with_replacement_and_conflicts_when_already_blocked(sb):
    r = sb.block_card(
        customer_id="C-1005",
        card_last4="3307",
        reason="lost",
        replacement="standard",
        idempotency_key="b1",
    )
    assert r == {"card_last4": "3307", "status": "blocked", "replacement": "standard"}
    assert (
        sb.block_card(customer_id="C-1004", card_last4="9912", reason="lost", idempotency_key="b2")[
            "error"
        ]
        == "conflict"
    )
    assert (
        sb.block_card(customer_id="C-1005", card_last4="0000", reason="lost", idempotency_key="b3")[
            "error"
        ]
        == "not_found"
    )


def test_create_ticket_validates_category(sb):
    r = sb.create_ticket(
        customer_id="C-1009", category="fee_dispute", summary="overdraft fee", idempotency_key="t1"
    )
    assert r["ticket_id"].startswith("T-5") and r["category"] == "fee_dispute"
    assert (
        sb.create_ticket(
            customer_id="C-1009", category="refund", summary="x", idempotency_key="t2"
        )["error"]
        == "invalid_argument"
    )


def test_handoff_always_succeeds_and_is_recorded(sb):
    assert sb.request_handoff(reason="suspected_fraud", summary="card 5560") == {
        "status": "transferring",
        "reason": "suspected_fraud",
    }
    assert sb.state()["handoff"] == {"reason": "suspected_fraud", "summary": "card 5560"}


def test_injected_timeout_hits_only_the_nth_call_and_writes_nothing(sb):
    sb.inject([{"tool": "cancel_appointment", "call_index": 1, "error": "timeout"}])
    first = sb.cancel_appointment(appointment_id="A-2001", idempotency_key="x1")
    assert first == {"error": "timeout", "message": "the request timed out"}
    assert (
        sb.lookup_account(phone="5550198823", date_of_birth="1975-07-04")["appointments"][0][
            "appointment_id"
        ]
        == "A-2001"
    )
    assert (
        sb.cancel_appointment(appointment_id="A-2001", idempotency_key="x2")["status"]
        == "cancelled"
    )


def test_call_dispatches_by_name_with_json_args_and_logs(sb):
    out = sb.call(
        "lookup_account", json.dumps({"phone": "5550142219", "date_of_birth": "1988-03-12"})
    )
    assert out["verified"] is True
    assert sb.call("nope", "{}")["error"] == "invalid_argument"
    assert sb.call("lookup_account", "{bad json")["error"] == "invalid_argument"
    assert [c["tool"] for c in sb.log] == ["lookup_account", "nope", "lookup_account"]


def test_diff_reports_only_changes_from_seed(sb):
    assert sb.diff() == {
        "appointments_new": [],
        "appointments_changed": [],
        "cards": [],
        "tickets": [],
    }
    sb.cancel_appointment(appointment_id="A-2002", idempotency_key="d1")
    sb.block_card(customer_id="C-1005", card_last4="3307", reason="lost", idempotency_key="d2")
    d = sb.diff()
    assert d["appointments_changed"] == [
        {"appointment_id": "A-2002", "status": "cancelled", "slot_start": "2026-11-10T11:00"}
    ]
    assert d["cards"] == [{"card_last4": "3307", "status": "blocked", "replacement": "none"}]
