"""Sandbox dispatcher + the two read tools, every error path from CONTRACT.md §4."""

from datetime import date

import pytest

from eval.sandbox import Sandbox

TODAY = date(2026, 11, 2)  # Monday


@pytest.fixture
def sb():
    return Sandbox(seed=42, today=TODAY)


def test_unknown_tool_is_invalid_argument(sb):
    r = sb.call("make_coffee", {})
    assert r == {"error": "invalid_argument", "message": "Unknown tool: make_coffee"}


def test_lookup_verified_returns_profile_and_logs(sb):
    r = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    assert r["verified"] is True
    assert r["customer_id"] == "C-1042"
    assert r["first_name"] == "Dana"
    assert r["branch_id"] == "market_st"
    assert r["appointments"] == []
    assert r["cards"] == [{"card_last4": "4219", "status": "active"}]
    assert len(sb.tool_log) == 1
    entry = sb.tool_log[0]
    assert entry.tool == "lookup_account"
    assert entry.args == {"phone": "5550142219", "date_of_birth": "1988-03-12"}
    assert entry.ok is True and entry.error is None and entry.injected is False


def test_lookup_normalizes_phone_punctuation(sb):
    r = sb.call("lookup_account", {"phone": "(555) 014-2219", "date_of_birth": "1988-03-12"})
    assert r["verified"] is True


def test_lookup_not_found_for_unknown_phone(sb):
    r = sb.call("lookup_account", {"phone": "5559990000", "date_of_birth": "1988-03-12"})
    assert r["error"] == "not_found"
    assert sb.tool_log[-1].ok is False and sb.tool_log[-1].error == "not_found"


def test_lookup_not_verified_for_wrong_dob(sb):
    r = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-13"})
    assert r["error"] == "not_verified"


def test_lookup_not_verified_for_malformed_dob(sb):
    r = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "march twelfth"})
    assert r["error"] == "not_verified"


def test_lookup_missing_argument_is_not_found(sb):
    r = sb.call("lookup_account", {"phone": "5550142219"})
    assert r["error"] == "not_found"


def test_lookup_shows_existing_appointments(sb):
    sb.apply_fixture(
        {
            "appointments": [
                {
                    "customer_id": "C-1042",
                    "branch_id": "marlow_st",
                    "service_type": "notary",
                    "slot_start": "2026-11-05T14:00",
                }
            ]
        }
    )
    r = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    assert len(r["appointments"]) == 1
    ap = r["appointments"][0]
    assert ap["appointment_id"].startswith("A-")
    assert (ap["branch_id"], ap["service_type"], ap["slot_start"]) == (
        "marlow_st",
        "notary",
        "2026-11-05T14:00",
    )


def test_injected_timeout_fires_on_nth_call_only(sb):
    sb.inject_timeout("lookup_account", nth_call=1)
    r1 = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    assert r1 == {"error": "timeout", "message": "The request timed out."}
    assert sb.tool_log[-1].injected is True
    r2 = sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    assert r2["verified"] is True


def test_check_availability_lists_open_slots_only(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "2026-11-04"},
    )
    assert set(r) == {"slots", "branch_id", "date"}
    assert 2 <= len(r["slots"]) <= 8
    assert r["slots"] == sorted(r["slots"])
    taken = {
        a.slot_start.strftime("%H:%M")
        for a in sb.appointments
        if a.branch_id == "market_st" and a.slot_start.date() == date(2026, 11, 4)
    }
    assert not (set(r["slots"]) & taken)


def test_check_availability_invalid_branch(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "main_st", "service_type": "notary", "date": "2026-11-04"},
    )
    assert r["error"] == "invalid_argument"
    assert "branch_id" in r["message"]


def test_check_availability_invalid_service(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "haircut", "date": "2026-11-04"},
    )
    assert r["error"] == "invalid_argument"


def test_check_availability_malformed_date(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "next tuesday"},
    )
    assert r["error"] == "invalid_argument"


def test_check_availability_unknown_key_is_invalid_argument(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "2026-11-04", "time": "09:00"},
    )
    assert r["error"] == "invalid_argument"


def test_check_availability_sunday_is_unavailable(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "2026-11-08"},
    )
    assert r["error"] == "unavailable"
    assert "closed" in r["message"].lower()


def test_check_availability_past_date_is_unavailable(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "2026-10-30"},
    )
    assert r["error"] == "unavailable"


def test_check_availability_beyond_horizon_is_unavailable(sb):
    r = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "2026-12-20"},
    )
    assert r["error"] == "unavailable"


def test_check_availability_full_day_returns_empty_slots(sb):
    sb.apply_fixture({"unavailable": [{"branch_id": "mill_rd", "date": "2026-11-05"}]})
    r = sb.call(
        "check_availability",
        {"branch_id": "mill_rd", "service_type": "notary", "date": "2026-11-05"},
    )
    assert r == {"slots": [], "branch_id": "mill_rd", "date": "2026-11-05"}


def test_check_availability_timeout_injection(sb):
    sb.inject_timeout("check_availability", nth_call=2)
    args = {"branch_id": "market_st", "service_type": "notary", "date": "2026-11-04"}
    assert "slots" in sb.call("check_availability", args)
    assert sb.call("check_availability", args)["error"] == "timeout"
    assert "slots" in sb.call("check_availability", args)


def test_reset_restores_seed_and_clears_log(sb):
    sb.apply_fixture({"unavailable": [{"branch_id": "mill_rd", "date": "2026-11-05"}]})
    sb.call("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    n_before = len(sb.appointments)
    sb.reset()
    assert sb.tool_log == []
    assert len(sb.appointments) == n_before  # fixture is part of the episode baseline
    sb2 = Sandbox(seed=42, today=TODAY)
    assert len(sb.appointments) > len(sb2.appointments)


def test_fixture_free_opens_named_slots_and_unavailable_fills_the_rest(sb):
    sb.apply_fixture(
        {
            "free": [{"branch_id": "marlow_st", "date": "2026-11-06", "slots": ["14:00", "14:30"]}],
            "unavailable": [{"branch_id": "marlow_st", "date": "2026-11-06"}],
        }
    )
    r = sb.call(
        "check_availability",
        {"branch_id": "marlow_st", "service_type": "notary", "date": "2026-11-06"},
    )
    assert r["slots"] == ["14:00", "14:30"]
