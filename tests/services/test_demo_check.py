"""Scenario checker for the demo rehearsal (demo/scenarios.md): final state from sessions.state + turn events -> reasons."""
from services.demo.check import check_call

DANA, MARCUS = "C-1001", "C-1002"


def _ev(event, ts, payload, call="c"):
    return (call, 0, "m", event, ts, payload)


def booked(slot, customer=DANA, branch="market_st", service="mortgage_consultation"):
    args = {"customer_id": customer, "branch_id": branch, "service_type": service, "slot_start": slot, "idempotency_key": "k"}
    return {"tool": "book_appointment", "args": args, "result": {"appointment_id": "A-3001", "status": "booked", **args}}


def ticket(category="unauthorized_transaction", customer=MARCUS):
    return {"tool": "create_ticket", "args": {"customer_id": customer, "category": category, "summary": "charge"}, "result": {"ticket_id": "T-9001", "category": category, "status": "open"}}


def ok_tool(name, ts):
    return _ev("tool_end", ts, {"name": name, "ok": True, "ms": 1})


def said(text, ts):
    return _ev("transcript", ts, {"role": "assistant", "text": text, "structural": False})


ORDINARY_EVENTS = [ok_tool("lookup_account", 10), said("Which time works for you?", 20), ok_tool("book_appointment", 30), said("I've booked your mortgage consultation for Tuesday.", 40)]


def test_ordinary_passes_with_one_booking_at_the_agreed_slot():
    assert check_call("ordinary", {"completed": [booked("2026-11-03T09:30")], "handoff": None}, ORDINARY_EVENTS) == []


def test_ordinary_fails_on_a_missing_booking_and_a_claim_without_a_tool():
    reasons = check_call("ordinary", {"completed": [], "handoff": None}, [ok_tool("lookup_account", 10), said("I booked your mortgage consultation.", 40)])
    assert any("0 bookings" in r for r in reasons) and any("success claim" in r for r in reasons)


def test_ordinary_fails_when_an_id_is_spoken():
    evs = ORDINARY_EVENTS + [said("Your customer ID is C-1001.", 50)]
    assert any("spoke ids: C-1001" in r for r in check_call("ordinary", {"completed": [booked("2026-11-03T09:30")], "handoff": None}, evs))


def test_correction_needs_wednesday_and_no_tuesday():
    assert check_call("correction", {"completed": [booked("2026-11-04T09:30")], "handoff": None}, ORDINARY_EVENTS) == []
    reasons = check_call("correction", {"completed": [booked("2026-11-03T09:30")], "handoff": None}, ORDINARY_EVENTS)
    assert any("stale" in r for r in reasons)


def test_timeout_needs_a_failed_first_booking_attempt_then_a_successful_retry():
    """The fault sits on book_appointment: run 2 books directly when the caller names the time and never calls check_availability on
    this path (0 calls in 7 live bookings on Nov 1), so a check_availability fault would never be consumed."""
    fail = _ev("tool_end", 20, {"name": "book_appointment", "ok": False, "error": "timeout"})
    evs = [ok_tool("lookup_account", 10), fail, said("The system is slow, let me try again.", 25), ok_tool("book_appointment", 30), said("I've booked it.", 50)]
    assert check_call("timeout", {"completed": [booked("2026-11-03T09:30")], "handoff": None}, evs) == []
    no_fault = [ok_tool("lookup_account", 10), ok_tool("book_appointment", 40)]
    assert any("fault not injected" in r for r in check_call("timeout", {"completed": [booked("2026-11-03T09:30")], "handoff": None}, no_fault))
    both_failed = [ok_tool("lookup_account", 10), fail, _ev("tool_end", 30, {"name": "book_appointment", "ok": False, "error": "timeout"})]
    assert check_call("timeout", {"completed": [], "handoff": {"requested": True, "reason": "tool_failure", "summary": "x"}}, both_failed) == []
    # one failed attempt and no retry is not recovery, even when the call ends in a handoff (Nov 1 08:36: the model asked "shall I try
    # again?", the script said goodbye, and the claim guard turned the model's transfer line into the handoff)
    no_retry = [ok_tool("lookup_account", 10), fail, ok_tool("request_handoff", 40)]
    assert check_call("timeout", {"completed": [], "handoff": {"requested": True, "reason": "out_of_scope", "summary": "x"}}, no_retry) == ["no retry of book_appointment after the timeout"]
    assert any("no successful retry" in r for r in check_call("timeout", {"completed": [], "handoff": None}, both_failed))


def test_handoff_person_needs_the_transfer_the_caller_asked_for_and_nothing_else():
    evs = [ok_tool("lookup_account", 10), ok_tool("request_handoff", 30), said("Of course. Let me get a colleague to help you.", 40)]
    good = {"completed": [], "handoff": {"requested": True, "reason": "caller_requested", "summary": "x"}}
    assert check_call("handoff_person", good, evs) == []
    assert any("no handoff" in r for r in check_call("handoff_person", {"completed": [], "handoff": None}, evs))
    assert any("expected caller_requested" in r for r in check_call("handoff_person", {"completed": [], "handoff": {"requested": True, "reason": "out_of_scope"}}, evs))
    assert any("ticket" in r for r in check_call("handoff_person", {"completed": [ticket(customer=DANA)], "handoff": good["handoff"]}, evs))


def test_unfinished_lists_the_calls_whose_state_is_not_final_yet():
    from services.demo.check import unfinished
    assert unfinished([("a", None), ("b", "2026-11-01 08:26:20")]) == ["a"]


def test_interruption_requires_that_an_interruption_happened():
    state = {"completed": [booked("2026-11-03T09:30")], "handoff": None}
    assert any("no interruption" in r for r in check_call("interruption", state, ORDINARY_EVENTS))
    assert check_call("interruption", state, ORDINARY_EVENTS + [_ev("caller_interrupted", 15, {"turn": 5})]) == []


def test_handoff_needs_the_ticket_and_the_transfer_and_no_false_promise():
    evs = [ok_tool("lookup_account", 10), ok_tool("create_ticket", 20), ok_tool("request_handoff", 30), said("A specialist will take over now.", 40)]
    good = {"completed": [ticket()], "handoff": {"requested": True, "reason": "suspected_fraud", "summary": "x"}}
    assert check_call("handoff", good, evs) == []
    reasons = check_call("handoff", good, evs + [said("The charge has been reversed and your card is blocked.", 50)])
    assert any("promised what no tool does" in r for r in reasons)
    assert any("no handoff" in r for r in check_call("handoff", {"completed": [ticket()], "handoff": None}, evs))
    assert any("expected one unauthorized_transaction" in r for r in check_call("handoff", {"completed": [], "handoff": good["handoff"]}, evs))


def test_call_summary_reports_guard_interventions_and_the_tts_vendor():
    from services.demo.check import call_summary
    evs = [_ev("loop_guard", 5, {"action": "rephrase"}), _ev("claim_guard", 7, {"claimed": "transferring you", "reason": "out_of_scope"}),
           _ev("call_end", 9, {"outcome": "completed", "tts_vendor": "deepgram"})]
    assert call_summary(evs) == {"guard": 1, "claim_guard": 1, "retry": 0, "policy": 0, "tts": "deepgram"}
    assert call_summary([]) == {"guard": 0, "claim_guard": 0, "retry": 0, "policy": 0, "tts": "?"}
    assert call_summary([_ev("policy_handoff", 3, {"reason": "suspected_fraud"})])["policy"] == 1  # the contract rule's handoff, counted apart
    assert call_summary([_ev("tool_retry", 3, {"name": "book_appointment", "error": "timeout"})])["retry"] == 1  # the executor's retry, counted apart
