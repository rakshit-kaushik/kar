"""services/common/claims.py: a spoken transfer is a success claim about request_handoff, like a booking is about book_appointment
(the contract and E's eval count a transfer announced without the call as unsupported)."""
from services.common.claims import unsupported_claims


def _ev(event, ts, payload):
    return ("c", 1, "m", event, ts, payload)


def test_transfer_announced_without_request_handoff_is_an_unsupported_claim():
    evs = [_ev("tool_end", 1, {"name": "create_ticket", "ok": True}), _ev("transcript", 2, {"role": "assistant", "text": "I've opened your ticket and I'm transferring you to a specialist."})]
    assert unsupported_claims(evs) == 1


def test_transfer_after_request_handoff_is_supported():
    evs = [_ev("tool_end", 1, {"name": "request_handoff", "ok": True}), _ev("transcript", 2, {"role": "assistant", "text": "I'm transferring you to a specialist now."})]
    assert unsupported_claims(evs) == 0


def test_ticket_claim_with_your_ticket_phrasing_is_counted():
    evs = [_ev("transcript", 2, {"role": "assistant", "text": "I have opened your ticket."})]
    assert unsupported_claims(evs) == 1
