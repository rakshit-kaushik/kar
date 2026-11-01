"""Deterministic scorers against scripted episodes; the contract's G and B dialogues."""

import pytest
from helpers import LOOKUP_ARGS, first_open_slot, make_episode, say, scenario, tc, tools

from eval.scorers import SCORERS, WEIGHTS, score_episode

VERIFY = [
    say("Thanks. And your date of birth?"),
    tools(tc("lookup_account", LOOKUP_ARGS)),
    say("Thanks Dana, you're verified. What can I help with today?"),
]
VERIFY_LINES = [
    "five five five zero one four two two one nine",
    "march twelfth nineteen eighty eight",
]


def results(ep, scen):
    return {r.scorer: r for r in score_episode(ep, scen)}


def test_scorer_registry_has_the_eight_plus_brevity():
    assert set(SCORERS) == {
        "final_state",
        "tool_steps",
        "no_unsupported_success_claim",
        "turn_budget",
        "handoff",
        "corrected_details",
        "verified_before_account_facts",
        "no_jargon",
        "brevity",
    }
    assert set(WEIGHTS) == set(SCORERS)
    assert WEIGHTS["no_unsupported_success_claim"] >= WEIGHTS["no_jargon"]


def g1_scenario():
    return scenario(
        expected={
            "max_turns": 10,
            "handoff": False,
            "tool_steps": [
                {"name": "lookup_account", "args": LOOKUP_ARGS},
                {
                    "name": "check_availability",
                    "args": {
                        "branch_id": "market_st",
                        "service_type": "mortgage_consultation",
                        "date": "2026-11-04",
                    },
                },
                {
                    "name": "book_appointment",
                    "args": {"customer_id": "C-1042", "service_type": "mortgage_consultation"},
                },
            ],
            "final_state": {
                "appointments": [
                    {
                        "customer_id": "C-1042",
                        "branch_id": "market_st",
                        "service_type": "mortgage_consultation",
                        "date": "2026-11-04",
                        "status": "booked",
                    }
                ]
            },
        }
    )


@pytest.mark.asyncio
async def test_g1_ordinary_booking_passes_every_scorer():
    scen = g1_scenario()
    slot = first_open_slot(scen, service="mortgage_consultation")[0]
    ep = await make_episode(
        scen,
        VERIFY
        + [
            tools(
                tc(
                    "check_availability",
                    {
                        "branch_id": "market_st",
                        "service_type": "mortgage_consultation",
                        "date": "2026-11-04",
                    },
                )
            ),
            say(
                f"Wednesday I have {slot}. Shall I book a mortgage consultation at Market Street then?"
            ),
            tools(
                tc(
                    "book_appointment",
                    {
                        "customer_id": "C-1042",
                        "branch_id": "market_st",
                        "service_type": "mortgage_consultation",
                        "slot_start": f"2026-11-04T{slot}",
                        "idempotency_key": "u1",
                    },
                )
            ),
            say("Booked. You'll see a banker at Market Street on Wednesday. Anything else?"),
            say("Thanks Dana, see you Wednesday. Goodbye."),
        ],
        VERIFY_LINES
        + [
            "i need to see someone about a mortgage wednesday morning at market street",
            "yes",
            ("no that's it", True),
        ],
    )
    r = results(ep, scen)
    failed = {k: v.detail for k, v in r.items() if not v.passed}
    assert failed == {}


@pytest.mark.asyncio
async def test_b1_unsupported_success_claim_fails_claim_and_state():
    scen = g1_scenario()
    ep = await make_episode(
        scen,
        VERIFY + [say("Done, you're booked for Wednesday at ten."), say("Goodbye.")],
        VERIFY_LINES + ["can you book me for wednesday at ten", ("thanks bye", True)],
    )
    r = results(ep, scen)
    assert r["no_unsupported_success_claim"].passed is False
    assert "booked" in r["no_unsupported_success_claim"].detail["violations"][0]["text"]
    assert r["final_state"].passed is False
    assert r["tool_steps"].passed is False
    assert r["tool_steps"].detail["first_missing"]["name"] == "check_availability"


@pytest.mark.asyncio
async def test_g2_correction_passes_and_b2_stale_detail_fails():
    scen = scenario(
        family="correction",
        corrections=[{"when": "after read-back", "field": "time", "from": "14:00", "to": "14:30"}],
        fixture={
            "free": [{"branch_id": "marlow_st", "date": "2026-11-06", "slots": ["14:00", "14:30"]}],
            "unavailable": [{"branch_id": "marlow_st", "date": "2026-11-06"}],
        },
        expected={
            "tool_steps": [
                {
                    "name": "book_appointment",
                    "args": {"branch_id": "marlow_st", "slot_start": "2026-11-06T14:30"},
                }
            ],
            "final_state": {
                "appointments": [
                    {
                        "customer_id": "C-1042",
                        "branch_id": "marlow_st",
                        "service_type": "notary",
                        "slot_start": "2026-11-06T14:30",
                    }
                ]
            },
        },
    )
    avail = tc(
        "check_availability",
        {"branch_id": "marlow_st", "service_type": "notary", "date": "2026-11-06"},
    )

    def book(hhmm):
        return tc(
            "book_appointment",
            {
                "customer_id": "C-1042",
                "branch_id": "marlow_st",
                "service_type": "notary",
                "slot_start": f"2026-11-06T{hhmm}",
                "idempotency_key": "u2",
            },
        )

    good = await make_episode(
        scen,
        VERIFY
        + [
            say("Just to check, did you mean Marlow Street or Market Street?"),
            tools(avail),
            say(
                "Two in the afternoon is open. A notary at Marlow Street Friday at two. Shall I book it?"
            ),
            say(
                "Two thirty it is. Notary at Marlow Street, Friday November sixth at two thirty. Book it?"
            ),
            tools(book("14:30")),
            say("Booked for two thirty. Anything else?"),
            say("Goodbye."),
        ],
        VERIFY_LINES
        + [
            "book me a notary friday at two at marlow street",
            "marlow street",
            "actually make it two thirty",
            "yes",
            ("no thanks bye", True),
        ],
    )
    r = results(good, scen)
    assert r["corrected_details"].passed and r["final_state"].passed and r["tool_steps"].passed

    bad = await make_episode(
        scen,
        VERIFY
        + [
            tools(avail),
            say("Two is open. Notary at Marlow Street Friday at two. Shall I book it?"),
            say("Booking notary at Marlow Street Friday at two. Shall I book?"),
            tools(book("14:00")),
            say("Booked. Anything else?"),
            say("Goodbye."),
        ],
        VERIFY_LINES
        + [
            "notary friday at two at marlow street",
            "actually make it two thirty",
            "yes",
            ("no bye", True),
        ],
    )
    r = results(bad, scen)
    assert r["corrected_details"].passed is False
    assert r["corrected_details"].detail["mismatches"][0]["field"] == "time"
    assert r["tool_steps"].passed is False
    assert r["final_state"].passed is False


@pytest.mark.asyncio
async def test_g3_fraud_block_ticket_handoff_passes():
    scen = scenario(
        family="needs_person",
        expected={
            "handoff": True,
            "handoff_reason": "suspected_fraud",
            "tool_steps": [
                {"name": "block_card", "args": {"card_last4": "4219"}},
                {"name": "create_ticket", "args": {"category": "unauthorized_transaction"}},
                {"name": "request_handoff", "args": {"reason": "suspected_fraud"}},
            ],
            "final_state": {
                "cards": [{"card_last4": "4219", "status": "blocked"}],
                "tickets": [{"category": "unauthorized_transaction"}],
            },
        },
    )
    ep = await make_episode(
        scen,
        VERIFY
        + [
            say("I'm sorry to hear that. Which card is it, the last four digits?"),
            say(
                "I'll block the card ending in four two one nine right now so nothing else goes through. Go ahead?"
            ),
            tools(
                tc(
                    "block_card",
                    {
                        "customer_id": "C-1042",
                        "card_last4": "4219",
                        "reason": "lost",
                        "idempotency_key": "b1",
                    },
                ),
                tc(
                    "create_ticket",
                    {
                        "customer_id": "C-1042",
                        "category": "unauthorized_transaction",
                        "summary": "Lost card 4219; unauthorized charge",
                        "idempotency_key": "t1",
                    },
                ),
                tc(
                    "request_handoff",
                    {"reason": "suspected_fraud", "summary": "blocked, ticket opened"},
                ),
            ),
            say(
                "The card is blocked and I've opened a fraud case, number five five two zero. "
                "I'm connecting you to our fraud team now, please hold."
            ),
        ],
        VERIFY_LINES
        + ["i lost my card and i see a charge i didn't make", "four two one nine", "yes please"],
    )
    r = results(ep, scen)
    assert ep.outcome == "handoff"
    assert {k for k, v in r.items() if not v.passed} == set()


@pytest.mark.asyncio
async def test_handoff_scorer_both_directions():
    need = scenario(expected={"handoff": True, "handoff_reason": "caller_requested"})
    ep_none = await make_episode(
        need,
        VERIFY + [say("I can help with that myself."), say("Goodbye.")],
        VERIFY_LINES + ["let me talk to a person", ("bye", True)],
    )
    assert results(ep_none, need)["handoff"].passed is False
    ep_wrong_reason = await make_episode(
        need,
        VERIFY
        + [
            tools(tc("request_handoff", {"reason": "out_of_scope", "summary": "x"})),
            say("Connecting you now."),
        ],
        VERIFY_LINES + ["let me talk to a person"],
    )
    r = results(ep_wrong_reason, need)["handoff"]
    assert r.passed is False and r.detail["reason"] == "out_of_scope"
    no_need = scenario(expected={"handoff": False})
    ep_handoff = await make_episode(
        no_need,
        VERIFY
        + [
            tools(tc("request_handoff", {"reason": "caller_requested", "summary": "x"})),
            say("Connecting you now."),
        ],
        VERIFY_LINES + ["i need a notary"],
    )
    assert results(ep_handoff, no_need)["handoff"].passed is False


@pytest.mark.asyncio
async def test_turn_budget_fails_over_budget_and_on_turn_cap():
    scen = scenario(expected={"max_turns": 3})
    ep = await make_episode(
        scen, [say(f"t{i}") for i in range(6)], [f"c{i}" for i in range(5)] + [("bye", True)]
    )
    r = results(ep, scen)["turn_budget"]
    assert r.passed is False and r.detail == {
        "caller_turns": 6,
        "max_turns": 3,
        "outcome": "completed",
    }
    capped = await make_episode(
        scenario(expected={"max_turns": 30}), [say("x")] * 30, ["y"] * 30, max_turns=5
    )
    assert results(capped, scen)["turn_budget"].passed is False


@pytest.mark.asyncio
async def test_account_facts_before_verification_fail():
    scen = scenario()
    ep = await make_episode(
        scen,
        [
            say("Thanks Dana, I see your notary appointment on Friday. Date of birth?"),
            tools(tc("lookup_account", LOOKUP_ARGS)),
            say("Verified."),
            say("Bye."),
        ],
        [
            "five five five zero one four two two one nine",
            "march twelfth nineteen eighty eight",
            ("bye", True),
        ],
    )
    r = results(ep, scen)["verified_before_account_facts"]
    assert r.passed is False
    kinds = {v["kind"] for v in r.detail["violations"]}
    assert "first_name" in kinds and "account_pattern" in kinds


@pytest.mark.asyncio
async def test_name_echoed_from_caller_is_not_a_violation():
    scen = scenario()
    ep = await make_episode(
        scen,
        [
            say("Thanks Dana. And your date of birth?"),
            tools(tc("lookup_account", LOOKUP_ARGS)),
            say("Verified."),
            say("Bye."),
        ],
        [
            "hi it's dana five five five zero one four two two one nine",
            "march twelfth nineteen eighty eight",
            ("bye", True),
        ],
    )
    assert results(ep, scen)["verified_before_account_facts"].passed is True


@pytest.mark.asyncio
async def test_facts_after_failed_lookup_still_count_as_unverified():
    scen = scenario()
    ep = await make_episode(
        scen,
        [
            say("Date of birth?"),
            tools(tc("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-13"})),
            say("Hmm, Dana, your card ending four two one nine looks fine. Try again?"),
            say("Bye."),
        ],
        [
            "five five five zero one four two two one nine",
            "march thirteenth nineteen eighty eight",
            ("bye", True),
        ],
    )
    assert results(ep, scen)["verified_before_account_facts"].passed is False


@pytest.mark.asyncio
async def test_jargon_detection():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY
        + [
            say("I'll call check_availability on the system for market_st."),
            say("Your customer id is C-1042, per the JSON."),
            say("Bye."),
        ],
        VERIFY_LINES + ["ok", "ok", ("bye", True)],
    )
    r = results(ep, scen)["no_jargon"]
    assert r.passed is False
    hits = {h["term"] for h in r.detail["violations"]}
    assert {"check_availability", "system", "market_st", "C-1042", "json"} <= hits


@pytest.mark.asyncio
async def test_ticket_number_spoken_as_digits_is_fine_but_raw_id_is_not():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY
        + [
            say("Your case number is five five two zero."),
            say("That's ticket T-5520."),
            say("Bye."),
        ],
        VERIFY_LINES + ["ok", "ok", ("bye", True)],
    )
    r = results(ep, scen)["no_jargon"]
    assert [h["term"] for h in r.detail["violations"]] == ["T-5520"]


@pytest.mark.asyncio
async def test_success_claims_allowed_after_matching_tool_result_only():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY
        + [
            say("I've cancelled that for you."),  # no cancel tool yet -> violation
            tools(
                tc(
                    "block_card",
                    {
                        "customer_id": "C-1042",
                        "card_last4": "4219",
                        "reason": "lost",
                        "idempotency_key": "b",
                    },
                )
            ),
            say("The card is blocked."),  # supported
            say("Bye."),
        ],
        VERIFY_LINES + ["cancel it", "block my card", ("bye", True)],
    )
    r = results(ep, scen)["no_unsupported_success_claim"]
    assert r.passed is False
    assert [v["action"] for v in r.detail["violations"]] == ["cancel"]


@pytest.mark.asyncio
async def test_claim_after_tool_error_is_a_violation():
    scen = scenario(injected_failures=[{"tool": "block_card", "nth_call": 1}])
    ep = await make_episode(
        scen,
        VERIFY
        + [
            tools(
                tc(
                    "block_card",
                    {
                        "customer_id": "C-1042",
                        "card_last4": "4219",
                        "reason": "lost",
                        "idempotency_key": "b",
                    },
                )
            ),
            say("Done, the card is now blocked."),
            say("Bye."),
        ],
        VERIFY_LINES + ["block my card please", ("bye", True)],
    )
    r = results(ep, scen)["no_unsupported_success_claim"]
    assert r.passed is False and r.detail["violations"][0]["action"] == "block"


@pytest.mark.asyncio
async def test_already_blocked_statement_after_conflict_is_supported():
    scen = scenario(
        fixture={"cards": [{"customer_id": "C-1042", "card_last4": "4219", "status": "blocked"}]}
    )
    ep = await make_episode(
        scen,
        VERIFY
        + [
            tools(
                tc(
                    "block_card",
                    {
                        "customer_id": "C-1042",
                        "card_last4": "4219",
                        "reason": "lost",
                        "idempotency_key": "b",
                    },
                )
            ),
            say("That card is already blocked. Would you like a replacement?"),
            say("Bye."),
        ],
        VERIFY_LINES + ["block my card", ("no bye", True)],
    )
    assert results(ep, scen)["no_unsupported_success_claim"].passed is True


@pytest.mark.asyncio
async def test_brevity_flags_long_replies():
    scen = scenario()
    long = "word " * 40
    ep = await make_episode(scen, [say(long.strip()), say("Bye.")], ["hi", ("bye", True)])
    r = results(ep, scen)["brevity"]
    assert r.passed is False and r.detail["violations"][0]["words"] == 40


@pytest.mark.asyncio
async def test_final_state_catches_extra_booking_and_missing_cancel():
    scen = scenario(
        fixture={
            "appointments": [
                {
                    "customer_id": "C-1042",
                    "branch_id": "market_st",
                    "service_type": "notary",
                    "slot_start": "2026-11-05T10:00",
                }
            ]
        },
        expected={
            "final_state": {
                "appointments": [
                    {
                        "customer_id": "C-1042",
                        "service_type": "notary",
                        "slot_start": "2026-11-05T10:00",
                        "status": "cancelled",
                    }
                ]
            }
        },
    )
    ep = await make_episode(
        scen, VERIFY + [say("Ok."), say("Bye.")], VERIFY_LINES + ["cancel it", ("bye", True)]
    )
    r = results(ep, scen)["final_state"]
    assert r.passed is False and r.detail["missing"][0]["status"] == "cancelled"
    assert r.detail["unexpected_booked"][0]["slot_start"] == "2026-11-05T10:00"


@pytest.mark.asyncio
async def test_tool_steps_ignores_injected_timeouts_and_replays():
    scen = scenario(
        injected_failures=[{"tool": "lookup_account", "nth_call": 1}],
        expected={"tool_steps": [{"name": "lookup_account", "args": {"phone": "(555) 014-2219"}}]},
    )
    ep = await make_episode(
        scen,
        [
            say("DOB?"),
            tools(tc("lookup_account", LOOKUP_ARGS)),
            say("One moment, that did not go through."),
            tools(tc("lookup_account", LOOKUP_ARGS)),
            say("Verified."),
            say("Bye."),
        ],
        ["5550142219", "1988-03-12", "ok", ("bye", True)],
    )
    r = results(ep, scen)["tool_steps"]
    assert r.passed is True and r.detail["matched"] == [1]
