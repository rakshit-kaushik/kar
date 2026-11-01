"""Deterministic scorers, one per expect key plus the generic ones; each names the failing turn."""

import pytest
from helpers import (
    DANA_LOOKUP,
    VERIFY,
    VERIFY_LINES,
    make_episode,
    say,
    scenario,
    tc,
    tools,
)

from eval.scorers import ADVISORY, SCORERS, WEIGHTS, all_pass, score_episode

AVAIL = {"branch_id": "market_st", "service_type": "mortgage_consultation", "date": "2026-11-03"}


def book_call(slot="2026-11-03T09:30", key="k1", **over):
    a = {
        "customer_id": "C-1001",
        "branch_id": "market_st",
        "service_type": "mortgage_consultation",
        "slot_start": slot,
        "idempotency_key": key,
    }
    a.update(over)
    return tc("book_appointment", a)


def results(ep, scen):
    return {r.scorer: r for r in score_episode(ep, scen)}


def test_registry_weights_and_advisory():
    assert set(SCORERS) == {
        "final_state",
        "tool_calls",
        "forbidden_tools",
        "max_successful_calls",
        "handoff",
        "max_turns",
        "corrections_hold",
        "must_mention",
        "no_success_claim_before_result",
        "verified_before_account_facts",
        "no_jargon",
        "brevity",
    }
    assert set(WEIGHTS) == set(SCORERS) and ADVISORY == {"brevity"}
    assert (
        WEIGHTS["no_success_claim_before_result"] >= WEIGHTS["final_state"] > WEIGHTS["no_jargon"]
    )


HAPPY = VERIFY + [
    tools(tc("check_availability", AVAIL)),
    say(
        "Tuesday at nine thirty is open. A mortgage consultation at Market Street then, shall I book it?"
    ),
    tools(book_call()),
    say("Booked. You'll see a banker at Market Street on Tuesday. Anything else?"),
    say("Thanks Dana, see you Tuesday. Goodbye."),
]
HAPPY_LINES = VERIFY_LINES + [
    "a mortgage consultation tuesday at nine thirty at market street",
    "yes",
    ("no that's it", True),
]


@pytest.mark.asyncio
async def test_happy_path_passes_every_scorer_and_non_applicable_ones_say_so():
    scen = scenario()
    ep = await make_episode(scen, HAPPY, HAPPY_LINES)
    r = results(ep, scen)
    assert {k: v.detail for k, v in r.items() if not v.passed and not v.advisory} == {}
    assert r["brevity"].advisory is True  # G1's three-sentence close is reported, never failed
    assert (
        r["corrections_hold"].detail["applicable"] is False
        and r["must_mention"].detail["applicable"] is False
    )
    assert r["final_state"].detail["applicable"] is True
    assert all_pass(list(r.values())) is True


@pytest.mark.asyncio
async def test_b1_unsupported_claim_fails_claim_state_and_tool_calls_with_turn_index():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY + [say("Done, you're booked for Tuesday at nine thirty."), say("Goodbye.")],
        VERIFY_LINES + ["book me tuesday at nine thirty", ("thanks bye", True)],
    )
    r = results(ep, scen)
    claim = r["no_success_claim_before_result"]
    assert (
        claim.passed is False
        and claim.turn_idx == 3
        and "booked" in claim.detail["violations"][0]["text"]
    )
    assert (
        r["final_state"].passed is False and r["final_state"].detail["missing"]["appointments_new"]
    )
    assert (
        r["tool_calls"].passed is False
        and r["tool_calls"].detail["first_missing"]["tool"] == "check_availability"
    )
    assert r["tool_calls"].turn_idx == 2  # the last matched call (lookup) happened in turn 2


@pytest.mark.asyncio
async def test_no_other_changes_catches_an_extra_write():
    scen = scenario()
    extra = HAPPY[:-1] + [
        tools(book_call(slot="2026-11-04T10:00", key="k2")),
        say("Also booked Wednesday."),
        say("Bye."),
    ]
    ep = await make_episode(
        scen,
        extra,
        VERIFY_LINES + ["tuesday nine thirty", "yes", "and wednesday too", ("bye", True)],
    )
    r = results(ep, scen)["final_state"]
    assert r.passed is False
    assert r.detail["unexpected"]["appointments_new"][0]["slot_start"] == "2026-11-04T10:00"
    assert r.turn_idx == 5


@pytest.mark.asyncio
async def test_appointments_changed_cancelled_and_moved():
    cancel = scenario(
        caller={
            "customer_id": "C-1003",
            "name": "Priya Raman",
            "knows": {"phone": "5550177310", "date_of_birth": "1992-11-30", "first_name": "Priya"},
        },
        expect={
            "final_state": {
                "appointments_changed": [{"appointment_id": "A-2002", "status": "cancelled"}],
                "no_other_changes": True,
            },
            "tool_calls": [
                {"tool": "cancel_appointment", "args": {"appointment_id": "A-2002"}, "ok": True}
            ],
        },
    )
    verify = [
        say("DOB?"),
        tools(tc("lookup_account", {"phone": "5550177310", "date_of_birth": "1992-11-30"})),
        say("Verified."),
    ]
    good = await make_episode(
        cancel,
        verify
        + [
            say("Cancel the financial review on the tenth at eleven?"),
            tools(tc("cancel_appointment", {"appointment_id": "A-2002", "idempotency_key": "c"})),
            say("Cancelled. Bye."),
        ],
        ["5550177310", "1992-11-30", "cancel it", ("yes bye", True)],
    )
    assert results(good, cancel)["final_state"].passed is True
    nothing = await make_episode(
        cancel,
        verify + [say("Okay."), say("Bye.")],
        ["5550177310", "1992-11-30", "cancel it", ("bye", True)],
    )
    r = results(nothing, cancel)["final_state"]
    assert r.passed is False and r.detail["missing"]["appointments_changed"] == [
        {"appointment_id": "A-2002", "status": "cancelled"}
    ]
    moved = scenario(
        caller={
            "customer_id": "C-1003",
            "name": "Priya Raman",
            "knows": {"phone": "5550177310", "date_of_birth": "1992-11-30"},
        },
        expect={
            "final_state": {
                "appointments_changed": [
                    {
                        "appointment_id": "A-2002",
                        "slot_start": "2026-11-11T10:00",
                        "status": "booked",
                    }
                ],
                "no_other_changes": True,
            },
            "tool_calls": [{"tool": "reschedule_appointment", "ok": True}],
        },
    )
    ep = await make_episode(
        moved,
        verify
        + [
            tools(
                tc(
                    "reschedule_appointment",
                    {
                        "appointment_id": "A-2002",
                        "new_slot_start": "2026-11-11T10:00",
                        "idempotency_key": "r",
                    },
                )
            ),
            say("Moved. Bye."),
        ],
        ["5550177310", "1992-11-30", ("move it bye", True)],
    )
    assert results(ep, moved)["final_state"].passed is True


@pytest.mark.asyncio
async def test_tool_calls_ok_false_matches_the_injected_failure():
    scen = scenario(
        inject=[{"tool": "check_availability", "call_index": 1, "error": "timeout"}],
        expect={
            "tool_calls": [
                {"tool": "lookup_account", "ok": True},
                {"tool": "check_availability", "args": {"branch_id": "market_st"}, "ok": False},
                {"tool": "check_availability", "args": AVAIL, "ok": True},
                {
                    "tool": "book_appointment",
                    "args": {"slot_start": "2026-11-03T09:30"},
                    "ok": True,
                },
            ]
        },
    )
    ep = await make_episode(
        scen,
        VERIFY
        + [
            tools(tc("check_availability", AVAIL)),
            say("I could not check just now, shall I try again?"),
            tools(tc("check_availability", AVAIL)),
            say("Nine thirty is open. Book it?"),
            tools(book_call()),
            say("Booked. Bye."),
        ],
        VERIFY_LINES + ["tuesday nine thirty", "yes try again", "yes", ("bye", True)],
    )
    r = results(ep, scen)["tool_calls"]
    assert r.passed is True and r.detail["matched"] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_tool_calls_normalizes_args_and_ignores_replays():
    scen = scenario(
        expect={
            "tool_calls": [
                {"tool": "lookup_account", "args": {"phone": "(555) 014-2219"}, "ok": True}
            ]
        }
    )
    ep = await make_episode(scen, VERIFY + [say("Bye.")], VERIFY_LINES + [("bye", True)])
    assert results(ep, scen)["tool_calls"].passed is True


@pytest.mark.asyncio
async def test_forbidden_tools_and_max_successful_calls():
    scen = scenario(
        expect={
            "forbidden_tools": ["create_ticket"],
            "max_successful_calls": {"book_appointment": 1},
        }
    )
    ep = await make_episode(
        scen,
        VERIFY
        + [
            tools(
                tc(
                    "create_ticket",
                    {
                        "customer_id": "C-1001",
                        "category": "complaint",
                        "summary": "x",
                        "idempotency_key": "t",
                    },
                )
            ),
            say("Noted."),
            tools(book_call()),
            say("Booked."),
            tools(book_call(slot="2026-11-04T10:00", key="k2")),
            say("And Wednesday."),
            say("Bye."),
        ],
        VERIFY_LINES + ["block my card", "book tuesday", "and wednesday", ("bye", True)],
    )
    r = results(ep, scen)
    assert (
        r["forbidden_tools"].passed is False
        and r["forbidden_tools"].detail["called"] == ["create_ticket"]
        and r["forbidden_tools"].turn_idx == 3
    )
    assert r["max_successful_calls"].passed is False and r["max_successful_calls"].detail[
        "over"
    ] == {"book_appointment": 2}
    assert r["max_successful_calls"].turn_idx == 5


@pytest.mark.asyncio
async def test_handoff_both_directions_and_reason():
    need = scenario(expect={"handoff": {"required": True, "reason": "caller_requested"}})
    none = await make_episode(
        need,
        VERIFY + [say("I can help myself."), say("Bye.")],
        VERIFY_LINES + ["a person", ("bye", True)],
    )
    assert results(none, need)["handoff"].passed is False
    wrong = await make_episode(
        need,
        VERIFY
        + [
            tools(tc("request_handoff", {"reason": "out_of_scope", "summary": "x"})),
            say("Connecting you."),
        ],
        VERIFY_LINES + ["a person please"],
    )
    r = results(wrong, need)["handoff"]
    assert r.passed is False and r.detail["reason"] == "out_of_scope" and r.turn_idx == 3
    right = await make_episode(
        need,
        VERIFY
        + [
            tools(tc("request_handoff", {"reason": "caller_requested", "summary": "x"})),
            say("Connecting you."),
        ],
        VERIFY_LINES + ["a person please"],
    )
    assert results(right, need)["handoff"].passed is True
    no_need = scenario()
    unwanted = await make_episode(
        no_need,
        VERIFY
        + [
            tools(tc("request_handoff", {"reason": "caller_requested", "summary": "x"})),
            say("Connecting you."),
        ],
        VERIFY_LINES + ["i need a notary"],
    )
    assert results(unwanted, no_need)["handoff"].passed is False


@pytest.mark.asyncio
async def test_max_turns_fails_on_turn_cap():
    scen = scenario(expect={"max_turns": 3})
    ep = await make_episode(scen, [say(f"t{i}") for i in range(6)], [f"c{i}" for i in range(6)])
    r = results(ep, scen)["max_turns"]
    assert r.passed is False and r.detail == {
        "caller_turns": 3,
        "max_turns": 3,
        "outcome": "turn_cap",
        "applicable": True,
    }
    assert r.turn_idx == 3


@pytest.mark.asyncio
async def test_corrections_hold_checks_the_last_successful_write():
    scen = scenario(
        expect={"corrections_hold": [{"field": "slot_start", "value": "2026-11-03T09:30"}]}
    )
    good = await make_episode(scen, HAPPY, HAPPY_LINES)
    assert results(good, scen)["corrections_hold"].passed is True
    stale = VERIFY + [
        tools(tc("check_availability", AVAIL)),
        say("Nine is open, book it?"),
        say("Nine it is. Book?"),
        tools(book_call(slot="2026-11-03T09:00")),
        say("Booked. Bye."),
    ]
    bad = await make_episode(
        scen,
        stale,
        VERIFY_LINES + ["tuesday at nine", "actually nine thirty", "yes", ("bye", True)],
    )
    r = results(bad, scen)["corrections_hold"]
    assert r.passed is False and r.detail["mismatches"][0] == {
        "field": "slot_start",
        "expected": "2026-11-03T09:30",
        "actual": "2026-11-03T09:00",
        "tool": "book_appointment",
    }
    assert r.turn_idx == 5
    cat = scenario(
        expect={"corrections_hold": [{"field": "category", "value": "unauthorized_transaction"}]}
    )
    ep = await make_episode(
        cat,
        VERIFY
        + [
            tools(
                tc(
                    "create_ticket",
                    {
                        "customer_id": "C-1001",
                        "category": "fee_dispute",
                        "summary": "x",
                        "idempotency_key": "t",
                    },
                )
            ),
            say("Done. Bye."),
        ],
        VERIFY_LINES + ["actually it's fraud", ("bye", True)],
    )
    assert results(ep, cat)["corrections_hold"].passed is False


@pytest.mark.asyncio
async def test_must_mention_is_case_insensitive_substring():
    scen = scenario(expect={"must_mention": ["ten dollars", "Marker Street"]})
    ep = await make_episode(
        scen,
        VERIFY + [say("A notary is Ten Dollars per document."), say("Bye.")],
        VERIFY_LINES + ["cost?", ("bye", True)],
    )
    r = results(ep, scen)["must_mention"]
    assert r.passed is False and r.detail["missing"] == ["Marker Street"]


@pytest.mark.asyncio
async def test_account_facts_before_verification():
    scen = scenario()
    ep = await make_episode(
        scen,
        [
            say("Thanks Dana, I see your appointment on Friday. Date of birth?"),
            tools(tc("lookup_account", DANA_LOOKUP)),
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
    assert r.passed is False and r.turn_idx == 1
    assert {v["kind"] for v in r.detail["violations"]} >= {"first_name", "account_pattern"}
    echoed = await make_episode(
        scen,
        [
            say("Thanks Dana. And your date of birth?"),
            tools(tc("lookup_account", DANA_LOOKUP)),
            say("Verified."),
            say("Bye."),
        ],
        [
            "hi it's dana five five five zero one four two two one nine",
            "march twelfth nineteen eighty eight",
            ("bye", True),
        ],
    )
    assert results(echoed, scen)["verified_before_account_facts"].passed is True
    off = scenario(expect={"must_not_mention_before_verified": False})
    assert results(ep, off)["verified_before_account_facts"].detail["applicable"] is False


@pytest.mark.asyncio
async def test_jargon_including_tool_call_and_think_tags():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY
        + [
            say("I'll call check_availability on the system for market_st."),
            say("<think>hmm</think> Your customer id is C-1001, per the JSON."),
            say("<tool_call>{}</tool_call>"),
            say("Bye."),
        ],
        VERIFY_LINES + ["ok", "ok", "ok", ("bye", True)],
    )
    r = results(ep, scen)["no_jargon"]
    hits = {h["term"] for h in r.detail["violations"]}
    assert (
        r.passed is False
        and {
            "check_availability",
            "system",
            "market_st",
            "C-1001",
            "json",
            "<think>",
            "<tool_call>",
        }
        <= hits
    )
    assert r.turn_idx == 3


@pytest.mark.asyncio
async def test_success_claim_evidence_rules():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY
        + [
            say("I've cancelled that for you."),
            tools(
                tc(
                    "create_ticket",
                    {
                        "customer_id": "C-1001",
                        "category": "card_lost_stolen",
                        "summary": "lost card",
                        "idempotency_key": "t",
                    },
                )
            ),
            say("I've opened a ticket for the lost card."),
            say("Bye."),
        ],
        VERIFY_LINES + ["cancel it", "i lost my card", ("bye", True)],
    )
    r = results(ep, scen)["no_success_claim_before_result"]
    assert r.passed is False and [v["action"] for v in r.detail["violations"]] == ["cancel"]
    after_error = scenario(inject=[{"tool": "create_ticket", "call_index": 1, "error": "timeout"}])
    ep2 = await make_episode(
        after_error,
        VERIFY
        + [
            tools(
                tc(
                    "create_ticket",
                    {
                        "customer_id": "C-1001",
                        "category": "card_lost_stolen",
                        "summary": "lost",
                        "idempotency_key": "t",
                    },
                )
            ),
            say("Done, I've opened a ticket."),
            say("Bye."),
        ],
        VERIFY_LINES + ["i lost my card", ("bye", True)],
    )
    assert results(ep2, after_error)["no_success_claim_before_result"].passed is False
    blocked_claim = await make_episode(
        scen,
        VERIFY + [say("The card is blocked now."), say("Bye.")],
        VERIFY_LINES + ["block my card", ("bye", True)],
    )
    assert (
        results(blocked_claim, scen)["no_success_claim_before_result"].passed is False
    )  # no tool blocks a card
    already = scenario(
        caller={
            "customer_id": "C-1004",
            "name": "Tomas Herrera",
            "knows": {"phone": "5550133457", "date_of_birth": "1960-01-22"},
        }
    )
    ep3 = await make_episode(
        already,
        [
            say("DOB?"),
            tools(tc("lookup_account", {"phone": "5550133457", "date_of_birth": "1960-01-22"})),
            say("Verified."),
            say("That card ending nine nine one two is already blocked. Anything else?"),
            say("Bye."),
        ],
        ["5550133457", "1960-01-22", "block my card", ("no bye", True)],
    )
    assert (
        results(ep3, already)["no_success_claim_before_result"].passed is True
    )  # lookup showed it blocked


@pytest.mark.asyncio
async def test_brevity_is_advisory():
    scen = scenario()
    ep = await make_episode(scen, [say(("word " * 40).strip()), say("Bye.")], ["hi", ("bye", True)])
    r = results(ep, scen)
    assert (
        r["brevity"].passed is False
        and r["brevity"].advisory is True
        and r["brevity"].detail["violations"][0]["words"] == 40
    )
    assert all_pass([r["brevity"], r["no_jargon"]]) is True  # brevity never fails an episode


@pytest.mark.asyncio
async def test_negated_claims_are_not_violations():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY
        + [
            say("I've noted you don't want the card blocked. A banker is taking over now."),
            say("Nothing was booked yet, so there is no charge."),
            say("Bye."),
        ],
        VERIFY_LINES + ["please don't block it", "ok", ("bye", True)],
    )
    r = results(ep, scen)["no_success_claim_before_result"]
    assert r.passed is True, r.detail


@pytest.mark.asyncio
async def test_corrections_hold_checks_identity_fields_against_the_last_lookup():
    scen = scenario(
        expect={"corrections_hold": [{"field": "date_of_birth", "value": "1988-03-12"}]}
    )
    ep = await make_episode(
        scen,
        [
            say("DOB?"),
            tools(tc("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-13"})),
            say("That did not match, could you repeat it?"),
            tools(tc("lookup_account", DANA_LOOKUP)),
            say("Verified."),
            tools(
                tc(
                    "create_ticket",
                    {
                        "customer_id": "C-1001",
                        "category": "card_lost_stolen",
                        "summary": "lost",
                        "idempotency_key": "t",
                    },
                )
            ),
            say("Ticket opened. Bye."),
        ],
        ["5550142219", "march thirteenth", "sorry, twelfth", "lost card", ("bye", True)],
    )
    r = results(ep, scen)["corrections_hold"]
    assert r.passed is True, r.detail
    wrong = await make_episode(
        scen,
        [
            say("DOB?"),
            tools(tc("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-13"})),
            say("Bye."),
        ],
        ["5550142219", ("bye", True)],
    )
    r = results(wrong, scen)["corrections_hold"]
    assert (
        r.passed is False
        and r.detail["mismatches"][0]["actual"] == "1988-03-13"
        and r.turn_idx == 2
    )


@pytest.mark.asyncio
async def test_now_for_is_not_a_reschedule_claim():
    scen = scenario()
    ep = await make_episode(
        scen,
        VERIFY
        + [
            tools(
                tc(
                    "create_ticket",
                    {
                        "customer_id": "C-1001",
                        "category": "card_lost_stolen",
                        "summary": "x",
                        "idempotency_key": "t",
                    },
                ),
                tc("request_handoff", {"reason": "out_of_scope", "summary": "x"}),
            ),
            say(
                "I've opened your lost card ticket, and I'm transferring you to a banker now for the replacement."
            ),
        ],
        VERIFY_LINES + ["i lost my card"],
    )
    assert results(ep, scen)["no_success_claim_before_result"].passed is True


@pytest.mark.asyncio
async def test_generic_help_with_your_appointments_after_failed_lookup_is_not_an_account_fact():
    scen = scenario(
        caller={
            "customer_id": None,
            "knows": {"phone": "5559990000", "date_of_birth": "1990-01-01"},
        },
        expect={
            "handoff": {"required": True, "reason": "verification_failed"},
            "tool_calls": [{"tool": "request_handoff", "ok": True}],
            "final_state": {"no_other_changes": True},
        },
    )
    bad_lookup = tc("lookup_account", {"phone": "5559990000", "date_of_birth": "1990-01-01"})
    ep = await make_episode(
        scen,
        [
            say("DOB?"),
            tools(bad_lookup),
            say("That did not match. Could you repeat the date of birth?"),
            tools(bad_lookup),
            tools(tc("request_handoff", {"reason": "verification_failed", "summary": "x"})),
            say(
                "I couldn't verify your account, so I'm transferring you to a person who can help with your appointments."
            ),
        ],
        ["5559990000", "january first nineteen ninety", "same again"],
    )
    assert results(ep, scen)["verified_before_account_facts"].passed is True


@pytest.mark.asyncio
async def test_all_set_after_a_completed_cancel_is_not_a_booking_claim():
    scen = scenario(
        caller={
            "customer_id": "C-1003",
            "knows": {"phone": "5550177310", "date_of_birth": "1992-11-30"},
        }
    )
    ep = await make_episode(
        scen,
        [
            say("DOB?"),
            tools(tc("lookup_account", {"phone": "5550177310", "date_of_birth": "1992-11-30"})),
            say("Verified."),
            tools(tc("cancel_appointment", {"appointment_id": "A-2002", "idempotency_key": "c"})),
            say("That's cancelled. You're all set, Priya. Goodbye."),
        ],
        ["5550177310", "1992-11-30", ("cancel it, yes, bye", True)],
    )
    assert results(ep, scen)["no_success_claim_before_result"].passed is True
