"""data.format.targets: pick up to three target turns per episode, filter by brevity, emit examples."""

import json

import pytest
from helpers import VERIFY, VERIFY_LINES, make_episode, say, scenario, tc, tools

from data.format.targets import (
    MAX_SENTENCES,
    MAX_WORDS,
    Target,
    brevity_ok,
    classify_targets,
    select_targets,
    to_example,
)

AVAIL = {"branch_id": "market_st", "service_type": "mortgage_consultation", "date": "2026-11-03"}
BOOK = {
    "customer_id": "C-1001",
    "branch_id": "market_st",
    "service_type": "mortgage_consultation",
    "slot_start": "2026-11-03T09:30",
    "idempotency_key": "k",
}
HAPPY = VERIFY + [
    tools(tc("check_availability", AVAIL)),
    say(
        "Tuesday at nine thirty is open. A mortgage consultation at Market Street then, shall I book it?"
    ),
    tools(tc("book_appointment", BOOK)),
    say("Booked. Anything else?"),
    say("Thanks Dana, goodbye."),
]
LINES = VERIFY_LINES + [
    "a mortgage consultation tuesday at nine thirty",
    "yes",
    ("no that's it", True),
]


def test_brevity_rule_counts_words_and_sentences():
    assert brevity_ok("Booked. Anything else?") == (True, 3, 2)
    ok, _words, sentences = brevity_ok("One. Two. Three sentences here.")
    assert (ok, sentences) == (False, 3)
    assert brevity_ok("word " * 36)[0] is False and brevity_ok("")[0] is True
    assert (MAX_WORDS, MAX_SENTENCES) == (35, 2)


@pytest.mark.asyncio
async def test_classify_targets_by_kind_and_turn():
    ep = await make_episode(scenario(), HAPPY, LINES)
    targets = classify_targets(ep)
    kinds = [(t.kind, t.turn) for t in targets]
    # greeting is never a target; tool-call turns, questions, and the final line are classified
    assert kinds[0] == ("clarification", 1)  # "Thanks. And your date of birth?"
    assert ("tool_call", 2) in kinds and ("tool_call", 3) in kinds and ("tool_call", 4) in kinds
    assert kinds[-1] == ("closing", 5)
    assert all(isinstance(t, Target) and t.message_idx > 1 for t in targets)
    lookup = next(t for t in targets if t.kind == "tool_call" and t.turn == 2)
    assert lookup.tool_names == ["lookup_account"] and lookup.text == ""


@pytest.mark.asyncio
async def test_select_targets_covers_tool_call_clarification_and_closing():
    ep = await make_episode(scenario(), HAPPY, LINES)
    chosen, dropped = select_targets(ep)
    assert [t.kind for t in chosen] == ["tool_call", "clarification", "closing"]
    assert chosen[0].tool_names == [
        "book_appointment"
    ]  # the write beats the lookup when both exist
    assert chosen[1].text.endswith("shall I book it?")
    assert chosen[2].text == "Thanks Dana, goodbye."
    assert dropped == []


@pytest.mark.asyncio
async def test_long_targets_are_dropped_not_truncated_and_the_rest_kept():
    long_close = say(
        "Your mortgage consultation is booked at Market Street on Tuesday, November third at nine thirty. "
        "A banker will meet you there. Please bring a photo id. Goodbye, Dana, and thank you for calling."
    )
    ep = await make_episode(scenario(), HAPPY[:-1] + [long_close], LINES)
    chosen, dropped = select_targets(ep)
    assert [t.kind for t in chosen] == ["tool_call", "clarification"]
    assert len(dropped) == 1 and dropped[0].kind == "closing" and dropped[0].reason == "brevity"
    assert dropped[0].sentences == 4


@pytest.mark.asyncio
async def test_to_example_has_the_loader_shape():
    ep = await make_episode(scenario(), HAPPY, LINES)
    chosen, _ = select_targets(ep)
    ex = to_example(ep, chosen[0])
    assert set(ex) == {"messages", "tools", "meta"}
    assert (
        ex["messages"][-1]["role"] == "assistant"
        and ex["messages"][-1]["tool_calls"][0]["function"]["name"] == "book_appointment"
    )
    assert isinstance(ex["messages"][-1]["tool_calls"][0]["function"]["arguments"], str)
    assert all(isinstance(m.get("content"), str) for m in ex["messages"])  # never null
    assert ex["messages"][0]["role"] == "system" and ex["messages"][1]["role"] == "assistant"
    assert [t["function"]["name"] for t in ex["tools"]] == [
        t["function"]["name"] for t in ex["tools"]
    ] and len(ex["tools"]) >= 7
    assert ex["meta"] == {
        "episode_id": ep.episode_id,
        "scenario_id": "ordinary_01",
        "family": "ordinary",
        "kind": "tool_call",
        "turn": 4,
        "message_idx": chosen[0].message_idx,
        "repeat_seed": 0,
    }
    json.dumps(ex)  # serialisable


TICKET = {
    "customer_id": "C-1001",
    "category": "card_lost_stolen",
    "summary": "Lost debit card ending 4219 at the gym; banker to block it.",
}
HANDOFF = {"reason": "out_of_scope", "summary": "Lost card ticket opened; card not yet blocked."}
LOST_CARD = VERIFY + [
    say(
        "Sorry to hear that. I'll open a lost card ticket for the card ending four two one nine, then transfer you to a banker. Is that right?"
    ),
    tools(tc("create_ticket", TICKET)),
    tools(tc("request_handoff", HANDOFF)),
    say("Your ticket is open and I'm transferring you to a banker now."),
]
LOST_CARD_LINES = VERIFY_LINES + ["i lost my debit card ending four two one nine", "yes"]
NOT_FOUND = {"phone": "5559999999", "date_of_birth": "1988-03-12"}
UNVERIFIED = [
    say("Thanks. And your date of birth?"),
    tools(tc("lookup_account", NOT_FOUND)),
    say("I couldn't find an account with that number. Could you repeat it, digit by digit?"),
    tools(tc("lookup_account", NOT_FOUND)),
    tools(
        tc("request_handoff", {"reason": "verification_failed", "summary": "Two lookups failed."})
    ),
    say("I couldn't verify your account, so I'm transferring you to a person."),
]
UNVERIFIED_LINES = [
    "five five five nine nine nine nine nine nine nine",
    "march twelfth nineteen eighty eight",
    "five five five nine nine nine nine nine nine nine",
]


@pytest.mark.asyncio
async def test_handoff_turn_takes_the_tool_call_slot_and_the_write_moves_up():
    """CONTRACT §1 handoff rules are scored by reason; the handoff decision must be a target even
    when a write (the ticket) precedes it. The read-back clarification is what gives way."""
    ep = await make_episode(scenario(family="needs_person"), LOST_CARD, LOST_CARD_LINES)
    assert ep.outcome == "handoff"
    chosen, dropped = select_targets(ep)
    assert [t.tool_names for t in chosen[:2]] == [["request_handoff"], ["create_ticket"]]
    assert [t.kind for t in chosen] == ["tool_call", "tool_call", "closing"]
    assert chosen[2].text.startswith("Your ticket is open")
    assert dropped == []


@pytest.mark.asyncio
async def test_handoff_without_a_write_keeps_the_clarification_slot():
    ep = await make_episode(scenario(family="needs_person"), UNVERIFIED, UNVERIFIED_LINES)
    assert ep.outcome == "handoff"
    chosen, _ = select_targets(ep)
    assert [t.kind for t in chosen] == ["tool_call", "clarification", "closing"]
    assert chosen[0].tool_names == ["request_handoff"]
    assert chosen[1].text.startswith("I couldn't find an account")
    assert chosen[2].text.startswith("I couldn't verify")


def _reject(*tool_names):
    """A `fits` callback that drops any target whose tool call is one of `tool_names`."""

    def fits(ex):
        calls = ex["messages"][-1].get("tool_calls") or []
        return not any(c["function"]["name"] in tool_names for c in calls)

    return fits


@pytest.mark.asyncio
async def test_when_the_write_overflows_the_fallback_is_the_latest_read_not_the_lookup():
    """v2 had 706 lookup_account targets out of 1,252 tool calls: the write turn overflowed 2,048
    in half the episodes and the fallback was the earliest call. The fallback must be the latest
    read (check_availability carries the branch, service and date arguments the scorers check)."""
    ep = await make_episode(scenario(), HAPPY, LINES)
    chosen, dropped = select_targets(ep, fits=_reject("book_appointment"))
    assert chosen[0].tool_names == ["check_availability"]
    assert [t.kind for t in chosen] == ["tool_call", "clarification", "closing"]
    assert [t.tool_names for t in dropped] == [["book_appointment"]]


@pytest.mark.asyncio
async def test_lookup_is_a_target_only_when_it_is_the_episodes_only_tool_call():
    ep = await make_episode(scenario(), HAPPY, LINES)
    chosen, _ = select_targets(ep, fits=_reject("book_appointment", "check_availability"))
    assert [t.kind for t in chosen] == ["clarification", "closing"]  # two targets, no lookup
    lookup_only = VERIFY + [say("I can only help with appointments and cards. Goodbye.")]
    ep2 = await make_episode(scenario(), lookup_only, VERIFY_LINES + [("ok bye", True)])
    chosen2, _ = select_targets(ep2)
    assert chosen2[0].tool_names == ["lookup_account"]


@pytest.mark.asyncio
async def test_extra_length_check_is_pluggable():
    ep = await make_episode(scenario(), HAPPY, LINES)
    chosen, dropped = select_targets(ep, fits=lambda ex: ex["meta"]["kind"] != "closing")
    assert [t.kind for t in chosen] == ["tool_call", "clarification"]
    assert dropped[0].reason == "length"
