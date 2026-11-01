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


@pytest.mark.asyncio
async def test_extra_length_check_is_pluggable():
    ep = await make_episode(scenario(), HAPPY, LINES)
    chosen, dropped = select_targets(ep, fits=lambda ex: ex["meta"]["kind"] != "closing")
    assert [t.kind for t in chosen] == ["tool_call", "clarification"]
    assert dropped[0].reason == "length"
