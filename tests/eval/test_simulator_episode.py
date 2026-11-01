"""Episode loop on schema.json scenarios: scripted agent + scripted caller against the seed sandbox."""

import json

import pytest
from helpers import (
    DANA_LOOKUP,
    VERIFY_LINES,
    ScriptedAgent,
    ScriptedCaller,
    make_episode,
    say,
    scenario,
    tc,
    tools,
)

from eval.sandbox import Sandbox
from eval.simulator.episode import GREETING, Episode, run_episode

AVAIL = {"branch_id": "market_st", "service_type": "mortgage_consultation", "date": "2026-11-03"}
BOOK = {
    "customer_id": "C-1001",
    "branch_id": "market_st",
    "service_type": "mortgage_consultation",
    "slot_start": "2026-11-03T09:30",
    "idempotency_key": "k1",
}


@pytest.mark.asyncio
async def test_ordinary_booking_flow_records_everything():
    scen = scenario()
    lookup = tc("lookup_account", DANA_LOOKUP, "c1")
    agent = ScriptedAgent(
        [
            say("Thanks. And your date of birth?"),
            tools(lookup),
            say("Thanks Dana, you're verified. What can I help with today?"),
            tools(tc("check_availability", AVAIL, "c2")),
            say(
                "Tuesday at nine thirty is open. Shall I book the mortgage consultation at Market Street?"
            ),
            tools(tc("book_appointment", BOOK, "c3")),
            say("Booked. Anything else?"),
            say("Thanks Dana, goodbye."),
        ]
    )
    caller = ScriptedCaller(
        VERIFY_LINES
        + ["mortgage consultation tuesday morning market street", "yes", ("no that's it", True)]
    )
    ep = await run_episode(
        scen, agent, caller, Sandbox(), model_id="scripted", run_id="r1", repeat_seed=1
    )
    assert isinstance(ep, Episode) and ep.outcome == "completed"
    assert (
        ep.messages[0]["role"] == "system"
        and "Monday, November second" in ep.messages[0]["content"]
    )
    assert ep.messages[1] == {"role": "assistant", "content": GREETING}
    assert ep.messages[5]["tool_calls"] == [lookup]
    assert ep.messages[6]["role"] == "tool" and ep.messages[6]["tool_call_id"] == "c1"
    assert json.loads(ep.messages[6]["content"])["verified"] is True
    assert [e["tool"] for e in ep.tool_log] == [
        "lookup_account",
        "check_availability",
        "book_appointment",
    ]
    assert ep.final_state["diff"]["appointments_new"] == [
        {
            "customer_id": "C-1001",
            "branch_id": "market_st",
            "service_type": "mortgage_consultation",
            "slot_start": "2026-11-03T09:30",
        }
    ]
    assert (
        ep.turns[-1]["caller_text"] == "no that's it"
        and ep.turns[-1]["assistant_text"] == "Thanks Dana, goodbye."
    )
    assert ep.n_agent_turns == 6 and len(ep.turns) == 5 and ep.n_model_requests == 8
    assert ep.scenario_id == "ordinary_01" and ep.family == "ordinary" and ep.repeat_seed == 1
    assert ep.model_id == "scripted" and ep.run_id == "r1"
    assert ep.turns[0]["finish_reasons"] == ["stop"] and ep.turns[0]["truncated"] is False
    assert caller.seen[0] == [("agent", GREETING)]
    assert ep.prompt_source.endswith("system_prompt.md") and ep.system_prompt_chars > 500


@pytest.mark.asyncio
async def test_noise_levels_map_to_stt_shaping():
    clean = await make_episode(
        scenario(),
        [say("ok"), say("bye")],
        [("My number is 555-014-2219, thanks!", False), ("bye", True)],
    )
    assert (
        clean.turns[0]["caller_text"]
        == "my number is five five five zero one four two two one nine thanks"
    )
    assert clean.turns[0]["caller_text_raw"] == "My number is 555-014-2219, thanks!"
    noisy = await make_episode(
        scenario(caller={"noise": "high"}),
        [say("ok"), say("bye")],
        [
            ("i would like to book for two people on the fourth of november please", False),
            ("bye", True),
        ],
    )
    assert (
        noisy.turns[0]["caller_text"]
        != "i would like to book for two people on the fourth of november please"
    )


@pytest.mark.asyncio
async def test_inject_from_scenario_is_armed_and_counted():
    scen = scenario(
        inject=[
            {"tool": "check_availability", "call_index": 1, "error": "timeout"},
            {"tool": "lookup_account", "call_index": 1, "error": "not_found"},
        ]
    )
    ep = await make_episode(
        scen,
        [
            tools(tc("lookup_account", DANA_LOOKUP)),
            say("That did not go through."),
            tools(tc("lookup_account", DANA_LOOKUP)),
            say("Verified."),
            tools(tc("check_availability", AVAIL)),
            say("Could not check, try again?"),
            tools(tc("check_availability", AVAIL)),
            say("Nine thirty is open."),
            say("Bye."),
        ],
        ["five five five...", "again", "tuesday", "yes", ("bye", True)],
    )
    log = ep.tool_log
    assert log[0]["result"]["error"] == "not_found" and log[0]["injected"] is True
    assert log[1]["ok"] is True
    assert (
        log[2]["result"] == {"error": "timeout", "message": "the request timed out"}
        and log[2]["injected"]
    )
    assert log[3]["ok"] is True


@pytest.mark.asyncio
async def test_max_turns_comes_from_expect():
    scen = scenario(expect={"max_turns": 3})
    ep = await make_episode(scen, [say(f"t{i}") for i in range(10)], [f"c{i}" for i in range(10)])
    assert ep.outcome == "turn_cap" and len(ep.turns) == 3


@pytest.mark.asyncio
async def test_handoff_ends_after_one_closing_sentence():
    agent = ScriptedAgent(
        [
            tools(
                tc("request_handoff", {"reason": "caller_requested", "summary": "wants a person"})
            ),
            say("I'm connecting you to a colleague now, please hold."),
            say("SHOULD NOT BE REACHED"),
        ]
    )
    ep = await make_episode(
        scenario(), [], [("a person please", False), ("hello?", False)], agent=agent
    )
    assert ep.outcome == "handoff" and ep.final_state["handoff"]["reason"] == "caller_requested"
    assert len(ep.turns) == 1 and len(agent.script) == 1


@pytest.mark.asyncio
async def test_hangup_semantics():
    with_text = await make_episode(scenario(), [say("Goodbye.")], [("thanks bye", True)])
    assert with_text.outcome == "completed" and with_text.turns[0]["assistant_text"] == "Goodbye."
    bare = await make_episode(scenario(), [say("never")], [("", True)])
    assert bare.outcome == "completed" and bare.turns == []


@pytest.mark.asyncio
async def test_empty_caller_text_without_hangup_keeps_the_call_alive():
    ep = await make_episode(
        scenario(), [say("Are you there?"), say("Bye.")], [("", False), ("bye", True)]
    )
    assert ep.turns[0]["caller_text"] == "hello" and ep.turns[0]["caller_text_raw"] == ""
    assert ep.outcome == "completed" and len(ep.turns) == 2


@pytest.mark.asyncio
async def test_invalid_json_arguments_become_tool_error_and_call_continues():
    bad = {
        "id": "b1",
        "type": "function",
        "function": {"name": "lookup_account", "arguments": "{phone: oops"},
    }
    ep = await make_episode(
        scenario(),
        [tools(bad), say("Sorry, could you repeat the number?"), say("bye")],
        ["my number is five five five", ("bye", True)],
    )
    result = json.loads(ep.messages[4]["content"])
    assert ep.messages[4]["tool_call_id"] == "b1"
    assert result["error"] == "invalid_argument" and "JSON" in result["message"]
    assert ep.tool_log[0]["ok"] is False and ep.outcome == "completed"


@pytest.mark.asyncio
async def test_tool_loop_is_capped():
    ep = await make_episode(
        scenario(),
        [tools(tc("check_availability", AVAIL)) for _ in range(10)],
        ["hi"],
        max_tool_rounds=3,
    )
    assert ep.outcome == "abandoned" and "tool rounds" in ep.error and len(ep.tool_log) == 4


@pytest.mark.asyncio
async def test_truncated_model_output_is_flagged():
    agent = ScriptedAgent(
        [say("I'm sorry, Dana. I'll"), say("Bye.")], finish_reasons=["length", "stop"]
    )
    ep = await make_episode(scenario(), [], ["i lost my card", ("bye", True)], agent=agent)
    assert ep.turns[0]["finish_reasons"] == ["length"] and ep.turns[0]["truncated"] is True
    assert ep.n_truncated == 1


@pytest.mark.asyncio
async def test_agent_exception_yields_error_outcome_not_crash():
    class Boom:
        async def respond(self, messages):
            raise RuntimeError("endpoint down")

    ep = await run_episode(
        scenario(), Boom(), ScriptedCaller(["hi"]), Sandbox(), model_id="s", run_id="r"
    )
    assert ep.outcome == "error" and "endpoint down" in ep.error


@pytest.mark.asyncio
async def test_episode_json_round_trip(tmp_path):
    ep = await make_episode(scenario(), [say("bye")], [("bye", True)])
    p = ep.save(tmp_path)
    data = json.loads(p.read_text())
    assert p.name == "ordinary_01.json"
    assert set(data) >= {
        "episode_id",
        "scenario_id",
        "family",
        "model_id",
        "run_id",
        "outcome",
        "messages",
        "turns",
        "tool_log",
        "final_state",
        "scenario",
        "usage",
        "started_at",
        "ended_at",
        "n_agent_turns",
        "repeat_seed",
        "prompt_source",
        "system_prompt_chars",
        "first_prompt_tokens",
    }
    assert data["scenario"]["id"] == "ordinary_01" and "source" not in data["scenario"]
    assert Episode.from_dict(data).outcome == "completed"


@pytest.mark.asyncio
async def test_episode_records_served_models():
    agent = ScriptedAgent([say("ok"), say("bye")])
    agent.served_models = {"r3-voice-base"}
    ep = await make_episode(scenario(), [], ["hi", ("bye", True)], agent=agent)
    assert ep.served_models == ["r3-voice-base"]
    assert "served_models" in ep.to_dict()
