"""Episode loop: scripted agent + scripted caller against a real Sandbox."""

import json
import uuid
from datetime import date

import pytest

from eval.sandbox import Sandbox
from eval.simulator.episode import GREETING, Episode, run_episode
from eval.simulator.scenario import Scenario

SCEN = Scenario.model_validate(
    {
        "id": "ordinary-01",
        "family": "ordinary",
        "today": "2026-11-02",
        "customer_id": "C-1042",
        "known_facts": {"phone": "5550142219", "date_of_birth": "1988-03-12"},
        "persona": {"name": "Dana"},
        "goal": "book a notary at market street wednesday",
        "expected": {"max_turns": 8},
    }
)


def tc(name, args, id_=None):
    return {
        "id": id_ or f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def say(text):
    return {"role": "assistant", "content": text}


def tools(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class ScriptedAgent:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    async def respond(self, messages):
        self.seen.append([dict(m) for m in messages])
        return self.script.pop(0)


class ScriptedCaller:
    """Yields (text, hangup) pairs; records the transcript it was shown."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.seen = []

    async def next_utterance(self, transcript, turn_idx):
        self.seen.append(list(transcript))
        if not self.lines:
            return ("", True)
        return self.lines.pop(0)


@pytest.mark.asyncio
async def test_ordinary_booking_flow_records_everything():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    slot = sb.call(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "2026-11-04"},
    )["slots"][0]
    sb.reset()
    lookup = tc("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"}, "c1")
    avail = tc(
        "check_availability",
        {"branch_id": "market_st", "service_type": "notary", "date": "2026-11-04"},
        "c2",
    )
    book = tc(
        "book_appointment",
        {
            "customer_id": "C-1042",
            "branch_id": "market_st",
            "service_type": "notary",
            "slot_start": f"2026-11-04T{slot}",
            "idempotency_key": "k1",
        },
        "c3",
    )
    agent = ScriptedAgent(
        [
            say("Thanks. And your date of birth?"),
            tools(lookup),
            say("Thanks Dana, you're verified. What can I help with today?"),
            tools(avail),
            say(f"Wednesday I have {slot}. Shall I book the notary at Market Street?"),
            tools(book),
            say("Booked. Anything else?"),
            say("Thanks Dana, goodbye."),
        ]
    )
    caller = ScriptedCaller(
        [
            ("five five five zero one four two two one nine", False),
            ("march twelfth nineteen eighty eight", False),
            ("i need a notary at market street wednesday", False),
            ("yes", False),
            ("no that's it", True),
        ]
    )
    ep = await run_episode(SCEN, agent, caller, sb, model_id="scripted", run_id="r1")

    assert isinstance(ep, Episode)
    assert ep.outcome == "completed"
    assert ep.messages[0]["role"] == "system"
    assert ep.messages[1] == {"role": "assistant", "content": GREETING}
    # the model's tool_calls objects are stored verbatim
    assert ep.messages[5]["tool_calls"] == [lookup]
    tool_msg = ep.messages[6]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["verified"] is True
    assert [e["tool"] for e in ep.tool_log] == [
        "lookup_account",
        "check_availability",
        "book_appointment",
    ]
    assert any(
        a["customer_id"] == "C-1042" and a["status"] == "booked"
        for a in ep.final_state["appointments"]
    )
    assert ep.turns[-1]["caller_text"] == "no that's it"
    assert ep.turns[-1]["assistant_text"] == "Thanks Dana, goodbye."
    assert ep.n_agent_turns == 6 and len(ep.turns) == 5
    assert ep.scenario_id == "ordinary-01" and ep.family == "ordinary"
    assert ep.model_id == "scripted" and ep.run_id == "r1"
    # what the agent saw on its last call is the full history
    assert agent.seen[-1][-1]["content"] == "no that's it"
    # the caller was shown the greeting first
    assert caller.seen[0] == [("agent", GREETING)]


@pytest.mark.asyncio
async def test_caller_text_is_stt_shaped_at_noise_level():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    scen = SCEN.model_copy(update={"noise_level": 0})
    agent = ScriptedAgent([say("ok"), say("bye")])
    caller = ScriptedCaller([("My number is 555-014-2219, thanks!", False), ("bye", True)])
    ep = await run_episode(scen, agent, caller, sb, model_id="s", run_id="r")
    assert ep.turns[0]["caller_text_raw"] == "My number is 555-014-2219, thanks!"
    assert (
        ep.turns[0]["caller_text"]
        == "my number is five five five zero one four two two one nine thanks"
    )
    assert ep.messages[2] == {"role": "user", "content": ep.turns[0]["caller_text"]}


@pytest.mark.asyncio
async def test_handoff_ends_after_one_closing_sentence():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    agent = ScriptedAgent(
        [
            tools(
                tc(
                    "request_handoff",
                    {"reason": "caller_requested", "summary": "wants a person"},
                    "h1",
                )
            ),
            say("I'm connecting you to a colleague now, please hold."),
            say("SHOULD NOT BE REACHED"),
        ]
    )
    caller = ScriptedCaller(
        [("let me talk to a person", False), ("hello?", False), ("hello??", False)]
    )
    ep = await run_episode(SCEN, agent, caller, sb, model_id="s", run_id="r")
    assert ep.outcome == "handoff"
    assert ep.final_state["handoff"]["reason"] == "caller_requested"
    assert len(ep.turns) == 1 and len(agent.script) == 1


@pytest.mark.asyncio
async def test_turn_cap_ends_episode():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    agent = ScriptedAgent([say(f"turn {i}") for i in range(40)])
    caller = ScriptedCaller([(f"caller {i}", False) for i in range(40)])
    ep = await run_episode(SCEN, agent, caller, sb, model_id="s", run_id="r", max_turns=20)
    assert ep.outcome == "turn_cap" and len(ep.turns) == 20


@pytest.mark.asyncio
async def test_hangup_with_text_gives_agent_a_closing_turn():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    agent = ScriptedAgent([say("Goodbye.")])
    caller = ScriptedCaller([("thanks bye", True)])
    ep = await run_episode(SCEN, agent, caller, sb, model_id="s", run_id="r")
    assert ep.outcome == "completed"
    assert ep.turns[0]["assistant_text"] == "Goodbye."


@pytest.mark.asyncio
async def test_bare_hangup_ends_without_agent_turn():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    agent = ScriptedAgent([say("never")])
    caller = ScriptedCaller([("", True)])
    ep = await run_episode(SCEN, agent, caller, sb, model_id="s", run_id="r")
    assert ep.outcome == "completed" and ep.turns == [] and len(agent.script) == 1


@pytest.mark.asyncio
async def test_invalid_json_arguments_become_tool_error_and_call_continues():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    bad = {
        "id": "b1",
        "type": "function",
        "function": {"name": "lookup_account", "arguments": "{phone: oops"},
    }
    agent = ScriptedAgent([tools(bad), say("Sorry, could you repeat the number?"), say("bye")])
    caller = ScriptedCaller([("my number is five five five", False), ("bye", True)])
    ep = await run_episode(SCEN, agent, caller, sb, model_id="s", run_id="r")
    result = json.loads(ep.messages[4]["content"])
    assert ep.messages[4]["tool_call_id"] == "b1"
    assert result["error"] == "invalid_argument" and "JSON" in result["message"]
    assert ep.tool_log[0]["ok"] is False
    assert ep.outcome == "completed"


@pytest.mark.asyncio
async def test_tool_loop_is_capped():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    avail = {"branch_id": "market_st", "service_type": "notary", "date": "2026-11-04"}
    agent = ScriptedAgent([tools(tc("check_availability", avail)) for _ in range(10)])
    caller = ScriptedCaller([("hi", False)])
    ep = await run_episode(SCEN, agent, caller, sb, model_id="s", run_id="r", max_tool_rounds=3)
    assert ep.outcome == "abandoned"
    assert ep.error and "tool rounds" in ep.error
    assert len(ep.tool_log) == 4  # initial + 3 more rounds, then stop


@pytest.mark.asyncio
async def test_injected_failures_from_scenario_are_armed():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    scen = Scenario.model_validate(
        {**SCEN.to_dict(), "injected_failures": [{"tool": "lookup_account", "nth_call": 1}]}
    )
    lookup = {"phone": "5550142219", "date_of_birth": "1988-03-12"}
    agent = ScriptedAgent(
        [
            tools(tc("lookup_account", lookup)),
            say("That didn't go through, one moment."),
            tools(tc("lookup_account", lookup)),
            say("Verified."),
            say("bye"),
        ]
    )
    caller = ScriptedCaller([("here are my details", False), ("ok", False), ("bye", True)])
    ep = await run_episode(scen, agent, caller, sb, model_id="s", run_id="r")
    assert ep.tool_log[0]["result"]["error"] == "timeout" and ep.tool_log[0]["injected"] is True
    assert ep.tool_log[1]["ok"] is True


@pytest.mark.asyncio
async def test_agent_exception_yields_error_outcome_not_crash():
    sb = Sandbox(seed=42, today=date(2026, 11, 2))

    class Boom:
        async def respond(self, messages):
            raise RuntimeError("endpoint down")

    caller = ScriptedCaller([("hi", False)])
    ep = await run_episode(SCEN, Boom(), caller, sb, model_id="s", run_id="r")
    assert ep.outcome == "error" and "endpoint down" in ep.error


@pytest.mark.asyncio
async def test_episode_json_round_trip(tmp_path):
    sb = Sandbox(seed=42, today=date(2026, 11, 2))
    agent = ScriptedAgent([say("bye")])
    caller = ScriptedCaller([("bye", True)])
    ep = await run_episode(SCEN, agent, caller, sb, model_id="s", run_id="r")
    p = ep.save(tmp_path)
    data = json.loads(p.read_text())
    assert p.name == "ordinary-01.json"
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
    }
    assert data["scenario"]["id"] == "ordinary-01"
    assert Episode.from_dict(data).outcome == "completed"
