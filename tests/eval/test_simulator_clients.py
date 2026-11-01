"""OpenAI-compatible agent and caller clients, driven through a fake transport."""

from types import SimpleNamespace

import pytest

from eval.simulator.clients import LLMCaller, OpenAIAgent
from eval.simulator.scenario import Scenario

SCEN = Scenario.model_validate(
    {
        "id": "corr-01",
        "family": "correction",
        "today": "2026-11-02",
        "customer_id": "C-1042",
        "known_facts": {"phone": "5550142219", "date_of_birth": "1988-03-12", "name": "Dana"},
        "persona": {"name": "Dana", "age": 38, "style": "terse"},
        "goal": "Book a notary at Marlow Street on Friday November 6 at two, then change it to two thirty.",
        "corrections": [
            {
                "when": "after the agent reads back two o'clock",
                "field": "time",
                "from": "14:00",
                "to": "14:30",
                "say": "actually make it two thirty",
            }
        ],
        "detours": [
            {
                "when": "once verified",
                "say": "what are your saturday hours",
                "resolution": "then continue",
            }
        ],
        "noise_level": 1,
    }
)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls

    def model_dump(self, exclude_none=False):
        d = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": self.tool_calls,
            "refusal": None,
        }
        return {k: v for k, v in d.items() if not (exclude_none and v is None)}


class FakeCompletions:
    def __init__(self, replies, fail_first=0):
        self.replies, self.calls, self.fail_first = list(replies), [], fail_first

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first:
            self.fail_first -= 1
            from openai import APIConnectionError

            raise APIConnectionError(request=SimpleNamespace())
        msg = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        )


def fake_client(replies, fail_first=0):
    comp = FakeCompletions(replies, fail_first)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


@pytest.mark.asyncio
async def test_agent_sends_tools_and_returns_verbatim_message():
    tc = [
        {"id": "x1", "type": "function", "function": {"name": "lookup_account", "arguments": "{}"}}
    ]
    client, comp = fake_client([FakeMessage(content="", tool_calls=tc)])
    agent = OpenAIAgent(client=client, model="m1")
    msg = await agent.respond(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    )
    assert msg == {"role": "assistant", "content": "", "tool_calls": tc}
    sent = comp.calls[0]
    assert sent["model"] == "m1" and len(sent["tools"]) == 8
    assert {t["function"]["name"] for t in sent["tools"]} >= {"lookup_account", "request_handoff"}
    assert "temperature" not in sent  # the hosted endpoint rejects it
    assert sent["max_tokens"] == 300
    assert agent.usage_totals == {"prompt_tokens": 100, "completion_tokens": 10, "requests": 1}


@pytest.mark.asyncio
async def test_agent_passes_temperature_only_when_set_and_retries_connection_errors():
    client, comp = fake_client([FakeMessage(content="hello")], fail_first=2)
    agent = OpenAIAgent(client=client, model="m1", temperature=0.2, retries=3, backoff_s=0)
    msg = await agent.respond([{"role": "user", "content": "hi"}])
    assert msg == {"role": "assistant", "content": "hello"}
    assert comp.calls[-1]["temperature"] == 0.2 and len(comp.calls) == 3


@pytest.mark.asyncio
async def test_agent_gives_up_after_retries():
    client, _ = fake_client([FakeMessage(content="late")], fail_first=5)
    agent = OpenAIAgent(client=client, model="m1", retries=2, backoff_s=0)
    from openai import APIConnectionError

    with pytest.raises(APIConnectionError):
        await agent.respond([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_caller_prompt_carries_scenario_and_flips_roles():
    client, comp = fake_client([FakeMessage(content="Hi, it's Dana, 555-014-2219. [hangup]")])
    caller = LLMCaller(SCEN, client=client, model="c1")
    text, hangup = await caller.next_utterance(
        [("agent", "Hello, phone number?"), ("caller", "hi"), ("agent", "number please")], 2
    )
    assert (text, hangup) == ("Hi, it's Dana, 555-014-2219.", True)
    sent = comp.calls[0]["messages"]
    system = sent[0]["content"]
    for needle in (
        "Dana",
        "5550142219",
        "1988-03-12",
        "two thirty",
        "saturday hours",
        "terse",
        "[hangup]",
        "Marlow Street",
    ):
        assert needle in system, needle
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "user"]
    assert sent[1]["content"] == "Hello, phone number?"
    assert "temperature" not in comp.calls[0]
    assert caller.usage_totals["requests"] == 1


@pytest.mark.asyncio
async def test_caller_strips_quotes_and_narration():
    client, _ = fake_client([FakeMessage(content='Caller: "yes please"')])
    caller = LLMCaller(SCEN, client=client, model="c1")
    text, hangup = await caller.next_utterance([("agent", "Shall I book it?")], 5)
    assert (text, hangup) == ("yes please", False)


@pytest.mark.asyncio
async def test_caller_empty_reply_becomes_hangup():
    client, _ = fake_client([FakeMessage(content="")])
    caller = LLMCaller(SCEN, client=client, model="c1")
    assert await caller.next_utterance([("agent", "Goodbye.")], 9) == ("", True)
