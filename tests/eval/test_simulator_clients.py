"""OpenAI-compatible agent and caller clients through a fake transport."""

from types import SimpleNamespace

import pytest
from helpers import scenario

from eval.simulator.clients import LLMCaller, OpenAIAgent, clean_caller_text

SCEN = scenario(
    caller={
        "persona": "Decisive, changes her mind once",
        "withholds": ["date of birth"],
        "noise": "high",
        "stt_quirks": ["say 'oh' for every zero"],
    },
    goal="Book a notary at Marker Street Thursday at two, then change it to two thirty.",
    script=[
        {"trigger": "agent reads back the 2:00 booking", "do": "say 'actually make it two thirty'"}
    ],
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
    def __init__(self, replies, fail_first=0, reject_temperature=False):
        self.replies, self.calls, self.fail_first = list(replies), [], fail_first
        self.reject_temperature = reject_temperature

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first:
            self.fail_first -= 1
            from openai import APIConnectionError

            raise APIConnectionError(request=SimpleNamespace())
        if self.reject_temperature and "temperature" in kwargs:
            from openai import BadRequestError

            raise BadRequestError(
                "`temperature` is deprecated for this model.",
                response=SimpleNamespace(status_code=400, headers={}, request=None),
                body={"error": {"message": "`temperature` is deprecated for this model."}},
            )
        msg = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        )


def fake_client(replies, **kw):
    comp = FakeCompletions(replies, **kw)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


@pytest.mark.asyncio
async def test_agent_sends_tools_and_returns_verbatim_message():
    tcs = [
        {"id": "x1", "type": "function", "function": {"name": "lookup_account", "arguments": "{}"}}
    ]
    client, comp = fake_client([FakeMessage(content="", tool_calls=tcs)])
    agent = OpenAIAgent(client=client, model="m1")
    msg = await agent.respond(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    )
    assert msg == {"role": "assistant", "content": "", "tool_calls": tcs}
    sent = comp.calls[0]
    assert sent["model"] == "m1" and {t["function"]["name"] for t in sent["tools"]} >= {
        "lookup_account",
        "request_handoff",
    }
    assert "temperature" not in sent and sent["max_tokens"] == 700
    assert agent.last_finish_reason == "stop" and agent.last_usage == {
        "prompt_tokens": 100,
        "completion_tokens": 10,
    }
    assert agent.usage_totals == {"prompt_tokens": 100, "completion_tokens": 10, "requests": 1}


@pytest.mark.asyncio
async def test_agent_retries_connection_errors_and_sends_temperature_when_set():
    client, comp = fake_client([FakeMessage(content="hello")], fail_first=2)
    agent = OpenAIAgent(client=client, model="m1", temperature=0.2, retries=3, backoff_s=0)
    assert await agent.respond([{"role": "user", "content": "hi"}]) == {
        "role": "assistant",
        "content": "hello",
    }
    assert comp.calls[-1]["temperature"] == 0.2 and len(comp.calls) == 3


@pytest.mark.asyncio
async def test_temperature_rejected_by_endpoint_falls_back_once_and_is_remembered():
    client, comp = fake_client(
        [FakeMessage(content="a"), FakeMessage(content="b")], reject_temperature=True
    )
    agent = OpenAIAgent(client=client, model="m1", temperature=0.3)
    assert (await agent.respond([{"role": "user", "content": "hi"}]))["content"] == "a"
    assert (await agent.respond([{"role": "user", "content": "hi"}]))["content"] == "b"
    assert [("temperature" in c) for c in comp.calls] == [True, False, False]
    assert agent.temperature_applied is False


@pytest.mark.asyncio
async def test_caller_prompt_carries_scenario_seed_and_flips_roles():
    client, comp = fake_client([FakeMessage(content="Hi, it's Dana, 555-014-2219. [hangup]")])
    caller = LLMCaller(SCEN, client=client, model="c1", seed=2)
    text, hangup = await caller.next_utterance(
        [("agent", "Hello, phone number?"), ("caller", "hi"), ("agent", "number please")], 2
    )
    assert (text, hangup) == ("Hi, it's Dana, 555-014-2219.", True) and caller.last_raw.startswith(
        "Hi, it's Dana"
    )
    sent = comp.calls[0]["messages"]
    system = sent[0]["content"]
    for needle in (
        "Dana Whitfield",
        "five five five zero one four two two one nine",
        "1988-03-12",
        "Decisive",
        "date of birth",
        "say 'oh' for every zero",
        "two thirty",
        "actually make it two thirty",
        "[hangup]",
        "seed 2",
        "Monday, November second",
    ):
        assert needle in system, needle
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "user"] and sent[1][
        "content"
    ] == "Hello, phone number?"
    assert comp.calls[0]["temperature"] == 0.3 and caller.usage_totals["requests"] == 1


def test_clean_caller_text_rules():
    assert clean_caller_text('Caller: "yes please"') == ("yes please", False)
    assert clean_caller_text("") == ("", False)
    assert clean_caller_text("[hangup]") == ("", True)
    assert clean_caller_text("okay thanks bye [HANG UP]") == ("okay thanks bye", True)
    assert clean_caller_text("*sighs* no, a person please") == ("no, a person please", False)
