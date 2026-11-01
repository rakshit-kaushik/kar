"""OpenAI-compatible chat clients for the agent under test and the simulated caller.

Verified against openai 3.42.0 in .venv: `AsyncOpenAI(api_key, base_url, timeout, max_retries)`,
`client.chat.completions.create(model, messages, tools, max_tokens, temperature)`,
`resp.choices[0].message.model_dump(exclude_none=True)`, `resp.choices[0].finish_reason`,
`resp.usage.prompt_tokens`. The hosted Anthropic endpoint rejects `temperature` for
claude-sonnet-5-5 with a 400 that names it; the client then drops temperature for the rest of
the run and records `temperature_applied = False`.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from eval.sandbox import TOOL_SCHEMAS, load_seed
from eval.simulator.farewell import is_farewell
from eval.simulator.prompts import PROMPT_DIR
from eval.simulator.scenario import Scenario

RETRYABLE = (APIConnectionError, RateLimitError, InternalServerError)
MESSAGE_KEYS = ("role", "content", "tool_calls")
NOISE_INSTRUCTIONS = {
    "none": 'Speak numbers as words the way a caller does: "five five five zero one four two two one nine", '
    '"march twelfth nineteen eighty eight", "two thirty".',
    "low": "You are on a phone line transcribed by speech recognition: lowercase, no punctuation, numbers as "
    'words, an occasional "um" or "uh".',
    "high": "You are on a bad phone line transcribed by speech recognition: lowercase, no punctuation, numbers as "
    'words, fillers ("um", "like"), a small word dropped now and then, digits spoken loosely.',
}


def make_client(base_url: str, api_key: str | None, timeout_s: float = 120.0) -> AsyncOpenAI:
    """One AsyncOpenAI per endpoint. SDK retries are off; `_ChatBase` retries visibly."""
    return AsyncOpenAI(
        base_url=base_url, api_key=api_key or "none", timeout=timeout_s, max_retries=0
    )


class _ChatBase:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        temperature: float | None,
        max_tokens: int,
        retries: int,
        backoff_s: float,
    ):
        self.client, self.model = client, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.temperature_applied: bool | None = None if temperature is None else True
        self.retries, self.backoff_s = max(1, retries), backoff_s
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "requests": 0,
        }
        self.last_usage: dict[str, int] = {}
        self.last_finish_reason: str | None = None
        self.length_retries = 0
        self.last_model: str | None = None
        self.served_models: set[str] = set()

    MAX_TOKENS_CAP = 2000

    async def _create(self, messages: list[dict], tools: list[dict] | None = None) -> Any:
        """One chat completion with visible retries. If the reply is cut by max_tokens before any
        visible text (hidden output ate the budget), retry once with a larger budget."""
        resp = await self._request(messages, tools, self.max_tokens)
        choice = resp.choices[0]
        msg = choice.message
        if (
            getattr(choice, "finish_reason", None) == "length"
            and not (msg.content or "").strip()
            and not getattr(msg, "tool_calls", None)
        ):
            bigger = min(self.MAX_TOKENS_CAP, self.max_tokens * 3)
            if bigger > self.max_tokens:
                self.length_retries += 1
                resp = await self._request(messages, tools, bigger)
        self.last_finish_reason = getattr(resp.choices[0], "finish_reason", None)
        return resp

    async def _request(
        self, messages: list[dict], tools: list[dict] | None, max_tokens: int
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = None
        for attempt in range(self.retries):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                break
            except BadRequestError as e:
                if "temperature" in kwargs and "temperature" in str(e).lower():
                    kwargs.pop("temperature")
                    self.temperature, self.temperature_applied = None, False
                    continue
                raise
            except RETRYABLE:
                if attempt == self.retries - 1:
                    raise
                await asyncio.sleep(self.backoff_s * (2**attempt))
        usage = getattr(resp, "usage", None)
        self.last_usage = {}
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
            self.usage_totals["prompt_tokens"] += self.last_usage["prompt_tokens"]
            self.usage_totals["completion_tokens"] += self.last_usage["completion_tokens"]
        self.usage_totals["requests"] += 1
        served = getattr(resp, "model", None)
        if isinstance(served, str) and served:
            self.last_model = served
            self.served_models.add(served)
        return resp


class OpenAIAgent(_ChatBase):
    """The agent under test: any OpenAI-compatible URL + the contract's tool schemas."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int = 700,
        retries: int = 3,
        backoff_s: float = 1.0,
    ):
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            backoff_s=backoff_s,
        )
        self.tools = tools if tools is not None else list(TOOL_SCHEMAS.values())

    async def respond(self, messages: list[dict]) -> dict:
        resp = await self._create(messages, self.tools)
        raw = resp.choices[0].message.model_dump(exclude_none=True)
        msg = {k: raw[k] for k in MESSAGE_KEYS if k in raw}
        msg["role"] = "assistant"
        if msg.get("content") is None:
            msg["content"] = ""
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)
        return msg


_HANGUP = re.compile(r"\[\s*hang\s*-?\s*up\s*\]", re.IGNORECASE)
_LABEL = re.compile(r"^\s*(caller|customer|me|[A-Z][a-z]+)\s*:\s*")
_STAGE = re.compile(r"\*[^*]{0,80}\*|\([^)]{0,80}\)")


def clean_caller_text(raw: str) -> tuple[str, bool]:
    """Strip labels, quotes and stage directions; detect [hangup]. Empty text is NOT a hangup, but a
    reply that only says goodbye is one: a caller who says "bye" has put the phone down, and the
    local base-model caller never emits the marker."""
    hangup = bool(_HANGUP.search(raw or ""))
    text = _HANGUP.sub(" ", raw or "")
    text = _STAGE.sub(" ", text)
    text = _LABEL.sub("", text.strip())
    text = text.strip().strip('"“”').strip()
    text = re.sub(r"\s+", " ", text)
    return text, hangup or is_farewell(text)


def _render_knows(knows: dict) -> str:
    lines = []
    spoken = {k[: -len("_spoken")]: v for k, v in knows.items() if k.endswith("_spoken")}
    for k, v in knows.items():
        if k.endswith("_spoken"):
            continue
        line = f"- {k}: {v}"
        if k in spoken:
            line += f" (say it as: {spoken[k]})"
        lines.append(line)
    return "\n".join(lines) or "- nothing beyond your goal"


class LLMCaller(_ChatBase):
    """A hosted model plays the caller described by the scenario (temperature 0.3, seeded phrasing)."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        client: Any,
        model: str,
        temperature: float | None = 0.3,
        seed: int = 0,
        max_tokens: int = 600,
        retries: int = 3,
        backoff_s: float = 1.0,
    ):
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            backoff_s=backoff_s,
        )
        self.scenario, self.seed = scenario, seed
        self.max_turns = scenario.expect.max_turns
        self.clock_spoken = load_seed().get("clock_spoken", "")
        self._template = (PROMPT_DIR / "caller_system.md").read_text()
        self.last_raw: str | None = None

    def system_prompt(self, turn_idx: int) -> str:
        s, c = self.scenario, self.scenario.caller
        noise = NOISE_INSTRUCTIONS[c.noise]
        if c.stt_quirks:
            noise += " Also: " + "; ".join(c.stt_quirks) + "."
        script = "\n".join(f"- When {it.trigger}: {it.do}." for it in s.script) or "- none"
        return self._template.format(
            name=c.name,
            persona=c.persona,
            today=self.clock_spoken,
            knows=_render_knows(c.knows),
            withholds=", ".join(c.withholds) or "nothing in particular",
            goal=s.goal,
            script=script,
            noise=noise,
            seed=self.seed,
            turn_idx=turn_idx,
            max_turns=self.max_turns,
        ).strip()

    async def next_utterance(
        self, transcript: list[tuple[str, str]], turn_idx: int
    ) -> tuple[str, bool]:
        messages = [{"role": "system", "content": self.system_prompt(turn_idx)}]
        for who, text in transcript:
            messages.append({"role": "user" if who == "agent" else "assistant", "content": text})
        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "(silence)"})
        resp = await self._create(messages)
        self.last_raw = resp.choices[0].message.content or ""
        return clean_caller_text(self.last_raw)
