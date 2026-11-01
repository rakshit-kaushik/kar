"""OpenAI-compatible chat clients for the agent under test and the simulated caller.

Verified against openai 3.42.0 in .venv: `AsyncOpenAI(api_key, base_url, timeout, max_retries)`,
`client.chat.completions.create(model, messages, tools, max_tokens, temperature)`,
`resp.choices[0].message.model_dump(exclude_none=True)`, `resp.usage.prompt_tokens`.
The hosted Anthropic endpoint rejects `temperature` for claude-sonnet-5-5, so it is only sent
when explicitly configured.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError

from eval.sandbox import TOOL_SCHEMAS
from eval.simulator.prompts import PROMPT_DIR, long_date
from eval.simulator.scenario import Scenario

RETRYABLE = (APIConnectionError, RateLimitError, InternalServerError)
MESSAGE_KEYS = ("role", "content", "tool_calls")


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
        self.retries, self.backoff_s = max(1, retries), backoff_s
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "requests": 0,
        }

    async def _create(self, messages: list[dict], tools: list[dict] | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        for attempt in range(self.retries):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                break
            except RETRYABLE:
                if attempt == self.retries - 1:
                    raise
                await asyncio.sleep(self.backoff_s * (2**attempt))
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.usage_totals["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage_totals["completion_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )
        self.usage_totals["requests"] += 1
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
        max_tokens: int = 300,
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
        if "content" not in msg or msg["content"] is None:
            msg["content"] = ""
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)
        return msg


_HANGUP = re.compile(r"\[\s*hang\s*-?\s*up\s*\]", re.IGNORECASE)
_LABEL = re.compile(r"^\s*(caller|customer|me|[A-Z][a-z]+)\s*:\s*", re.IGNORECASE)


def clean_caller_text(raw: str) -> tuple[str, bool]:
    hangup = bool(_HANGUP.search(raw))
    text = _HANGUP.sub(" ", raw)
    text = _LABEL.sub("", text.strip())
    text = text.strip().strip('"“”').strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return "", True
    return text, hangup


class LLMCaller(_ChatBase):
    """A hosted model plays the caller described by the scenario."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        client: Any,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 120,
        max_turns: int = 20,
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
        self.scenario, self.max_turns = scenario, max_turns
        self._template = (PROMPT_DIR / "caller_system.md").read_text()

    def system_prompt(self, turn_idx: int) -> str:
        s = self.scenario
        p = s.persona
        persona = f"{p.name}" + (f", {p.age}" if p.age else "") + f"; manner: {p.style}"
        if p.notes:
            persona += f"; {p.notes}"
        facts = (
            "; ".join(f"{k}: {v}" for k, v in s.known_facts.items()) or "nothing beyond your goal"
        )
        hints = " ".join(s.script_hints) or "Follow the agent's questions naturally."
        corrections = (
            " ".join(
                f"When {c.when}: change {c.field} from {c.from_ or 'what you said'} to {c.to}"
                + (f', saying "{c.say}"' if c.say else "")
                + "."
                for c in s.corrections
            )
            or "none"
        )
        detours = (
            " ".join(
                f'When {d.when}: say "{d.say}".' + (f" {d.resolution}." if d.resolution else "")
                for d in s.detours
            )
            or "none"
        )
        return self._template.format(
            persona=persona,
            facts=facts,
            weekday=s.today.strftime("%A"),
            today_long=long_date(s.today),
            goal=s.goal,
            hints=hints,
            corrections=corrections,
            detours=detours,
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
        raw = resp.choices[0].message.content or ""
        return clean_caller_text(raw)
