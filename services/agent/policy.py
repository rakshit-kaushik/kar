"""Contract handoff rules as application logic (reviewer decision pending, Nov 1 09:20).

CONTRACT.md's handoff rules say a suspected-fraud report ends in a transfer. In the 08:40 rehearsal the final model opened the right
unauthorized_transaction ticket in 5 of 5 calls and transferred in 0 of 5 on its own (E's main remaining failure family). This
module enforces the rule after the fact: when a reply ends and the call state holds a fraud-category ticket with no handoff
recorded by the model or a guard, the agent records request_handoff(suspected_fraud) through the executor and appends one fixed
sentence. Identical for every served model; logged as a `policy_handoff` event with its own count, never shown as the model's
behaviour. AGENT_POLICY_HANDOFF=0 turns it off."""
from __future__ import annotations

import inspect
import os
from typing import Any, AsyncIterable, Callable

FRAUD_CATEGORIES = ("unauthorized_transaction", "card_lost_stolen")
POLICY_LINE = os.environ.get("AGENT_POLICY_HANDOFF_LINE", "I'm transferring you to our fraud team now.")


def required_handoff(state: dict[str, Any]) -> str | None:
    """The handoff reason the contract requires for this call state, or None when none is due or one is already recorded."""
    if state.get("handoff"):
        return None
    for c in state.get("completed") or []:
        if c.get("tool") == "create_ticket" and str((c.get("args") or {}).get("category", "")) in FRAUD_CATEGORIES:
            return "suspected_fraud"
    return None


class HandoffPolicy:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @classmethod
    def from_env(cls) -> "HandoffPolicy":
        return cls(enabled=os.environ.get("AGENT_POLICY_HANDOFF", "1") != "0")


def _text_of(chunk: Any) -> str | None:
    if isinstance(chunk, str):
        return chunk
    return getattr(getattr(chunk, "delta", None), "content", None)


async def policy_stream(source: AsyncIterable[Any], policy: HandoffPolicy, state_view: Callable[[], dict[str, Any]], on_policy: Callable[[str], Any]):
    """Pass the reply through. When it ends without a tool call and state_view() (read then, after any claim-guard handoff) still
    requires a handoff, run on_policy(reason) (awaited if async) and yield POLICY_LINE as one extra sentence."""
    if not policy.enabled:
        async for chunk in source:
            yield chunk
        return
    tool_call = False
    async for chunk in source:
        if not _text_of(chunk) and getattr(getattr(chunk, "delta", None), "tool_calls", None):
            tool_call = True  # the model is acting this turn (possibly request_handoff itself); the rule applies to the spoken reply after the result
        yield chunk
    if tool_call:
        return
    reason = required_handoff(state_view())
    if reason:
        res = on_policy(reason)
        if inspect.isawaitable(res):
            await res
        yield " " + POLICY_LINE
