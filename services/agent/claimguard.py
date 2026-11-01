"""Claim guard for the agent's LLM node (application logic, no retrain; reviewer decision from the run-1 failure labels).

In 5 of 81 text episodes the run-1 adapter said "I've opened your ticket and I'm transferring you" right after create_ticket
and called request_handoff a turn late or never: the contract counts a transfer announced without the tool as an unsupported
claim. The guard watches the streamed reply; the moment the text completes a transfer phrase while no handoff exists in the
call state, it records request_handoff through the executor (reason derived from the state) before the completing chunk
reaches the TTS, and logs a `claim_guard` turn event. Identical for every served model; reported as the agent's intervention,
never as the model's recovery. AGENT_CLAIM_GUARD=0 turns it off."""
from __future__ import annotations

import inspect
import os
import re
from typing import Any, AsyncIterable, Callable

from services.agent.loopguard import normalize

TRANSFER_RE = re.compile(
    r"\b(transferr?ing you|transfer you|hand(?:ing)? you (?:over|off)|connect(?:ing)? you (?:to|with)|put(?:ting)? you through|"
    r"(?:a |my |our )?(?:specialist|colleague|team member|representative|agent|person) (?:will|is going to|can) (?:take over|take it from here|help|be with you|assist)|"
    r"let me get (?:a|my) (?:colleague|specialist))\b"
)


def transfer_claimed(text: str) -> bool:
    return TRANSFER_RE.search(normalize(text)) is not None


def handoff_reason(state: dict[str, Any]) -> str:
    """The contract's handoff reason that fits the call so far (schemas/tools/request_handoff.json)."""
    for c in state.get("completed") or []:
        if c.get("tool") == "create_ticket" and str((c.get("args") or {}).get("category", "")) in ("unauthorized_transaction", "suspected_fraud", "fraud"):
            return "suspected_fraud"
    if (state.get("verification_failures") or 0) > 0 and not (state.get("verified") or {}).get("customer_id"):
        return "verification_failed"
    return "out_of_scope"


class ClaimGuard:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @classmethod
    def from_env(cls) -> "ClaimGuard":
        return cls(enabled=os.environ.get("AGENT_CLAIM_GUARD", "1") != "0")


def _text_of(chunk: Any) -> str | None:
    if isinstance(chunk, str):
        return chunk
    return getattr(getattr(chunk, "delta", None), "content", None)


async def claim_guarded_stream(source: AsyncIterable[Any], guard: ClaimGuard, handoff_done: Callable[[], bool], on_claim: Callable[[str], Any]):
    """Pass every chunk through; the chunk that completes a transfer phrase waits until on_claim(text_so_far) has run (awaited
    if async) when no handoff exists yet. At most once per reply."""
    if not guard.enabled:
        async for chunk in source:
            yield chunk
        return
    buf, fired = "", False
    async for chunk in source:
        text = _text_of(chunk)
        if text and not fired:
            buf += text
            if transfer_claimed(buf) and not handoff_done():
                fired = True
                res = on_claim(buf.strip())
                if inspect.isawaitable(res):
                    await res
        yield chunk
