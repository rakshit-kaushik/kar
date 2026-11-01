"""Claim guard for the agent's LLM node (application logic, no retrain; reviewer decision from the run-1 failure labels).

In 5 of 81 text episodes the run-1 adapter said "I've opened your ticket and I'm transferring you" right after create_ticket
and called request_handoff a turn late or never: the contract counts a transfer announced without the tool as an unsupported
claim. The guard watches the streamed reply; the moment the text completes a transfer phrase while no handoff exists in the
call state, it records request_handoff through the executor (reason derived from the state) before the completing chunk
reaches the TTS, and logs a `claim_guard` turn event. Identical for every served model; reported as the agent's intervention,
never as the model's recovery. AGENT_CLAIM_GUARD=0 turns it off.

Completed-action claims (reviewer, Nov 1 08:15, after a live call on run 2 said "I booked your mortgage consultation" to a caller
who had just declined, with no book_appointment call): a sentence that claims a finished write (booked, moved/rescheduled,
cancelled, ticket opened) with no successful result of the matching tool in the call state is held back whole and replaced by a
question ("I haven't booked anything yet. Would you like me to go ahead and book it?"). Text is held only until its sentence ends
(Deepgram's TTS buffers to the sentence boundary anyway); read-backs, questions and future tense ("I'll book that") pass untouched;
a tool call in the same reply releases the held text unchecked, since the action is being taken. Same `claim_guard` event, with
`kind` and `tool`."""
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


def state_view(state: Any) -> dict[str, Any]:
    """What the guard reads: the contract's session state plus the verification-failure count, which the state schema does not
    carry (CallState.to_dict is the schema; lane S does not widen it)."""
    return {**state.to_dict(), "verification_failures": getattr(state, "verification_failures", 0),
            "tool_failures": getattr(state, "tool_failures", 0), "business_errors": getattr(state, "business_errors", 0)}


def handoff_reason(state: dict[str, Any]) -> str:
    """The contract's handoff reason that fits the call so far (schemas/tools/request_handoff.json)."""
    for c in state.get("completed") or []:
        if c.get("tool") == "create_ticket" and str((c.get("args") or {}).get("category", "")) in ("unauthorized_transaction", "suspected_fraud", "fraud"):
            return "suspected_fraud"
    if (state.get("verification_failures") or 0) > 0 and not (state.get("verified") or {}).get("customer_id"):
        return "verification_failed"
    if (state.get("tool_failures") or 0) > 0:  # an infrastructure failure earlier in the call (timeout, unavailable: state.INFRA_ERRORS)
        return "tool_failure"
    return "out_of_scope"


ACTION_CLAIM_RE = re.compile(
    r"\b(?:I(?:'ve| have)? (?:booked|rescheduled|moved|cancell?ed)|you(?:'re| are) (?:now |all )?booked|"
    r"(?:is|are|was|has been|have been) (?:now )?(?:booked|rescheduled|moved|cancell?ed)|"
    r"I(?:'ve| have)? (?:opened|created|filed|raised) (?:an? |your |the )?(?:[\w-]+ ){0,4}?ticket|"
    r"ticket (?:has been|is now|is|was) (?:created|opened|filed|raised))\b", re.I)
ACTION_TOOLS = {"booked": "book_appointment", "moved": "reschedule_appointment", "rescheduled": "reschedule_appointment",
                "cancelled": "cancel_appointment", "canceled": "cancel_appointment", "ticket": "create_ticket"}
SUPPORTING_TOOLS = {  # a rescheduled appointment is booked at its new time; the other claims need their own tool
    "book_appointment": {"book_appointment", "reschedule_appointment"}, "reschedule_appointment": {"reschedule_appointment"},
    "cancel_appointment": {"cancel_appointment"}, "create_ticket": {"create_ticket"}}
REPLACEMENT_LINES = {
    "book_appointment": "I haven't booked anything yet. Would you like me to go ahead and book it?",
    "reschedule_appointment": "I haven't moved anything yet. Would you like me to go ahead and reschedule it?",
    "cancel_appointment": "I haven't cancelled anything yet. Would you like me to go ahead and cancel it?",
    "create_ticket": "I haven't opened a ticket yet. Would you like me to go ahead and open one?",
}
SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s*$")


def action_claimed(text: str) -> tuple[str, str] | None:
    """(kind, tool) of the first completed-action claim in the text, or None. Kinds: booked, moved, rescheduled, cancelled, ticket."""
    m = ACTION_CLAIM_RE.search(text)
    if not m:
        return None
    words = m.group(0).lower()
    for word, tool in ACTION_TOOLS.items():
        if word in words:
            return ("cancelled" if word == "canceled" else word), tool
    return None


def supported(tool: str, completed: set[str]) -> bool:
    return bool(SUPPORTING_TOOLS.get(tool, {tool}) & set(completed))


def replacement_line(tool: str) -> str:
    return REPLACEMENT_LINES.get(tool, "I haven't done that yet. Would you like me to go ahead?")


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


async def _await(res: Any) -> None:
    if inspect.isawaitable(res):
        await res


async def claim_guarded_stream(source: AsyncIterable[Any], guard: ClaimGuard, handoff_done: Callable[[], bool], on_claim: Callable[[str], Any],
                               completed_tools: Callable[[], set[str]] | None = None, on_action_claim: Callable[[str, str, str], Any] | None = None):
    """Transfers: every chunk passes through; the chunk that completes a transfer phrase waits until on_claim(text_so_far) has run
    (awaited if async) when no handoff exists yet, at most once per reply. Completed-action claims (only when completed_tools is
    given): text is held until its sentence ends; a sentence claiming a finished write with no supporting tool in completed_tools()
    is dropped together with the rest of the reply, on_action_claim(kind, tool, sentence) runs, and the replacement question is
    yielded instead. A chunk carrying tool calls releases the held text unchecked."""
    if not guard.enabled:
        async for chunk in source:
            yield chunk
        return
    buf, fired = "", False
    held: list[Any] = []
    sent, replaced = "", False
    async for chunk in source:
        if replaced:
            continue  # the rest of a replaced reply is dropped
        text = _text_of(chunk)
        if text and not fired:
            buf += text
            if transfer_claimed(buf) and not handoff_done():
                fired = True
                await _await(on_claim(buf.strip()))
        if completed_tools is None:
            yield chunk
            continue
        if not text:
            if getattr(getattr(chunk, "delta", None), "tool_calls", None):
                for c in held:
                    yield c
                held, sent = [], ""
            yield chunk
            continue
        held.append(chunk)
        sent += text
        if not SENTENCE_END_RE.search(sent):
            continue
        verdict = action_claimed(sent)
        if verdict and not supported(verdict[1], completed_tools()):
            replaced = True
            if on_action_claim is not None:
                await _await(on_action_claim(verdict[0], verdict[1], sent.strip()))
            yield replacement_line(verdict[1])
            held, sent = [], ""
            continue
        for c in held:
            yield c
        held, sent = [], ""
    if held and not replaced:
        verdict = action_claimed(sent)
        if verdict and completed_tools is not None and not supported(verdict[1], completed_tools()):
            if on_action_claim is not None:
                await _await(on_action_claim(verdict[0], verdict[1], sent.strip()))
            yield replacement_line(verdict[1])
            return
        for c in held:
            yield c
