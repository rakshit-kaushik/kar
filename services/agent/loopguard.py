"""Loop guard for the agent's LLM node (application logic, no retrain).

E's run-1 eval found 6 of 81 episodes repeating one assistant line three or more times (re-asking the day, repeating a claim
instead of handing off); the first live calls on the candidate showed the same loop in 2 of 4 calls. The guard holds a
streaming reply only while it is still identical to the previous assistant line (ordinary replies diverge within the first
words, so they stream untouched); an exact repeat is swallowed and REPHRASE is spoken instead, a second consecutive repeat
becomes HANDOFF_LINE plus a request_handoff recorded by the agent. A repeated farewell is exempt (the caller kept saying
goodbye without hanging up). The model never sees the guard: what was spoken is what
enters the chat context. AGENT_LOOP_GUARD=0 turns it off; every intervention is a `loop_guard` turn event."""
from __future__ import annotations

import inspect
import os
import re
from typing import Any, AsyncIterable, Callable

REPHRASE = os.environ.get("AGENT_LOOP_REPHRASE", "Sorry, I may have missed that. Could you tell me again what you'd like?")
HANDOFF_LINE = os.environ.get("AGENT_LOOP_HANDOFF_LINE", "I'm sorry, I'm having trouble with this one. Let me get a colleague to help you.")


FAREWELL_RE = re.compile(r"\b(goodbye|good bye|bye|take care|have a (good|great|nice) (day|one|evening|afternoon|morning))\b")


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def is_farewell(norm: str | None) -> bool:
    """A repeated closing line is not a loop: the caller kept saying goodbye without hanging up."""
    return bool(norm) and FAREWELL_RE.search(norm) is not None


class TurnFilter:
    """One reply: feed() the text deltas as they stream; the state is 'hold' while the text so far still matches the previous
    line, 'release' once it differs (or outgrows it). finish() at the end of the stream says 'repeat' or 'release'."""

    def __init__(self, prev_norm: str | None) -> None:
        self.prev = prev_norm
        self.buf = ""
        self.state = "hold" if prev_norm else "release"

    def feed(self, delta: str) -> str:
        self.buf += delta
        if self.state == "hold":
            n = normalize(self.buf)
            if len(n) > len(self.prev) or not self.prev.startswith(n):  # type: ignore[arg-type]
                self.state = "release"
        return self.state

    def finish(self) -> str:
        if self.state == "hold" and self.prev and normalize(self.buf) == self.prev:
            return "repeat"
        return "release"


class LoopGuard:
    def __init__(self, enabled: bool = True, rephrase: str = REPHRASE, handoff_line: str = HANDOFF_LINE, handoff_after: int = 3) -> None:
        self.enabled = enabled
        # consecutive exact repeats before the guard hands off; repeats below it are rephrased. 3, not 2: a caller utterance split
        # by turn detection ("five five five zero one four" / "two two one nine", Nov 1 07:12) makes one re-ask legitimate
        self.handoff_after = max(1, handoff_after)
        self.rephrase = rephrase
        self.handoff_line = handoff_line
        self.prev: str | None = None  # normalized previous model reply (what the model generated, even when swallowed)
        self.repeats = 0  # consecutive repeats caught

    @classmethod
    def from_env(cls) -> "LoopGuard":
        return cls(enabled=os.environ.get("AGENT_LOOP_GUARD", "1") != "0", handoff_after=int(os.environ.get("AGENT_LOOP_GUARD_HANDOFF_AFTER", "3")))


def _text_of(chunk: Any) -> str | None:
    if isinstance(chunk, str):
        return chunk
    return getattr(getattr(chunk, "delta", None), "content", None)


async def guarded_stream(source: AsyncIterable[Any], guard: LoopGuard, on_loop: Callable[[str, str], Any]):
    """Wrap an LLM node's chunk stream. Tool-call chunks and sentinels pass through at once; text is held only while it is still
    a prefix of the previous reply. on_loop(action, repeated_text) runs (awaited if async) when a repeat is replaced."""
    if not guard.enabled:
        async for chunk in source:
            yield chunk
        return
    turn, held = TurnFilter(None if is_farewell(guard.prev) else guard.prev), []
    async for chunk in source:
        text = _text_of(chunk)
        if not text:
            yield chunk
            continue
        if turn.state == "release":
            turn.feed(text)  # keep the full text for the next comparison
            yield chunk
            continue
        held.append(chunk)
        if turn.feed(text) == "release":
            for c in held:
                yield c
            held = []
    action = turn.finish()
    guard.prev = normalize(turn.buf)
    if action == "release":
        guard.repeats = 0
        for c in held:
            yield c
        return
    guard.repeats += 1
    kind = "rephrase" if guard.repeats < guard.handoff_after else "handoff"
    res = on_loop(kind, turn.buf.strip())
    if inspect.isawaitable(res):
        await res
    yield guard.rephrase if kind == "rephrase" else guard.handoff_line
