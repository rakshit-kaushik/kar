"""The simulated caller's view of a call: per-turn latency as a listener would experience it.

Events, all in wall-clock ms: the moment the caller's last audio sample was played out
(`caller_speech_end`), the agent's state attribute flips (`agent_state`: listening/thinking/speaking),
and received agent audio frames (`agent_audio` with their RMS). The first non-silent frame after the
caller stopped talking is "first audio", the number the plan budgets at p50 <= 1.0 s / p95 <= 1.6 s.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class Turn:
    idx: int
    caller_speech_end_ms: int
    thinking_ms: int | None = None
    speaking_ms: int | None = None
    first_audio_ms: int | None = None
    agent_speech_end_ms: int | None = None
    extra: dict = field(default_factory=dict)


class CallerTimeline:
    def __init__(self, silence_rms: float = 300.0) -> None:
        self.turns: list[Turn] = []
        self.silence_rms = silence_rms
        self._state = "unknown"
        self._awaiting_reply = False

    # -- events --------------------------------------------------------------------------------------
    def caller_speech_end(self, ts_ms: int) -> Turn:
        t = Turn(idx=len(self.turns) + 1, caller_speech_end_ms=ts_ms)
        self.turns.append(t)
        self._awaiting_reply = True
        return t

    def agent_state(self, ts_ms: int, state: str) -> None:
        prev, self._state = self._state, state
        t = self.turns[-1] if self.turns and self._awaiting_reply else None
        if t is not None:
            if state == "thinking" and t.thinking_ms is None:
                t.thinking_ms = ts_ms
            elif state == "speaking" and t.speaking_ms is None:
                t.speaking_ms = ts_ms
            elif state == "listening" and prev == "speaking" and (t.speaking_ms is not None or t.first_audio_ms is not None):
                t.agent_speech_end_ms = ts_ms
                self._awaiting_reply = False

    def agent_audio(self, ts_ms: int, rms: float) -> None:
        if rms < self.silence_rms or not self.turns or not self._awaiting_reply:
            return
        t = self.turns[-1]
        if t.first_audio_ms is None and ts_ms > t.caller_speech_end_ms:
            t.first_audio_ms = ts_ms

    # -- queries -------------------------------------------------------------------------------------
    def agent_is_listening(self) -> bool:
        return self._state == "listening" and not self._awaiting_reply

    def latencies(self) -> list[dict]:
        out = []
        for t in self.turns:
            e = t.caller_speech_end_ms
            out.append({
                "turn": t.idx,
                "eos_to_thinking_ms": None if t.thinking_ms is None else t.thinking_ms - e,
                "eos_to_speaking_ms": None if t.speaking_ms is None else t.speaking_ms - e,
                "eos_to_first_audio_ms": None if t.first_audio_ms is None else t.first_audio_ms - e,
                "agent_speech_ms": None if (t.agent_speech_end_ms is None or t.speaking_ms is None) else t.agent_speech_end_ms - t.speaking_ms,
            })
        return out

    def summary(self) -> dict:
        lat = self.latencies()
        xs = [l["eos_to_first_audio_ms"] for l in lat if l["eos_to_first_audio_ms"] is not None]
        s: dict = {"turns": len(lat), "turns_with_audio": len(xs)}
        if xs:
            s["eos_to_first_audio_ms"] = {"p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": max(xs), "mean": statistics.fmean(xs)}
        return s
