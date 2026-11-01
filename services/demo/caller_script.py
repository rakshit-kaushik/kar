"""Caller brains for scripts/sim_caller.py.

AnsweringScript: fixed lines, except that when the agent asks again for a fact the caller has already given (phone, date of
birth, weekday, time, branch) the caller repeats that fact instead of reading its next line, as a person would, and the script
does not advance. Nov 1 rehearsal: the fixed-line caller fell a turn behind whenever the model re-asked the weekday or a
credential, so those rows measured the harness as much as the model. Rows run with this brain are labelled -adaptive."""
from __future__ import annotations

import re

RE_ASKS = {
    "phone": re.compile(r"phone number", re.I),
    "dob": re.compile(r"date of birth|birth ?date", re.I),
    "day": re.compile(r"\b(which|what) day\b", re.I),
    "time": re.compile(r"\b(which|what) time\b", re.I),
    "branch": re.compile(r"\b(which|what) branch\b", re.I),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


class AnsweringScript:
    def __init__(self, lines: list[str], facts: dict[str, str], max_reasks: int = 3) -> None:
        self.lines, self.i = list(lines), 0
        self.facts = {k: v for k, v in facts.items() if k in RE_ASKS and v}
        self.max_reasks, self.reasks = max_reasks, 0
        self.last_reask: str | None = None

    def given(self, key: str, transcript: list[tuple[str, str]]) -> bool:
        fact = _norm(self.facts[key])
        return any(fact in _norm(text) for role, text in transcript if role == "caller")

    def answer_for(self, transcript: list[tuple[str, str]]) -> tuple[str, str] | None:
        last = next((text for role, text in reversed(transcript) if role == "agent"), None)
        if not last:
            return None
        for key, rx in RE_ASKS.items():
            if key in self.facts and rx.search(last) and self.given(key, transcript):
                return key, self.facts[key]
        return None

    async def next_line(self, transcript: list[tuple[str, str]]) -> str | None:
        self.last_reask = None
        if self.reasks < self.max_reasks:
            hit = self.answer_for(transcript)
            if hit:
                self.reasks += 1
                self.last_reask = hit[0]
                return hit[1]
        if self.i >= len(self.lines):
            return None
        line = self.lines[self.i]
        self.i += 1
        return line


def role_for(attrs: dict, own_track_sids: set[str]) -> str:
    """Who a lk.transcription text stream is about. The agent publishes its own speech and the caller's transcription alike,
    both with lk.transcribed_track_id; a line is the caller's only when that id is one of the caller's own published tracks."""
    tid = attrs.get("lk.transcribed_track_id")
    return "caller" if tid and tid in own_track_sids else "agent"
