"""Caller brains for scripts/sim_caller.py.

AnsweringScript: fixed lines, except that when the agent asks again for a fact the caller has already given (phone, date of
birth, weekday, time, branch) the caller repeats that fact instead of reading its next line, as a person would, and the script
does not advance. Nov 1 rehearsal: the fixed-line caller fell a turn behind whenever the model re-asked the weekday or a
credential, so those rows measured the harness as much as the model. Rows run with this brain are labelled -adaptive.

Nov 1, after the redeploy of run 2: the fixed lines had been tuned on run 1's flow (re-ask the day, ask the time, read back). Run 2
proposes a slot straight after the request and says goodbye once it has booked, so a fixed-line caller accepted a time it never
wanted and kept talking after the call was over. Three more rules, in this order after the re-ask rule: a proposal or read-back
naming a different time, day, branch or service than the caller's fact gets the fact ("nine thirty, please"); the agent's goodbye
gets the script's last line and ends the call; a scripted line that only restates a fact already given is skipped."""
from __future__ import annotations

import re

RE_ASKS = {
    "phone": re.compile(r"phone number", re.I),
    "dob": re.compile(r"date of birth|birth ?date", re.I),
    "day": re.compile(r"\b(which|what) day\b", re.I),
    "time": re.compile(r"\b(which|what) time\b", re.I),
    "branch": re.compile(r"\b(which|what) branch\b", re.I),
    "service": re.compile(r"\b(which|what) (service|type of appointment|kind of appointment)\b", re.I),
}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_BRANCHES = ("market street", "marlow street", "marker street", "mill road")  # eval/sandbox/seed.json branch names
_SERVICES = ("account opening", "mortgage consultation", "loan application", "financial review", "notary", "safe deposit box")
_TIME = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)( (oh five|fifteen|thirty|forty five|o clock))? (in the (morning|afternoon|evening)|a m|p m)\b|\bnoon\b")
_FAREWELL = re.compile(r"\b(goodbye|bye)\b|have a (great|good|nice|lovely) (day|evening|afternoon)", re.I)
_FILLER = {"", "is fine", "works", "please", "sounds good", "is good", "would be great", "that works"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


class AnsweringScript:
    def __init__(self, lines: list[str], facts: dict[str, str], max_reasks: int = 3, max_corrections: int = 3) -> None:
        self.lines, self.i = list(lines), 0
        self.facts = {k: v for k, v in facts.items() if k in RE_ASKS and v}
        self.max_reasks, self.reasks = max_reasks, 0
        self.max_corrections, self.corrections = max_corrections, 0
        self.last_reask: str | None = None
        self.done = False

    def contradicted(self, agent_text: str) -> str | None:
        """The first fact the agent's line names a different value for (it mentions a value of that kind, but not the caller's)."""
        text = _norm(agent_text)
        kinds = {"time": bool(_TIME.search(text)), "day": any(w in text.split() for w in _WEEKDAYS),
                 "branch": any(b in text for b in _BRANCHES), "service": any(v in text for v in _SERVICES)}
        for key in ("service", "day", "branch", "time"):
            if key in self.facts and kinds[key] and _norm(self.facts[key]) not in text:
                return key
        return None

    def redundant(self, line: str, transcript: list[tuple[str, str]]) -> bool:
        """A scripted line that only restates a fact the caller has already given ("nine thirty is fine" after "nine thirty, please")."""
        text = _norm(line)
        for key, value in self.facts.items():
            v = _norm(value)
            if v in text and text.replace(v, "").strip() in _FILLER and self.given(key, transcript):
                return True
        return False

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
        if self.done:
            return None
        if self.reasks < self.max_reasks:
            hit = self.answer_for(transcript)
            if hit:
                self.reasks += 1
                self.last_reask = hit[0]
                return hit[1]
        last = next((text for role, text in reversed(transcript) if role == "agent"), "")
        if last and self.corrections < self.max_corrections:
            key = self.contradicted(last)
            if key:
                self.corrections += 1
                self.last_reask = f"{key}:corrected"
                return f"{self.facts[key]}, please"
        if last and _FAREWELL.search(last) and "?" not in last and self.lines:
            self.done, self.last_reask = True, "farewell"
            return self.lines[-1]
        while self.i < len(self.lines) and self.redundant(self.lines[self.i], transcript):
            self.i += 1
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
