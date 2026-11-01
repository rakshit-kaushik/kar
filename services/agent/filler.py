"""Filler line before tool-call turns (reviewer decision, Nov 1 04:50).

A tool-call turn has no audio until the model has finished writing the tool call, the tool has run and the model has
spoken the result (about 2 s end to end on the run-1 candidate); a short spoken filler the moment a tool starts fills
that gap. It is spoken only when a tool is about to execute (never on spoken turns), at most once per caller turn, never
while the agent is already speaking, and it is not added to the chat context, so the model never sees it. AGENT_FILLER
sets the text; empty switches it off. The agent logs each one as a `filler_start` turn event so scripts/g1_report.py can
show first audio heard (filler included) beside the first audible reply."""
from __future__ import annotations

import os

DEFAULT_FILLER = "One moment."


def filler_text() -> str:
    return os.environ.get("AGENT_FILLER", DEFAULT_FILLER).strip()


class FillerPolicy:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self._last_turn: int | None = None

    def decide(self, turn_idx: int, agent_state: str, tool_name: str) -> str | None:
        """The text to speak before `tool_name` runs in caller turn `turn_idx`, or None."""
        if not self.text or agent_state == "speaking" or self._last_turn == turn_idx:
            return None
        self._last_turn = turn_idx
        return self.text
