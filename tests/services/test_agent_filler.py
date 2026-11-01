"""Filler line before tool-call turns (reviewer decision Nov 1 04:50): spoken once per caller turn, only when a tool
is about to execute, never while the agent is already speaking; AGENT_FILLER empty switches it off."""
from services.agent.filler import FillerPolicy, filler_text


def test_filler_once_per_turn_and_not_while_speaking():
    p = FillerPolicy("One moment.")
    assert p.decide(turn_idx=3, agent_state="thinking", tool_name="lookup_account") == "One moment."
    assert p.decide(turn_idx=3, agent_state="thinking", tool_name="book_appointment") is None  # second tool, same turn
    assert p.decide(turn_idx=4, agent_state="speaking", tool_name="lookup_account") is None  # already talking
    assert p.decide(turn_idx=4, agent_state="thinking", tool_name="lookup_account") == "One moment."  # refusal did not use the turn


def test_empty_filler_disables(monkeypatch):
    monkeypatch.setenv("AGENT_FILLER", "")
    assert filler_text() == ""
    assert FillerPolicy("").decide(turn_idx=1, agent_state="thinking", tool_name="lookup_account") is None
    monkeypatch.delenv("AGENT_FILLER")
    assert filler_text() == "One moment."
