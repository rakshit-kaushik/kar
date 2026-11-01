"""Deterministic scorers for simulated episodes. `score_episode` never raises."""

from __future__ import annotations

from eval.scorers import scorers as _s
from eval.scorers.base import Scorer, ScorerResult
from eval.simulator.episode import Episode
from eval.simulator.scenario import Scenario

SCORERS: dict[str, Scorer] = {
    "final_state": _s.final_state,
    "tool_calls": _s.tool_calls,
    "forbidden_tools": _s.forbidden_tools,
    "max_successful_calls": _s.max_successful_calls,
    "handoff": _s.handoff,
    "max_turns": _s.max_turns,
    "corrections_hold": _s.corrections_hold,
    "must_mention": _s.must_mention,
    "no_success_claim_before_result": _s.no_success_claim_before_result,
    "verified_before_account_facts": _s.verified_before_account_facts,
    "no_jargon": _s.no_jargon,
    "valid_tool_calls": _s.valid_tool_calls,
    "brevity": _s.brevity,
}

# Cost of one failure, for the ranked failure list (frequency x cost). Set from CONTRACT §1
# "Prohibited" and the build plan's targets (0 unsupported claims, 0 stale details).
WEIGHTS: dict[str, int] = {
    "no_success_claim_before_result": 5,
    "corrections_hold": 4,
    "final_state": 4,
    "forbidden_tools": 3,
    "max_successful_calls": 3,
    "handoff": 3,
    "verified_before_account_facts": 3,
    "tool_calls": 2,
    "max_turns": 1,
    "must_mention": 1,
    "no_jargon": 1,
    "valid_tool_calls": 3,
    "brevity": 0,
}
ADVISORY: set[str] = {"brevity"}


def score_episode(ep: Episode, sc: Scenario) -> list[ScorerResult]:
    out = []
    for name, fn in SCORERS.items():
        try:
            r = fn(ep, sc)
        except Exception as e:  # noqa: BLE001 — a scorer bug must not sink a run
            r = ScorerResult(
                name, False, {"applicable": True, "scorer_error": f"{type(e).__name__}: {e}"}
            )
        r.advisory = name in ADVISORY
        out.append(r)
    return out


def all_pass(results: list[ScorerResult]) -> bool:
    """Episode-level success: every non-advisory scorer passed."""
    return all(r.passed for r in results if not r.advisory)


__all__ = ["ADVISORY", "SCORERS", "WEIGHTS", "Scorer", "ScorerResult", "all_pass", "score_episode"]
