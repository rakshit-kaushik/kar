"""Deterministic scorers for simulated episodes. `score_episode` never raises."""

from __future__ import annotations

from eval.scorers import scorers as _s
from eval.scorers.base import Scorer, ScorerResult
from eval.simulator.episode import Episode
from eval.simulator.scenario import Scenario

SCORERS: dict[str, Scorer] = {
    "final_state": _s.final_state,
    "tool_steps": _s.tool_steps,
    "no_unsupported_success_claim": _s.no_unsupported_success_claim,
    "turn_budget": _s.turn_budget,
    "handoff": _s.handoff,
    "corrected_details": _s.corrected_details,
    "verified_before_account_facts": _s.verified_before_account_facts,
    "no_jargon": _s.no_jargon,
    "brevity": _s.brevity,
}

# Cost of a failure, for the ranked failure list (frequency x cost). Proposals until the
# baseline; derived from CONTRACT §1 "Prohibited" and the build plan's targets.
WEIGHTS: dict[str, int] = {
    "no_unsupported_success_claim": 5,
    "corrected_details": 4,
    "final_state": 4,
    "handoff": 3,
    "verified_before_account_facts": 3,
    "tool_steps": 2,
    "turn_budget": 1,
    "no_jargon": 1,
    "brevity": 1,
}


def score_episode(ep: Episode, sc: Scenario) -> list[ScorerResult]:
    out = []
    for name, fn in SCORERS.items():
        try:
            out.append(fn(ep, sc))
        except Exception as e:  # noqa: BLE001 — a scorer bug must not sink a run
            out.append(ScorerResult(name, False, {"scorer_error": f"{type(e).__name__}: {e}"}))
    return out


__all__ = ["SCORERS", "WEIGHTS", "Scorer", "ScorerResult", "score_episode"]
