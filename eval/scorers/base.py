from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from eval.simulator.episode import Episode
from eval.simulator.scenario import Scenario


@dataclass
class ScorerResult:
    scorer: str
    passed: bool
    detail: dict = field(default_factory=dict)
    turn_idx: int | None = None  # caller turn (1-based) where the failure happened, when known
    advisory: bool = False  # report only; never fails an episode

    def to_dict(self) -> dict:
        return {
            "scorer": self.scorer,
            "passed": self.passed,
            "detail": self.detail,
            "turn_idx": self.turn_idx,
            "advisory": self.advisory,
        }


Scorer = Callable[[Episode, Scenario], ScorerResult]
