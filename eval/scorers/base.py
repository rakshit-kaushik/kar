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

    def to_dict(self) -> dict:
        return {"scorer": self.scorer, "passed": self.passed, "detail": self.detail}


Scorer = Callable[[Episode, Scenario], ScorerResult]
