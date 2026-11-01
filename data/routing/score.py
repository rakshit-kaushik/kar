"""Deterministic routing scorer: label validation, per-category P/R/F1, costly confusions, coverage."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from data.routing.tickets import LABELS

NEEDS_REVIEW = "needs_review"
_COSTLY = [
    ("unauthorized_transaction", "fee_dispute"),
    ("card_lost_stolen", "card_declined"),
    ("complaint", "other"),
]
COSTLY_PAIRS: set[tuple[str, str]] = {p for a, b in _COSTLY for p in ((a, b), (b, a))}
COSTLY_WEIGHT = 3
_LABEL_RE = re.compile(
    r"\b(" + "|".join(sorted(LABELS, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def parse_label(text: str) -> str:
    """The one taxonomy label in the model's output, else needs_review (CONTRACT §9)."""
    found = {m.lower() for m in _LABEL_RE.findall(text or "")}
    return found.pop() if len(found) == 1 else NEEDS_REVIEW


def score_predictions(gold: list[str], pred: list[str]) -> dict:
    n = len(gold)
    conf: dict[str, Counter] = defaultdict(Counter)
    for g, p in zip(gold, pred, strict=True):
        conf[g][p] += 1
    labels = sorted({*gold, *(p for p in pred if p != NEEDS_REVIEW)})
    per: dict[str, dict] = {}
    for lab in labels:
        tp = conf[lab][lab]
        fp = sum(conf[g][lab] for g in conf if g != lab)
        fn = sum(c for p, c in conf[lab].items() if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(conf[lab].values())}
    gold_labels = [lab for lab in labels if any(g == lab for g in gold)]
    costly = Counter()
    weighted = 0
    for g, p in zip(gold, pred, strict=True):
        if g == p:
            continue
        if (g, p) in COSTLY_PAIRS:
            costly[f"{g}->{p}"] += 1
            weighted += COSTLY_WEIGHT
        else:
            weighted += 1
    return {
        "n": n,
        "coverage": sum(1 for p in pred if p != NEEDS_REVIEW) / n if n else 0.0,
        "accuracy": sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / n if n else 0.0,
        "macro_f1": sum(per[lab]["f1"] for lab in gold_labels) / len(gold_labels)
        if gold_labels
        else 0.0,
        "per_category": per,
        "confusion": {g: dict(c) for g, c in conf.items()},
        "costly_confusions": dict(costly),
        "weighted_error": weighted,
    }
