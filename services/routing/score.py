"""Routing scores (CONTRACT §9): accuracy, macro F1 over the categories, coverage (valid labels), and the
confusion count weighted 3x for the costly pairs. `needs_review` is never a class, only a miss."""
from __future__ import annotations

from collections import Counter

from services.routing.batch import LABELS

COSTLY = {
    ("unauthorized_transaction", "fee_dispute"), ("fee_dispute", "unauthorized_transaction"),  # fraud has a regulatory clock
    ("card_lost_stolen", "card_declined"), ("card_declined", "card_lost_stolen"),                # block vs no block
    ("complaint", "other"),                                                                       # lost escalation
}


def score(gold: list[str], pred: list[str]) -> dict:
    assert len(gold) == len(pred)
    n = len(gold)
    pairs = list(zip(gold, pred))
    correct = sum(g == p for g, p in pairs)
    per_label: dict[str, dict] = {}
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(p == lab and g != lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = (2 * prec * rec / (prec + rec)) if prec is not None and rec is not None and (prec + rec) else (0.0 if tp + fp + fn else None)
        per_label[lab] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
        if f1 is not None:
            f1s.append(f1)
    return {
        "n": n, "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "coverage": sum(p in LABELS for p in pred) / n if n else 0.0,
        "weighted_error": sum(0 if g == p else (3 if (g, p) in COSTLY else 1) for g, p in pairs),
        "costly": dict(Counter(f"{g}->{p}" for g, p in pairs if g != p and (g, p) in COSTLY)),
        "per_label": per_label,
    }
