"""Synthetic support tickets labeled at generation time (CONTRACT §9 taxonomy).

The hosted model writes tickets for one category per request under a variation spec (length,
tone, noise, channel); the label is the category asked for, never inferred. Tickets are deduped,
given ids R-00001.., and split with a stratified 100-ticket test set. Training files are produced
by T's `train.routing_format` from the resulting tickets.jsonl.
"""

from __future__ import annotations

import json
import random
import re

from eval.sandbox.store import TICKET_CATEGORIES

LABELS: list[str] = list(TICKET_CATEGORIES)
LENGTHS = ["short", "medium", "long"]
TONES = [
    "annoyed",
    "polite",
    "worried",
    "terse",
    "rambling",
    "formal",
    "confused",
    "angry but civil",
    "apologetic",
]
NOISES = [
    "clean",
    "typos",
    "no punctuation or caps",
    "voice transcript (fillers, no punctuation)",
    "copy-pasted with extra whitespace",
    "non-native phrasing",
]
CHANNELS = ["email", "chat", "voicemail transcript", "web form"]


def variation_specs(n: int, seed: int) -> list[dict]:
    """n specs; lengths cycle so every batch covers short/medium/long; the rest is seeded random."""
    rng = random.Random(seed)
    return [
        {
            "length": LENGTHS[i % 3],
            "tone": rng.choice(TONES),
            "noise": rng.choice(NOISES),
            "channel": rng.choice(CHANNELS),
        }
        for i in range(n)
    ]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_batch(raw: str, label: str, spec: dict) -> list[dict]:
    """Model output (a JSON list of strings, possibly fenced) -> tickets labeled `label`."""
    text = _FENCE.sub("", raw or "").strip()
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        items = json.loads(text[i : j + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    out = []
    for s in items:
        if isinstance(s, str) and s.strip():
            out.append({"text": s.strip(), "label": label, **spec})
    return out


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def dedupe(tickets: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for t in tickets:
        k = _key(t["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def assign_ids(tickets: list[dict]) -> list[dict]:
    return [{**t, "ticket_id": f"R-{i:05d}"} for i, t in enumerate(tickets, start=1)]


def split_tickets(tickets: list[dict], test_size: int = 100, seed: int = 0) -> list[dict]:
    """Stratified by label (largest remainder); deterministic; returns copies with `split`."""
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for i, t in enumerate(tickets):
        by_label.setdefault(t["label"], []).append(i)
    n = len(tickets)
    exact = {lab: test_size * len(idx) / n for lab, idx in by_label.items()}
    quota = {lab: int(v) for lab, v in exact.items()}
    for lab in sorted(exact, key=lambda k: exact[k] - quota[k], reverse=True)[
        : test_size - sum(quota.values())
    ]:
        quota[lab] += 1
    test_idx: set[int] = set()
    for lab in sorted(by_label):
        idx = list(by_label[lab])
        rng.shuffle(idx)
        test_idx.update(idx[: quota[lab]])
    return [{**t, "split": "test" if i in test_idx else "train"} for i, t in enumerate(tickets)]


def extend_tickets(existing: list[dict], new: list[dict]) -> list[dict]:
    """Append new tickets as train-only rows with continuing ids; existing rows and the test split stay."""
    seen = {_key(t["text"]) for t in existing}
    out = list(existing)
    next_id = max((int(t["ticket_id"].split("-")[1]) for t in existing), default=0) + 1
    for t in new:
        k = _key(t["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append({**t, "ticket_id": f"R-{next_id:05d}", "split": "train"})
        next_id += 1
    return out
