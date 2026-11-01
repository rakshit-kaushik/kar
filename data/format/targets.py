"""Pick the training targets of a scored episode and emit mlx_lm `messages` + `tools` examples.

A target is one assistant turn of the episode; the example is the conversation up to and
including that turn (the prefix is masked by the trainer, the loss falls on the target). At most
three targets per episode, chosen to cover a tool call (a write beats the lookup), a
clarification (a question or read-back), and the closing line. Two per-target filters, applied to
the target alone so the episode's other targets survive: the contract's brevity rule (at most two
sentences and 35 spoken words, CONTRACT §2 rule 4) and an optional `fits` callback (the 2,048-token
check, see data/format/length.py). Targets are dropped, never truncated.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from eval.sandbox import TOOL_SCHEMAS
from eval.scorers.normalize import WRITE_TOOLS
from eval.simulator.episode import Episode

MAX_WORDS = 35
MAX_SENTENCES = 2
KINDS = ("tool_call", "clarification", "closing", "other")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> int:
    return len([s for s in _SENT.split(text.strip()) if s.strip()]) if text.strip() else 0


def brevity_ok(text: str) -> tuple[bool, int, int]:
    """(ok, words, sentences) for one spoken reply."""
    words = len(text.split())
    sentences = _sentences(text)
    return (words <= MAX_WORDS and sentences <= MAX_SENTENCES, words, sentences)


@dataclass
class Target:
    message_idx: int
    turn: int
    kind: str
    text: str
    tool_names: list[str]
    words: int = 0
    sentences: int = 0
    reason: str | None = None  # set when dropped: "brevity" | "length"


def _message_turn(ep: Episode) -> dict[int, int]:
    out = {}
    for t in ep.turns:
        for i in t.get("message_indices", []):
            out[i] = t["idx"]
    return out


def classify_targets(ep: Episode) -> list[Target]:
    """Every assistant message after the greeting, with its kind."""
    turns = _message_turn(ep)
    candidates = [
        (i, m) for i, m in enumerate(ep.messages) if i > 1 and m.get("role") == "assistant"
    ]
    last_text_idx = max(
        (i for i, m in candidates if (m.get("content") or "").strip()), default=None
    )
    out: list[Target] = []
    for i, m in candidates:
        text = (m.get("content") or "").strip()
        tool_names = [
            (c.get("function") or {}).get("name") or "?" for c in (m.get("tool_calls") or [])
        ]
        if tool_names:
            kind = "tool_call"
        elif i == last_text_idx:
            kind = "closing"
        elif text.endswith("?"):
            kind = "clarification"
        else:
            kind = "other"
        _ok, words, sentences = brevity_ok(text)
        out.append(Target(i, turns.get(i, 0), kind, text, tool_names, words, sentences))
    return out


def to_example(ep: Episode, target: Target) -> dict:
    """mlx_lm chat example: prefix + target as the LAST message, the contract's tools, provenance."""
    messages = []
    for m in ep.messages[: target.message_idx + 1]:
        mm = {
            "role": m["role"],
            "content": m.get("content") if isinstance(m.get("content"), str) else "",
        }
        if m.get("tool_calls"):
            mm["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id") is not None:
            mm["tool_call_id"] = m["tool_call_id"]
        messages.append(mm)
    return {
        "messages": messages,
        "tools": list(TOOL_SCHEMAS.values()),
        "meta": {
            "episode_id": ep.episode_id,
            "scenario_id": ep.scenario_id,
            "family": ep.family,
            "kind": target.kind,
            "turn": target.turn,
            "message_idx": target.message_idx,
            "repeat_seed": ep.repeat_seed,
        },
    }


def _pick_tool_call(cands: list[Target]) -> Target | None:
    writes = [t for t in cands if any(n in WRITE_TOOLS for n in t.tool_names)]
    if writes:
        return writes[0]
    return cands[0] if cands else None


def select_targets(
    ep: Episode, fits: Callable[[dict], bool] | None = None, max_targets: int = 3
) -> tuple[list[Target], list[Target]]:
    """-> (chosen, dropped). Chosen covers tool_call, clarification, closing when available; a
    target failing brevity or `fits` is dropped with a reason and the next candidate of that kind
    is tried."""
    all_targets = classify_targets(ep)
    by_kind: dict[str, list[Target]] = {k: [] for k in KINDS}
    for t in all_targets:
        by_kind[t.kind].append(t)
    # order candidates per kind: the first write tool call beats the lookup; the clarification is
    # preferably the last question before the first write (the read-back), after verification
    tool_order = sorted(
        by_kind["tool_call"],
        key=lambda t: (not any(n in WRITE_TOOLS for n in t.tool_names), t.message_idx),
    )
    lookup_idx = next(
        (t.message_idx for t in by_kind["tool_call"] if "lookup_account" in t.tool_names), -1
    )
    write_idx = next(
        (t.message_idx for t in tool_order if any(n in WRITE_TOOLS for n in t.tool_names)), None
    )
    after = [t for t in by_kind["clarification"] if t.message_idx > lookup_idx]
    before_write = [t for t in after if write_idx is None or t.message_idx < write_idx]
    rest = [t for t in by_kind["clarification"] if t not in before_write]
    clar_order = sorted(before_write, key=lambda t: -t.message_idx) + rest
    ordered: dict[str, list[Target]] = {
        "tool_call": tool_order,
        "clarification": clar_order,
        "closing": by_kind["closing"],
    }
    chosen: list[Target] = []
    dropped: list[Target] = []
    for kind in ("tool_call", "clarification", "closing"):
        for t in ordered[kind]:
            if len(chosen) >= max_targets:
                break
            ok, _, _ = brevity_ok(t.text) if t.text else (True, 0, 0)
            if not ok:
                t.reason = "brevity"
                dropped.append(t)
                continue
            if fits is not None and not fits(to_example(ep, t)):
                t.reason = "length"
                dropped.append(t)
                continue
            chosen.append(t)
            break
    return chosen, dropped
