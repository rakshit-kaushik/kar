"""Pick the training targets of a scored episode and emit mlx_lm `messages` + `tools` examples.

A target is one assistant turn of the episode; the example is the conversation up to and
including that turn (the prefix is masked by the trainer, the loss falls on the target). At most
three targets per episode, chosen to cover a tool call (the handoff decision beats a write; a write
beats the latest read; the lookup only when nothing else was called), a clarification (a question or
read-back; when the episode has both a handoff and a write, the write takes this slot instead), and
the closing line. Three per-target filters, applied to
the target alone so the episode's other targets survive: the contract's brevity rule (at most two
sentences and 35 spoken words, CONTRACT §2 rule 4), an optional `fits` callback (the 2,048-token
check, see data/format/length.py), and for clarifications the re-ask rule (a question about a branch,
day, time or service the caller had already given earlier in the prefix teaches over-asking; see
eval.scorers.scorers.reask_slots). Targets are dropped, never truncated.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from eval.sandbox import TOOL_SCHEMAS
from eval.scorers.normalize import WRITE_TOOLS
from eval.scorers.scorers import reask_slots, slot_truth
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
    reason: str | None = None  # set when dropped: "brevity" | "length" | "reask"


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


def _expected_slots(ep: Episode) -> dict[str, str]:
    """Branch / service / date / time the scenario expects to be booked (the caller's truth)."""
    sc = ep.scenario
    if isinstance(sc, dict):
        steps = (sc.get("expect") or {}).get("tool_calls") or []
    else:
        steps = [{"tool": st.tool, "args": st.args} for st in sc.expect.tool_calls]
    return slot_truth(steps)


def _caller_texts_before(ep: Episode, message_idx: int) -> list[str]:
    return [
        m.get("content") or ""
        for m in ep.messages[:message_idx]
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]


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

    # order candidates per kind: the handoff decision beats everything (CONTRACT §1 scores it by
    # reason, and it is rare), then the first write tool call beats the lookup; the clarification
    # is preferably the last question before the first write (the read-back), after verification
    def _is_handoff(t: Target) -> bool:
        return "request_handoff" in t.tool_names

    def _is_write(t: Target) -> bool:
        return any(n in WRITE_TOOLS for n in t.tool_names)

    def _is_lookup_only(t: Target) -> bool:
        return t.tool_names == ["lookup_account"]

    # Fallback order when the write overflows the length check: the LATEST read (check_availability
    # carries the branch, service and date the scorers check), never the lookup. lookup_account is a
    # target only when it is the episode's sole tool call (verification-failed flows); v2 otherwise
    # had 706 lookup targets out of 1,252 tool calls.
    tool_cands = by_kind["tool_call"]
    if len(tool_cands) > 1:
        tool_cands = [t for t in tool_cands if not _is_lookup_only(t)]
    tool_order = sorted(
        tool_cands,
        key=lambda t: (
            not _is_handoff(t),
            not _is_write(t),
            t.message_idx if _is_write(t) else -t.message_idx,
        ),
    )
    lookup_idx = next(
        (t.message_idx for t in by_kind["tool_call"] if "lookup_account" in t.tool_names), -1
    )
    write_idx = next((t.message_idx for t in tool_order if _is_write(t)), None)
    after = [t for t in by_kind["clarification"] if t.message_idx > lookup_idx]
    before_write = [t for t in after if write_idx is None or t.message_idx < write_idx]
    rest = [t for t in by_kind["clarification"] if t not in before_write]
    clar_order = sorted(before_write, key=lambda t: -t.message_idx) + rest
    # when the episode has both a handoff and a write (lost card: ticket, then transfer), the write
    # takes the second slot instead of the read-back, so the ticket category stays supervised
    second_slot = clar_order
    if tool_order and _is_handoff(tool_order[0]) and write_idx is not None:
        second_slot = [t for t in tool_order[1:] if _is_write(t)] + clar_order
    slots: list[list[Target]] = [tool_order, second_slot, by_kind["closing"]]
    truth = _expected_slots(ep)
    chosen: list[Target] = []
    dropped: list[Target] = []
    for cands in slots:
        for t in cands:
            if len(chosen) >= max_targets:
                break
            ok, _, _ = brevity_ok(t.text) if t.text else (True, 0, 0)
            if not ok:
                t.reason = "brevity"
                dropped.append(t)
                continue
            if (
                t.kind == "clarification"
                and truth
                and reask_slots(truth, _caller_texts_before(ep, t.message_idx), t.text)
            ):
                t.reason = "reask"
                dropped.append(t)
                continue
            if fits is not None and not fits(to_example(ep, t)):
                t.reason = "length"
                dropped.append(t)
                continue
            chosen.append(t)
            break
    return chosen, dropped
