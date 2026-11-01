"""Normalize a chat-completion SSE stream into strict OpenAI delta semantics.

Why this exists (see CONTRACT.md §7 and the G1 smoke test): vllm-mlx 0.5.0 with the qwen3 parser
emits each complete tool call twice (mid-stream and again in the finish chunk, different ids), and a
model can leak `<think>` or raw `<tool_call>` text into content. Consumers (LiveKit's OpenAI plugin)
concatenate deltas, so they must never see any of that.

Stateful, per request. `feed(chunk) -> [chunk]`, `finish() -> [chunk]`. Chunks are plain dicts.
"""
from __future__ import annotations

import copy
import json
import uuid

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


def _held_suffix(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
    best = 0
    for tag in tags:
        for k in range(min(len(text), len(tag) - 1), 0, -1):
            if tag.startswith(text[-k:]):
                best = max(best, k)
                break
    return best


class StreamTransformer:
    def __init__(self) -> None:
        self._buf = ""
        self._mode = "text"  # text | think | tool
        self._lstrip_next = False
        self._calls: dict[int, dict[str, str]] = {}
        self._next_index = 0
        self._converted = False
        self._envelope: dict | None = None
        self.stats = {"tool_calls": 0, "dropped_duplicate_tool_calls": 0, "stripped_think_chars": 0, "converted_raw_tool_calls": 0, "bad_raw_tool_calls": 0}

    # -- public -------------------------------------------------------------------------------------
    def feed(self, chunk: dict) -> list[dict]:
        choices = chunk.get("choices") or []
        if not choices:
            return [chunk]
        self._envelope = chunk
        choice = choices[0]
        delta = dict(choice.get("delta") or {})
        delta.pop("reasoning_content", None)

        tool_calls = [tc for tc in (delta.pop("tool_calls", None) or []) if not self._is_duplicate(tc)]
        content = delta.pop("content", None)
        pieces: list[dict] = []
        if content:
            pieces += self._consume_text(content)
        if tool_calls:
            pieces.append({"tool_calls": tool_calls})

        finish = choice.get("finish_reason")
        if finish == "stop" and self._converted:
            finish = "tool_calls"
        carries_meta = bool(delta) or finish is not None or "usage" in chunk
        if not pieces and not carries_meta:
            return []
        if not pieces:
            pieces = [{}]
        out = []
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            d = {**delta, **piece} if last else piece
            out.append(self._chunk(chunk, d, finish if last else None, include_usage=last))
        return out

    def finish(self) -> list[dict]:
        if self._envelope is None or not self._buf:
            return []
        pieces: list[dict] = []
        if self._mode == "text":
            pieces += self._emit_text(self._buf)
        elif self._mode == "tool":
            pieces += self._convert_raw(self._buf)
        else:
            self.stats["stripped_think_chars"] += len(self._buf)
        self._buf = ""
        return [self._chunk(self._envelope, p, None, include_usage=False) for p in pieces]

    # -- tool-call dedupe -------------------------------------------------------------------------
    def _is_duplicate(self, tc: dict) -> bool:
        idx = tc.get("index", 0)
        fn = tc.get("function") or {}
        name, args = fn.get("name") or "", fn.get("arguments") or ""
        seen = self._calls.get(idx)
        if seen is not None and name and args and seen["name"] == name and seen["args"] == args:
            self.stats["dropped_duplicate_tool_calls"] += 1
            return True
        if seen is None:
            self._calls[idx] = {"name": "", "args": ""}
            seen = self._calls[idx]
        if name:
            self.stats["tool_calls"] += 1
        seen["name"] += name
        seen["args"] += args
        return False

    # -- content state machine --------------------------------------------------------------------
    def _consume_text(self, text: str) -> list[dict]:
        self._buf += text
        pieces: list[dict] = []
        while True:
            if self._mode == "text":
                p_think, p_tool = self._buf.find(THINK_OPEN), self._buf.find(TOOL_OPEN)
                hits = [(p, t) for p, t in ((p_think, THINK_OPEN), (p_tool, TOOL_OPEN)) if p >= 0]
                if hits:
                    p, tag = min(hits)
                    pieces += self._emit_text(self._buf[:p])
                    self._buf = self._buf[p + len(tag):]
                    self._mode = "think" if tag == THINK_OPEN else "tool"
                    continue
                k = _held_suffix(self._buf, (THINK_OPEN, TOOL_OPEN))
                pieces += self._emit_text(self._buf[: len(self._buf) - k])
                self._buf = self._buf[len(self._buf) - k:] if k else ""
                return pieces
            if self._mode == "think":
                p = self._buf.find(THINK_CLOSE)
                if p >= 0:
                    self.stats["stripped_think_chars"] += p
                    self._buf = self._buf[p + len(THINK_CLOSE):]
                    self._mode = "text"
                    self._lstrip_next = True
                    continue
                k = _held_suffix(self._buf, (THINK_CLOSE,))
                self.stats["stripped_think_chars"] += len(self._buf) - k
                self._buf = self._buf[len(self._buf) - k:] if k else ""
                return pieces
            # tool
            p = self._buf.find(TOOL_CLOSE)
            if p < 0:
                return pieces
            pieces += self._convert_raw(self._buf[:p])
            self._buf = self._buf[p + len(TOOL_CLOSE):]
            self._mode = "text"
            self._lstrip_next = True

    def _emit_text(self, text: str) -> list[dict]:
        if self._lstrip_next:
            text = text.lstrip()
            if not text:
                return []
            self._lstrip_next = False
        return [{"content": text}] if text else []

    def _convert_raw(self, raw: str) -> list[dict]:
        try:
            obj = json.loads(raw.strip())
            name = obj["name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self.stats["bad_raw_tool_calls"] += 1
            return []
        args = obj.get("arguments", {})
        args_s = args if isinstance(args, str) else json.dumps(args)
        idx = self._next_index
        self._next_index += 1
        self._calls[idx] = {"name": name, "args": args_s}
        self._converted = True
        self.stats["tool_calls"] += 1
        self.stats["converted_raw_tool_calls"] += 1
        return [{"tool_calls": [{"index": idx, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": name, "arguments": args_s}}]}]

    # -- envelope -----------------------------------------------------------------------------------
    @staticmethod
    def _chunk(template: dict, delta: dict, finish, include_usage: bool) -> dict:
        c = copy.deepcopy(template)
        c["choices"] = [dict(c["choices"][0])]
        c["choices"][0]["delta"] = delta
        c["choices"][0]["finish_reason"] = finish
        if not include_usage:
            c.pop("usage", None)
        return c
