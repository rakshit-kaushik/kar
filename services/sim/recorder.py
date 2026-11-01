"""Two-channel recording of a simulated call: left = what the caller sent, right = what the agent sent, placed on one
wall-clock timeline (gaps are silence) so a demo scenario's backup recording plays the way the call sounded.
Consecutive frames stay contiguous (timestamps jitter by a few ms); a gap over 50 ms re-anchors on the clock."""
from __future__ import annotations

import pathlib
import wave

import numpy as np

CHANNELS = ("caller", "agent")
REANCHOR_MS = 50


class CallRecorder:
    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self._t0: int | None = None
        self._chunks: dict[str, list[tuple[int, np.ndarray]]] = {c: [] for c in CHANNELS}

    def add(self, who: str, ts_ms: int, pcm: np.ndarray) -> None:
        if self._t0 is None:
            self._t0 = ts_ms
        self._chunks[who].append((ts_ms, np.asarray(pcm, dtype=np.int16).copy()))

    def _place(self, chunks: list[tuple[int, np.ndarray]]) -> list[tuple[int, np.ndarray]]:
        out, cursor = [], 0
        for ts, pcm in chunks:
            clock = round((ts - (self._t0 or 0)) * self.sr / 1000)
            if abs(clock - cursor) > self.sr * REANCHOR_MS // 1000:
                cursor = clock
            out.append((cursor, pcm))
            cursor += len(pcm)
        return out

    def render(self) -> np.ndarray:
        placed = {who: self._place(chunks) for who, chunks in self._chunks.items()}
        total = max((off + len(p) for ch in placed.values() for off, p in ch), default=0)
        out = np.zeros((total, 2), dtype=np.int16)
        for col, who in enumerate(CHANNELS):
            for off, pcm in placed[who]:
                out[off:off + len(pcm), col] = pcm
        return out

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(self.render().tobytes())
