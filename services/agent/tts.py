"""TTS vendor selection. AGENT_TTS=cartesia (default; CARTESIA_MODEL/CARTESIA_VOICE) or deepgram (Aura-2, same key as STT).
Nov 1 04:24: Cartesia answered 402 quota_exceeded (2 credits left) and every greeting failed with "no audio frames were
pushed"; Deepgram TTS worked on the first try. The vendor is written into every call's events (tts_vendor) for the cost ledger."""
from __future__ import annotations

import os


def tts_vendor() -> str:
    return os.environ.get("AGENT_TTS", "cartesia").strip().lower()


def make_tts():
    vendor = tts_vendor()
    if vendor == "deepgram":
        from livekit.plugins import deepgram
        return deepgram.TTS(model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"), sample_rate=24000)
    if vendor == "cartesia":
        from livekit.plugins import cartesia
        return cartesia.TTS(model=os.environ.get("CARTESIA_MODEL", "sonic-3"), voice=os.environ["CARTESIA_VOICE"])
    raise ValueError(f"AGENT_TTS={vendor!r}: expected cartesia or deepgram")
