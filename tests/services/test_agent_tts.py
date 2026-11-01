"""TTS vendor switch (Nov 1 04:24: Cartesia returned 402 quota_exceeded, every greeting failed; Deepgram Aura-2 works with the STT key)."""
from services.agent.tts import make_tts, tts_vendor


def test_default_is_cartesia(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "x"); monkeypatch.setenv("CARTESIA_VOICE", "v"); monkeypatch.delenv("AGENT_TTS", raising=False)
    from livekit.plugins import cartesia
    assert tts_vendor() == "cartesia" and isinstance(make_tts(), cartesia.TTS)


def test_deepgram_fallback_selected_by_env(monkeypatch):
    monkeypatch.setenv("AGENT_TTS", "deepgram"); monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
    from livekit.plugins import deepgram
    assert tts_vendor() == "deepgram" and isinstance(make_tts(), deepgram.TTS)
