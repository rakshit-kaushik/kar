"""scripts/sim_caller.py speaks the caller's lines through Cartesia; when Cartesia fails (402 quota, Nov 1 07:16: both scenario calls
aborted before the first word) or SIM_TTS=deepgram, Deepgram Aura speaks them, cached under the Deepgram voice so the fixed Dana
lines keep their Cartesia audio and earlier blocks stay comparable."""
import importlib.util
import pathlib

import httpx
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("sim_caller_under_test", ROOT / "scripts" / "sim_caller.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeResp:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"{self.status_code}", request=httpx.Request("POST", "http://x"), response=httpx.Response(self.status_code))


PCM = np.full(4800, 1000, dtype=np.int16).tobytes()


@pytest.fixture
def sc(monkeypatch, tmp_path):
    m = load()
    monkeypatch.setattr(m, "CACHE", tmp_path)
    monkeypatch.setenv("CARTESIA_API_KEY", "car-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key")
    monkeypatch.delenv("SIM_TTS", raising=False)
    return m


def test_deepgram_speaks_when_selected(sc, monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_TTS", "deepgram")
    calls = []

    def fake_post(url, **kw):
        calls.append((url, kw))
        return FakeResp(200, PCM)

    monkeypatch.setattr(sc, "_post", fake_post)
    pcm = sc.tts_cached("hello there")
    url, kw = calls[0]
    assert "api.deepgram.com/v1/speak" in url and kw["headers"]["Authorization"] == "Token dg-key"
    assert kw["params"]["encoding"] == "linear16" and kw["params"]["sample_rate"] == sc.SR and kw["params"]["container"] == "none"
    assert kw["params"]["model"] == sc.DEEPGRAM_CALLER_VOICE and kw["json"] == {"text": "hello there"}
    assert len(pcm) > 0 and list((tmp_path / sc.DEEPGRAM_CALLER_VOICE).glob("*.pcm"))


def test_cartesia_failure_falls_back_to_deepgram_and_the_result_is_cached(sc, monkeypatch, tmp_path):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        return FakeResp(402) if "cartesia" in url else FakeResp(200, PCM)

    monkeypatch.setattr(sc, "_post", fake_post)
    pcm = sc.tts_cached("a brand new scenario line")
    assert [("cartesia" in u, "deepgram" in u) for u in calls] == [(True, False), (False, True)]
    assert len(pcm) > 0 and list((tmp_path / sc.DEEPGRAM_CALLER_VOICE).glob("*.pcm")) and not list((tmp_path / sc.CALLER_VOICE).glob("*.pcm"))
    calls.clear()
    sc.tts_cached("a brand new scenario line")
    assert calls == []


def test_cartesia_stays_first_by_default(sc, monkeypatch, tmp_path):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        return FakeResp(200, PCM)

    monkeypatch.setattr(sc, "_post", fake_post)
    sc.tts_cached("the usual line")
    assert len(calls) == 1 and "cartesia" in calls[0] and list((tmp_path / sc.CALLER_VOICE).glob("*.pcm"))
