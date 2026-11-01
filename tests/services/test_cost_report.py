"""Cost per successfully completed task (plan §Demo, economics). Per call: Deepgram minutes, Cartesia characters,
LiveKit minutes (caller + agent), LLM tokens by model; the local path is the Mac's hourly cost divided by calls
per hour at the admission limit; the hosted path is list-price tokens. Measured and estimated are kept apart."""
from services.cost.report import call_usage, mac_hourly_usd, price_call

RATES = {
    "deepgram.nova3.streaming_usd_per_min": 0.0077, "cartesia.sonic.usd_per_char": 0.00004,
    "livekit.agent_session_usd_per_min": 0.01, "livekit.webrtc_participant_usd_per_min": 0.0005,
    "anthropic.claude-sonnet-5-5.input_usd_per_mtok": 2.0, "anthropic.claude-sonnet-5-5.output_usd_per_mtok": 10.0,
    "mac.purchase_usd": 4099, "mac.amortization_hours": 26280, "mac.power_kw": 0.1, "electricity_usd_per_kwh": 0.15,
    "admission.concurrent_calls": 4,
}


def ev(event, ts_ms, model="r3-voice-base", **payload):
    return ("c1", 0, model, event, ts_ms, payload)


def test_call_usage_reads_the_agents_real_events():
    """Deepgram bills audio seconds (stt_metrics.audio_duration); Cartesia bills characters (tts_first_audio.characters);
    LiveKit bills the session minutes (call_end.duration_s); the LLM bills tokens per model (llm_done)."""
    events = [
        ev("call_start", 0), ev("stt_metrics", 900, audio_duration=30.0),
        ev("tts_first_audio", 1000, characters=12), ev("transcript", 1000, role="assistant", text="Hello there."),
        ev("llm_done", 2000, prompt_tokens=1500, completion_tokens=40),
        ev("llm_done", 3000, model="claude-sonnet-5-5", prompt_tokens=2000, completion_tokens=60),
        ev("stt_metrics", 4500, audio_duration=15.0),
        ev("transcript", 4000, role="user", text="ignored"), ev("tts_first_audio", 5000, characters=5), ev("transcript", 5000, role="assistant", text="Bye."),
        ev("tool_end", 5500, name="book_appointment", ok=True), ev("call_end", 130000, outcome="completed", duration_s=120.0),
    ]
    u = call_usage(events)
    assert u["duration_min"] == 2.0            # call_end.duration_s wins over the event span
    assert u["stt_audio_min"] == 0.75          # 45 s of audio sent to Deepgram
    assert u["agent_chars"] == 17              # Cartesia's own character count, not the transcript length
    assert u["success"] is True
    assert u["tokens"] == {"r3-voice-base": {"prompt": 1500, "completion": 40}, "claude-sonnet-5-5": {"prompt": 2000, "completion": 60}}


def test_call_usage_falls_back_to_transcripts_and_event_span_without_vendor_metrics():
    events = [ev("call_start", 0), ev("transcript", 1000, role="assistant", text="Hello there."), ev("transcript", 5000, role="assistant", text="Bye."), ev("call_end", 120000, outcome="abandoned")]
    u = call_usage(events)
    assert u["duration_min"] == 2.0 and u["stt_audio_min"] == 2.0 and u["agent_chars"] == 16 and u["success"] is False


def test_mac_hourly_is_amortization_plus_power():
    assert round(mac_hourly_usd(RATES), 4) == round(4099 / 26280 + 0.1 * 0.15, 4)


def test_price_call_local_path_divides_machine_hour_by_calls_per_hour_at_admission():
    u = {"duration_min": 2.0, "stt_audio_min": 1.5, "agent_chars": 1000, "success": True, "tokens": {"r3-voice-base": {"prompt": 10000, "completion": 300}}}
    c = price_call(u, RATES, path="local", avg_call_min=2.0)
    audio = 1.5 * 0.0077 + 2.0 * (0.01 + 2 * 0.0005) + 1000 * 0.00004
    calls_per_hour = 4 * 60 / 2.0
    assert round(c["audio_usd"], 6) == round(audio, 6)
    assert round(c["llm_usd"], 6) == round(mac_hourly_usd(RATES) / calls_per_hour, 6)
    assert c["llm_basis"] == "estimated: Mac hourly cost / calls per hour at admission limit 4"


def test_price_call_hosted_path_uses_list_token_prices():
    u = {"duration_min": 1.0, "stt_audio_min": 1.0, "agent_chars": 0, "success": False, "tokens": {"claude-sonnet-5-5": {"prompt": 1_000_000, "completion": 100_000}}}
    c = price_call(u, RATES, path="hosted", avg_call_min=1.0)
    assert round(c["llm_usd"], 6) == round(2.0 + 1.0, 6) and c["llm_basis"] == "measured tokens at list price"


def test_local_llm_seconds_from_measured_throughput():
    """Text-eval runs on the Mac have no wall time in the ledger: estimate GPU seconds from measured prefill and decode rates."""
    from services.cost.report import local_llm_seconds
    rates = {**RATES, "mac.prefill_tok_s": 4700.0, "mac.decode_tok_s": 78.0}
    assert round(local_llm_seconds(prompt_tokens=470_000, completion_tokens=7_800, rates=rates), 1) == 100.0 + 100.0
