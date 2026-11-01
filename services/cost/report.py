"""Cost per successfully completed task, from measured usage (turn_events) and cost_rates.

Per call: Deepgram streaming minutes, Cartesia characters (assistant transcript text), LiveKit agent-session and
participant minutes (caller + agent), LLM tokens by model. Paths: `local` (base or adapter on the Mac: machine
hourly cost / calls per hour at the admission limit) and `hosted` (tokens at list price). Measured quantities
and estimated bases are labelled apart, as the plan requires.
"""
from __future__ import annotations

from collections import defaultdict

HOSTED_MODEL = "claude-sonnet-5-5"


def call_usage(events: list[tuple]) -> dict:
    """events: rows (call_id, turn_idx, model_id, event, ts_ms, payload) of one call, in time order."""
    ts = [e[4] for e in events]
    start = next((e[4] for e in events if e[3] == "call_start"), min(ts))
    end = next((e[4] for e in events if e[3] == "call_end"), max(ts))
    duration_s = next((e[5].get("duration_s") for e in events if e[3] == "call_end" and e[5].get("duration_s") is not None), None)
    duration_min = (float(duration_s) / 60.0) if duration_s is not None else (end - start) / 60000.0
    # Deepgram bills the audio it received (stt_metrics.audio_duration, seconds); without those metrics assume the whole call
    stt_s = sum(float(e[5].get("audio_duration") or 0) for e in events if e[3] == "stt_metrics")
    stt_audio_min = stt_s / 60.0 if stt_s > 0 else duration_min
    # Cartesia bills characters; its own count rides on tts_first_audio, the transcript length is the fallback
    tts_chars = [int(e[5].get("characters") or 0) for e in events if e[3] == "tts_first_audio" and e[5].get("characters") is not None]
    chars = sum(tts_chars) if tts_chars else sum(len(e[5].get("text") or "") for e in events if e[3] == "transcript" and e[5].get("role") == "assistant")
    tokens: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0})
    for e in events:
        if e[3] == "llm_done":
            tokens[e[2]]["prompt"] += int(e[5].get("prompt_tokens") or 0)
            tokens[e[2]]["completion"] += int(e[5].get("completion_tokens") or 0)
    outcome = next((e[5].get("outcome") for e in events if e[3] == "call_end"), None)
    booked = any(e[3] == "tool_end" and e[5].get("name") in ("book_appointment", "reschedule_appointment", "cancel_appointment", "create_ticket") and e[5].get("ok") for e in events)
    success = outcome == "completed" if outcome in ("completed", "abandoned", "failed", "handoff") else booked
    return {"duration_min": duration_min, "stt_audio_min": stt_audio_min, "agent_chars": chars, "tokens": {k: dict(v) for k, v in tokens.items()}, "success": bool(success), "outcome": outcome}


def mac_hourly_usd(rates: dict, workday: bool = False) -> float:
    hours = rates["mac.workday_amortization_hours"] if workday else rates["mac.amortization_hours"]
    return rates["mac.purchase_usd"] / hours + rates["mac.power_kw"] * rates["electricity_usd_per_kwh"]


def price_call(u: dict, rates: dict, path: str, avg_call_min: float, workday: bool = False) -> dict:
    m = u["duration_min"]
    audio = u.get("stt_audio_min", m) * rates["deepgram.nova3.streaming_usd_per_min"] \
        + m * (rates["livekit.agent_session_usd_per_min"] + 2 * rates["livekit.webrtc_participant_usd_per_min"]) \
        + u["agent_chars"] * rates["cartesia.sonic.usd_per_char"]
    if path == "hosted":
        t = u["tokens"].get(HOSTED_MODEL, {"prompt": 0, "completion": 0})
        llm = t["prompt"] / 1e6 * rates[f"anthropic.{HOSTED_MODEL}.input_usd_per_mtok"] + t["completion"] / 1e6 * rates[f"anthropic.{HOSTED_MODEL}.output_usd_per_mtok"]
        basis = "measured tokens at list price"
    else:
        adm = int(rates["admission.concurrent_calls"])
        calls_per_hour = adm * 60.0 / max(avg_call_min, 1e-6)
        llm = mac_hourly_usd(rates, workday) / calls_per_hour
        basis = f"estimated: Mac hourly cost / calls per hour at admission limit {adm}"
    return {"audio_usd": audio, "llm_usd": llm, "total_usd": audio + llm, "llm_basis": basis, "audio_basis": "measured minutes and characters at list price"}
