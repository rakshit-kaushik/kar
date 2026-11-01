"""CallerTimeline: the simulated caller's view of a call. It records when the caller stopped speaking,
the agent's state flips (lk.agent.state attribute: listening/thinking/speaking) and the first non-silent
agent audio frame, and turns them into per-turn latencies as the caller hears them."""
from services.sim.timeline import CallerTimeline


def test_greeting_before_any_caller_turn_creates_no_turn_and_marks_agent_listening():
    tl = CallerTimeline()
    assert tl.agent_is_listening() is False
    tl.agent_state(500, "speaking")
    tl.agent_audio(600, rms=900)
    tl.agent_state(2500, "listening")
    assert tl.turns == [] and tl.agent_is_listening() is True


def test_one_turn_measures_eos_to_thinking_speaking_and_first_audio():
    tl = CallerTimeline()
    tl.agent_state(0, "listening")
    tl.caller_speech_end(1000)
    assert tl.agent_is_listening() is False  # we are waiting for the reply now
    tl.agent_audio(1100, rms=10)  # silence frames do not count
    tl.agent_state(1350, "thinking")
    tl.agent_state(1900, "speaking")
    tl.agent_audio(1950, rms=800)
    tl.agent_audio(2000, rms=900)  # later frames do not move first audio
    tl.agent_state(3200, "listening")
    [t] = tl.latencies()
    assert t == {"turn": 1, "eos_to_thinking_ms": 350, "eos_to_speaking_ms": 900, "eos_to_first_audio_ms": 950, "agent_speech_ms": 1300}
    assert tl.agent_is_listening() is True


def test_summary_gives_p50_p95_over_turns_and_skips_turns_without_audio():
    tl = CallerTimeline()
    for i, (eos, audio) in enumerate([(1000, 1800), (5000, 6200), (9000, 9700), (13000, None)]):
        tl.agent_state(eos - 10, "listening")
        tl.caller_speech_end(eos)
        if audio is not None:
            tl.agent_state(audio - 50, "speaking")
            tl.agent_audio(audio, rms=1000)
            tl.agent_state(audio + 1000, "listening")
    s = tl.summary()
    assert s["turns"] == 4 and s["turns_with_audio"] == 3
    assert s["eos_to_first_audio_ms"]["p50"] == 800 and s["eos_to_first_audio_ms"]["p95"] == 1160
    assert s["eos_to_first_audio_ms"]["max"] == 1200
