"""scripts/g1_report.py pure helpers (the script itself needs Postgres)."""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("g1_report", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "g1_report.py")
g1_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1_report)


def test_decode_rate_is_tokens_after_the_first_over_the_streaming_window():
    assert round(g1_report.decode_rate(completion_tokens=79, first_ms=400, done_ms=1400), 1) == 78.0


def test_decode_rate_is_none_for_a_tool_call_turn_whose_first_token_is_the_whole_turn():
    # vllm-mlx emits a tool call only when complete: first token and done are a few ms apart, the window measures nothing
    assert g1_report.decode_rate(completion_tokens=46, first_ms=1078, done_ms=1081) is None


def test_decode_rate_is_none_without_enough_tokens():
    assert g1_report.decode_rate(completion_tokens=1, first_ms=100, done_ms=900) is None


def test_turn_kind_splits_tool_call_turns_from_spoken_turns():
    assert g1_report.turn_kind({"tool_calls": 1, "completion_tokens": 46}) == "tool"
    assert g1_report.turn_kind({"tool_calls": 0, "completion_tokens": 53}) == "spoken"
    assert g1_report.turn_kind({}) == "spoken"


def _ev(call, event, ts, payload):
    return (call, 0, "m", event, ts, payload)


def test_success_claims_without_a_tool_result_are_counted():
    # call 2 on the candidate (Nov 1 04:33): "I booked your mortgage consultation" with no book_appointment tool_end before it
    evs = [_ev("c", "transcript", 1, {"role": "assistant", "text": "To confirm: ... Shall I book it?"}),
           _ev("c", "transcript", 2, {"role": "assistant", "text": "I booked your mortgage consultation at the Market Street branch."})]
    assert g1_report.unsupported_claims(evs) == 1
    ok = [_ev("c", "tool_end", 1, {"name": "book_appointment", "ok": True}),
          _ev("c", "transcript", 2, {"role": "assistant", "text": "I've booked your mortgage consultation."})]
    assert g1_report.unsupported_claims(ok) == 0
    failed_tool = [_ev("c", "tool_end", 1, {"name": "book_appointment", "ok": False}),
                   _ev("c", "transcript", 2, {"role": "assistant", "text": "Your appointment is booked."})]
    assert g1_report.unsupported_claims(failed_tool) == 1


def test_filler_timing_splits_filler_audio_from_the_reply():
    # a turn anchored at 0: filler spoken at 900 ms (audible 1150), the real reply audible at 1900
    evs = [_ev("c", "filler_start", 900, {"tool": "lookup_account", "text": "One moment."}),
           _ev("c", "tts_first_audible", 1150, {}), _ev("c", "tts_first_audible", 1900, {})]
    assert g1_report.filler_timing(evs, 0) == (1150, 1900)
    assert g1_report.filler_timing([_ev("c", "tts_first_audible", 700, {})], 0) == (None, None)  # no filler: nothing to split
    assert g1_report.filler_timing([_ev("c", "filler_start", 900, {})], 0) == (None, None)  # filler never became audible


def test_loop_guard_interventions_counts_the_agent_events():
    evs = [_ev("c", "loop_guard", 1, {"action": "rephrase", "repeats": 1}), _ev("c", "transcript", 2, {"role": "assistant", "text": "x"}),
           _ev("c", "loop_guard", 3, {"action": "handoff", "repeats": 2})]
    assert g1_report.loop_guard_interventions(evs) == 2
    assert g1_report.loop_guard_interventions([]) == 0



def test_claim_guard_interventions_counts_the_agent_events_apart_from_the_loop_guard():
    evs = [_ev("c", "claim_guard", 1, {"claimed": "I'm transferring you", "reason": "suspected_fraud"}), _ev("c", "loop_guard", 2, {"action": "rephrase", "repeats": 1})]
    assert g1_report.claim_guard_interventions(evs) == 1
    assert g1_report.loop_guard_interventions(evs) == 1
    assert g1_report.claim_guard_interventions([]) == 0
