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
