"""Proof of served weights (Nov 1 incident: :8000 served run 1's directory under run 2's name for two hours)."""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("serving_proof", ROOT / "scripts" / "serving_proof.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def test_model_dir_from_vllm_mlx_launcher_cmdline():
    cmd = (".venv/bin/python scripts/vllm_mlx_serve.py serve /Users/abgup/r3-artifacts/models/r3-voice-206506525c --host 127.0.0.1 "
           "--port 8000 --served-model-name r3-voice-730f8ba68e --prefix-trie-cache")
    assert sp.model_dir_from_cmdline(cmd) == "/Users/abgup/r3-artifacts/models/r3-voice-206506525c"


def test_model_dir_from_plain_vllm_mlx_and_mlx_lm_server():
    assert sp.model_dir_from_cmdline("vllm-mlx serve /m/r3-voice-730f8ba68e --port 8010") == "/m/r3-voice-730f8ba68e"
    assert sp.model_dir_from_cmdline("python -m mlx_lm.server --model /m/Qwen3-4B-8bit --adapter-path /m/a --port 8010") == "/m/Qwen3-4B-8bit"
    assert sp.model_dir_from_cmdline("python something_else.py") is None


def test_weekday_reask_is_the_run1_signature():
    assert sp.reasks_weekday("Which day would you like the mortgage consultation?", []) is True
    assert sp.reasks_weekday("What day works best for you?", []) is True
    assert sp.reasks_weekday("What time on Tuesday morning works for you?", []) is False
    assert sp.reasks_weekday("", [{"name": "check_availability", "arguments": {"branch_id": "market_st"}}]) is False


def test_fingerprint_prefix_is_the_live_call_with_a_real_lookup():
    msgs = sp.fingerprint_messages()
    assert msgs[0]["role"] == "system" and "{today}" not in msgs[0]["content"]
    call = next(m for m in msgs if m.get("tool_calls"))
    tool = msgs[msgs.index(call) + 1]
    assert tool["role"] == "tool" and tool["tool_call_id"] == call["tool_calls"][0]["id"]
    assert json.loads(tool["content"])["verified"] is True  # the sandbox's own answer for Dana, not a hand-written one
    assert msgs[-1] == {"role": "user", "content": "I'd like to book a mortgage consultation Tuesday morning at the Market Street branch."}


def test_reply_parsing_handles_parsed_and_raw_tool_calls():
    parsed = {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "check_availability", "arguments": "{\"day\": \"2026-11-03\"}"}}]}}]}
    assert sp.parse_reply(parsed) == ("", [{"name": "check_availability", "arguments": {"day": "2026-11-03"}}])
    raw = {"choices": [{"message": {"content": "<tool_call>\n{\"name\": \"check_availability\", \"arguments\": {\"day\": \"2026-11-03\"}}\n</tool_call>"}}]}
    assert sp.parse_reply(raw) == ("", [{"name": "check_availability", "arguments": {"day": "2026-11-03"}}])
    text = {"choices": [{"message": {"content": "Which day would you like?"}}]}
    assert sp.parse_reply(text) == ("Which day would you like?", [])


def test_expected_name_follows_the_directory():
    assert sp.expected_name("/Users/abgup/r3-artifacts/models/r3-voice-730f8ba68e") == "r3-voice-730f8ba68e"
    assert sp.expected_name("/Users/abgup/r3-artifacts/models/Qwen3-4B-8bit/") == "r3-voice-base"
