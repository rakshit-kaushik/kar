"""Stream transformer for the proxy on 8001: normalizes vllm-mlx chat-completion SSE deltas
into strict OpenAI semantics before they reach the LiveKit plugin."""
import json

from services.proxy.transform import StreamTransformer


def chunk(delta=None, finish=None, usage=None):
    c = {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


def tc(index, id_, name, arguments):
    return {"index": index, "id": id_, "type": "function", "function": {"name": name, "arguments": arguments}}


ARGS = json.dumps({"phone": "5550142219", "date_of_birth": "1988-03-12"})


def feed_all(chunks):
    t = StreamTransformer()
    out = []
    for c in chunks:
        out += t.feed(c)
    out += t.finish()
    return t, out


def deltas(out):
    return [c["choices"][0]["delta"] for c in out if c.get("choices")]


def test_duplicate_full_tool_call_repeated_in_finish_chunk_is_dropped():
    _, out = feed_all([
        chunk({"role": "assistant"}),
        chunk({"tool_calls": [tc(0, "call_a", "lookup_account", ARGS)]}),
        chunk({"tool_calls": [tc(0, "call_b", "lookup_account", ARGS)]}, finish="tool_calls", usage={"prompt_tokens": 2023, "completion_tokens": 46}),
    ])
    tool_deltas = [d for d in deltas(out) if d.get("tool_calls")]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["tool_calls"][0]["function"]["arguments"] == ARGS
    # the finish chunk survives, minus the duplicate
    last = out[-1]
    assert last["choices"][0]["finish_reason"] == "tool_calls"
    assert last["usage"]["completion_tokens"] == 46
    assert not last["choices"][0]["delta"].get("tool_calls")


def test_incremental_tool_call_arguments_pass_through_untouched():
    chunks = [
        chunk({"tool_calls": [tc(0, "call_a", "lookup_account", "")]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"phone": "555'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '0142219"}'}}]}),
        chunk(finish="tool_calls"),
    ]
    _, out = feed_all(chunks)
    assert deltas(out) == [c["choices"][0]["delta"] for c in chunks]


def test_two_distinct_tool_calls_both_pass_through():
    _, out = feed_all([
        chunk({"tool_calls": [tc(0, "call_a", "cancel_appointment", '{"appointment_id": "A-2001"}')]}),
        chunk({"tool_calls": [tc(1, "call_b", "create_ticket", '{"category": "unauthorized_transaction"}')]}),
        chunk(finish="tool_calls"),
    ])
    names = [d["tool_calls"][0]["function"]["name"] for d in deltas(out) if d.get("tool_calls")]
    assert names == ["cancel_appointment", "create_ticket"]


def test_think_block_is_stripped_from_content():
    _, out = feed_all([
        chunk({"role": "assistant"}),
        chunk({"content": "<think>"}),
        chunk({"content": "the caller wants a slot"}),
        chunk({"content": "</think>"}),
        chunk({"content": "Which day works for you?"}),
        chunk(finish="stop"),
    ])
    text = "".join(d.get("content") or "" for d in deltas(out))
    assert text == "Which day works for you?"


def test_think_tags_split_across_chunks_are_still_stripped():
    _, out = feed_all([
        chunk({"content": "<thi"}),
        chunk({"content": "nk>\nhidden</th"}),
        chunk({"content": "ink>\n\nHi Dana."}),
        chunk(finish="stop"),
    ])
    text = "".join(d.get("content") or "" for d in deltas(out))
    assert text == "Hi Dana."


def test_raw_tool_call_text_in_content_becomes_a_tool_calls_delta():
    raw = '<tool_call>\n{"name": "lookup_account", "arguments": {"phone": "5550142219", "date_of_birth": "1988-03-12"}}\n</tool_call>'
    _, out = feed_all([
        chunk({"content": raw[:12]}),
        chunk({"content": raw[12:40]}),
        chunk({"content": raw[40:]}),
        chunk(finish="stop"),
    ])
    ds = deltas(out)
    assert "".join(d.get("content") or "" for d in ds) == ""
    calls = [d["tool_calls"][0] for d in ds if d.get("tool_calls")]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup_account"
    assert json.loads(calls[0]["function"]["arguments"]) == {"phone": "5550142219", "date_of_birth": "1988-03-12"}
    assert out[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_reasoning_content_field_is_dropped():
    _, out = feed_all([
        chunk({"reasoning_content": "let me think"}),
        chunk({"content": "Sure."}),
        chunk(finish="stop"),
    ])
    assert all("reasoning_content" not in d for d in deltas(out))
    assert "".join(d.get("content") or "" for d in deltas(out)) == "Sure."


def test_plain_text_reply_passes_through_unchanged():
    chunks = [chunk({"role": "assistant"}), chunk({"content": "Thanks Dana, "}), chunk({"content": "you're verified."}), chunk(finish="stop", usage={"prompt_tokens": 2140, "completion_tokens": 9})]
    _, out = feed_all(chunks)
    assert out == chunks


def test_stats_count_what_was_normalized():
    t, _ = feed_all([
        chunk({"content": "<think>x</think>"}),
        chunk({"tool_calls": [tc(0, "a", "lookup_account", ARGS)]}),
        chunk({"tool_calls": [tc(0, "b", "lookup_account", ARGS)]}, finish="tool_calls"),
    ])
    assert t.stats["dropped_duplicate_tool_calls"] == 1
    assert t.stats["stripped_think_chars"] == len("x")
    assert t.stats["tool_calls"] == 1


def test_bracketed_tool_preamble_is_stripped_even_when_split_across_chunks():
    _, out = feed_all([
        chunk({"content": "[Call"}),
        chunk({"content": "ing tool: lookup_acc"}),
        chunk({"content": "ount]"}),
        chunk({"tool_calls": [tc(0, "a", "lookup_account", ARGS)]}, finish="tool_calls"),
    ])
    assert "".join(d.get("content") or "" for d in deltas(out)) == ""
    assert sum(1 for d in deltas(out) if d.get("tool_calls")) == 1


def test_unclosed_tool_preamble_is_dropped_at_finish():
    t, out = feed_all([chunk({"content": "[Calling tool"}), chunk(finish="stop")])
    assert "".join(d.get("content") or "" for d in deltas(out)) == ""
    assert t.stats["stripped_preamble_chars"] == len("[Calling tool")


def test_ordinary_brackets_in_speech_pass_through():
    _, out = feed_all([chunk({"content": "Your card [ending 4219] is active."}), chunk(finish="stop")])
    assert "".join(d.get("content") or "" for d in deltas(out)) == "Your card [ending 4219] is active."
