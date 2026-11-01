"""The proxy on 8001: forwards chat completions to the model server, normalizes the SSE stream,
emits llm_request / llm_first_token / llm_done turn events keyed by the caller's X-R3-Call-Id."""
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from services.proxy.app import create_app

ARGS = json.dumps({"phone": "5550142219", "date_of_birth": "1988-03-12"})


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def chunk(delta=None, finish=None, usage=None):
    c = {"id": "cmpl-1", "object": "chat.completion.chunk", "model": "r3-voice-base", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


def tool(id_):
    return {"index": 0, "id": id_, "type": "function", "function": {"name": "lookup_account", "arguments": ARGS}}


def fake_upstream(received: list):
    async def chat(request: Request):
        body = await request.json()
        received.append(body)

        async def gen():
            yield sse(chunk({"role": "assistant"}))
            yield sse(chunk({"tool_calls": [tool("call_a")]}))
            yield sse(chunk({"tool_calls": [tool("call_b")]}, finish="tool_calls", usage={"prompt_tokens": 2023, "completion_tokens": 46, "total_tokens": 2069}))
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def models(_: Request):
        return JSONResponse({"object": "list", "data": [{"id": "r3-voice-base", "object": "model"}]})

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/models", models)])


@pytest.fixture
def harness():
    received, events = [], []
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_upstream(received)), base_url="http://upstream")

    async def sink(ev):
        events.append(ev)

    app = create_app(upstream_client=upstream, event_sink=sink)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    return client, received, events


def parse_sse(text: str):
    out = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


BODY = {"model": "r3-voice-base", "messages": [{"role": "user", "content": "hi"}], "tools": [], "stream": True}


async def test_streamed_tool_call_is_deduplicated_end_to_end(harness):
    client, received, _ = harness
    r = await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "call-1"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    items = parse_sse(r.text)
    assert items[-1] == "[DONE]"
    chunks = [c for c in items if c != "[DONE]"]
    tool_deltas = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_deltas) == 1
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"]["completion_tokens"] == 46
    assert received[0]["model"] == "r3-voice-base" and received[0]["stream"] is True


async def test_turn_events_emitted_with_tokens_and_call_id(harness):
    client, _, events = harness
    await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "call-7"})
    kinds = [e["event"] for e in events]
    assert kinds == ["llm_request", "llm_first_token", "llm_done"]
    assert all(e["call_id"] == "call-7" and e["model_id"] == "r3-voice-base" for e in events)
    assert events[0]["payload"]["messages"] == 1
    assert events[2]["payload"]["prompt_tokens"] == 2023 and events[2]["payload"]["completion_tokens"] == 46
    assert events[2]["payload"]["tool_calls"] == 1 and events[2]["payload"]["dropped_duplicate_tool_calls"] == 1
    assert events[0]["ts_ms"] <= events[1]["ts_ms"] <= events[2]["ts_ms"]


async def test_turn_index_increments_per_call_id(harness):
    client, _, events = harness
    for _ in range(2):
        await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "call-9"})
    await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "call-10"})
    turns = [(e["call_id"], e["turn_idx"]) for e in events if e["event"] == "llm_request"]
    assert turns == [("call-9", 0), ("call-9", 1), ("call-10", 0)]


async def test_models_endpoint_passes_through(harness):
    client, _, _ = harness
    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "r3-voice-base"


async def test_health_reports_upstream(harness):
    client, _, _ = harness
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["upstream"] == "ok"
