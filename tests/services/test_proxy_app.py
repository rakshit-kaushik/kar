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


# ---- hosted fallback at a turn boundary (CONTRACT §7 / plan T+13.5): on upstream error or no first
# token within the timeout, the same request is re-issued to the hosted model before any token has
# been sent to the client, and a `fallback` event is logged.

def slow_or_dead_upstream(mode: str):
    import asyncio

    async def chat(request: Request):
        if mode == "dead":
            return JSONResponse({"error": "engine crashed"}, status_code=500)

        async def gen():
            await asyncio.sleep(0.5)  # first token far later than the test's timeout
            yield sse(chunk({"content": "late"}, finish="stop"))
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def hosted_upstream(received: list):
    async def chat(request: Request):
        received.append(await request.json())

        async def gen():
            yield sse({**chunk({"role": "assistant"}), "model": "claude-sonnet-5-5"})
            yield sse({**chunk({"content": "Hello from hosted."}, finish="stop", usage={"prompt_tokens": 900, "completion_tokens": 5}), "model": "claude-sonnet-5-5"})
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def fallback_harness(mode: str):
    hosted_received, events = [], []
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=slow_or_dead_upstream(mode)), base_url="http://upstream")
    hosted = httpx.AsyncClient(transport=httpx.ASGITransport(app=hosted_upstream(hosted_received)), base_url="http://hosted")

    async def sink(ev):
        events.append(ev)

    app = create_app(upstream_client=upstream, event_sink=sink, fallback_client=hosted, fallback_model="claude-sonnet-5-5", first_token_timeout=0.1)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy"), hosted_received, events


async def test_upstream_server_error_falls_back_to_hosted_before_any_token():
    client, hosted_received, events = fallback_harness("dead")
    r = await client.post("/v1/chat/completions", json={**BODY, "temperature": 0.2, "top_p": 1.0}, headers={"X-R3-Call-Id": "c1"})
    assert r.status_code == 200
    chunks = [c for c in parse_sse(r.text) if c != "[DONE]"]
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "Hello from hosted."
    assert hosted_received[0]["model"] == "claude-sonnet-5-5" and hosted_received[0]["messages"] == BODY["messages"]
    # Claude 5.5's OpenAI-compatible endpoint rejects sampling knobs with 400 ("`temperature` is deprecated for this model")
    assert "temperature" not in hosted_received[0] and "top_p" not in hosted_received[0]
    fb = [e for e in events if e["event"] == "fallback"]
    assert len(fb) == 1 and fb[0]["payload"]["reason"] == "upstream_error" and fb[0]["payload"]["to_model"] == "claude-sonnet-5-5"
    assert events[-1]["event"] == "llm_done" and events[-1]["model_id"] == "claude-sonnet-5-5"


async def test_first_token_timeout_falls_back_to_hosted():
    client, hosted_received, events = fallback_harness("slow")
    r = await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "c2"})
    chunks = [c for c in parse_sse(r.text) if c != "[DONE]"]
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "Hello from hosted."
    assert len(hosted_received) == 1
    fb = [e for e in events if e["event"] == "fallback"]
    assert fb[0]["payload"]["reason"] == "first_token_timeout"


async def test_without_fallback_configured_upstream_error_is_a_502(harness):
    client, _, _ = harness
    dead = httpx.AsyncClient(transport=httpx.ASGITransport(app=slow_or_dead_upstream("dead")), base_url="http://upstream")
    app = create_app(upstream_client=dead, event_sink=None)
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 500 or r.status_code == 502


# ---- hosted unavailable (Nov 1: "400 Your credit balance is too low"): a hosted 4xx opens a breaker; the
# abandoned local request is re-issued without the first-token timeout, and while the breaker is open the
# proxy waits on local instead of falling back, so slow turns are not turned into errors.

def hosted_400_upstream(received: list):
    async def chat(request: Request):
        received.append(await request.json())
        return JSONResponse({"error": {"message": "Your credit balance is too low to access the Anthropic API."}}, status_code=400)

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


async def test_hosted_4xx_opens_breaker_and_local_is_retried_without_timeout():
    hosted_received, events = [], []
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=slow_or_dead_upstream("slow")), base_url="http://upstream")
    hosted = httpx.AsyncClient(transport=httpx.ASGITransport(app=hosted_400_upstream(hosted_received)), base_url="http://hosted")

    async def sink(ev):
        events.append(ev)

    app = create_app(upstream_client=upstream, event_sink=sink, fallback_client=hosted, fallback_model="claude-sonnet-5-5", first_token_timeout=0.1)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    r1 = await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "b1"})
    assert r1.status_code == 200
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in parse_sse(r1.text) if c != "[DONE]") == "late"
    assert len(hosted_received) == 1
    assert [e["event"] for e in events].count("fallback_unavailable") == 1
    assert r1.headers["X-R3-Model"] == "r3-voice-base"

    r2 = await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "b2"})
    assert r2.status_code == 200
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in parse_sse(r2.text) if c != "[DONE]") == "late"
    assert len(hosted_received) == 1  # breaker open: hosted not tried again, local waited past the timeout

    h = (await client.get("/health")).json()
    assert h["fallback"]["available"] is False and "400" in h["fallback"]["unavailable_reason"]


# ---- hosted requests drop the token budget (Nov 1: claude-sonnet-5-5's hidden reasoning consumed a max_tokens=96 budget
# and the stream ended with finish_reason=length and no visible token) and admission control routes calls above the
# measured concurrency limit to the hosted model instead of queueing them behind the SimpleEngine (plan T+13.5).

async def test_hosted_request_drops_max_tokens():
    client, hosted_received, _ = fallback_harness("dead")
    r = await client.post("/v1/chat/completions", json={**BODY, "max_tokens": 96, "max_completion_tokens": 96}, headers={"X-R3-Call-Id": "m1"})
    assert r.status_code == 200 and len(hosted_received) == 1
    assert "max_tokens" not in hosted_received[0] and "max_completion_tokens" not in hosted_received[0]


async def test_second_concurrent_call_above_admission_limit_is_routed_to_hosted():
    import asyncio
    hosted_received, events = [], []
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=slow_or_dead_upstream("slow")), base_url="http://upstream")
    hosted = httpx.AsyncClient(transport=httpx.ASGITransport(app=hosted_upstream(hosted_received)), base_url="http://hosted")

    async def sink(ev):
        events.append(ev)

    app = create_app(upstream_client=upstream, event_sink=sink, fallback_client=hosted, fallback_model="claude-sonnet-5-5", first_token_timeout=5.0, admission_limit=1)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    async def call(cid):
        r = await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": cid})
        return r.headers["X-R3-Model"], "".join(c["choices"][0]["delta"].get("content") or "" for c in parse_sse(r.text) if c != "[DONE]")

    first = asyncio.create_task(call("a1"))
    await asyncio.sleep(0.1)  # the first call is in flight locally (0.5 s to its first token)
    second = await call("a2")
    assert second == ("claude-sonnet-5-5", "Hello from hosted.")
    assert await first == ("r3-voice-base", "late")
    adm = [e for e in events if e["event"] == "admission"]
    assert len(adm) == 1 and adm[0]["call_id"] == "a2" and adm[0]["payload"] == {"in_flight": 1, "limit": 1, "routed_to": "claude-sonnet-5-5"}
    assert len(hosted_received) == 1
    # once the first call is done, the next call is served locally again
    assert await call("a3") == ("r3-voice-base", "late")


async def test_admission_limit_without_hosted_available_queues_locally():
    import asyncio
    hosted_received, events = [], []
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=slow_or_dead_upstream("slow")), base_url="http://upstream")
    hosted = httpx.AsyncClient(transport=httpx.ASGITransport(app=hosted_400_upstream(hosted_received)), base_url="http://hosted")

    async def sink(ev):
        events.append(ev)

    app = create_app(upstream_client=upstream, event_sink=sink, fallback_client=hosted, fallback_model="claude-sonnet-5-5", first_token_timeout=5.0, admission_limit=1)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    rs = await asyncio.gather(*[client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": f"q{i}"}) for i in range(2)])
    # the second call was routed to hosted, hosted answered 400, the breaker opened and the call was served locally after all
    assert [r.headers["X-R3-Model"] for r in rs] == ["r3-voice-base", "r3-voice-base"]
    assert len(hosted_received) == 1
    r3 = await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "q3"})
    assert r3.headers["X-R3-Model"] == "r3-voice-base" and len(hosted_received) == 1  # breaker open: no routing to hosted


# ---- hedged fallback: when the first-token timeout fires the local request keeps running and races the hosted one;
# the first stream to produce a token is served and the other is cancelled. (Nov 1: cancelling the local request and
# re-issuing it after a hosted 400 cost 5-8 s per turn while the engine finished the orphaned request first.)

def slow_hosted_upstream(received: list, delay_s: float):
    import asyncio

    async def chat(request: Request):
        received.append(await request.json())

        async def gen():
            await asyncio.sleep(delay_s)
            yield sse({**chunk({"role": "assistant"}), "model": "claude-sonnet-5-5"})
            yield sse({**chunk({"content": "Hello from hosted."}, finish="stop"), "model": "claude-sonnet-5-5"})
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


async def test_local_keeps_running_after_the_timeout_and_wins_against_a_slower_hosted_reply():
    hosted_received, events = [], []
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=slow_or_dead_upstream("slow")), base_url="http://upstream")  # first token at 0.5 s
    hosted = httpx.AsyncClient(transport=httpx.ASGITransport(app=slow_hosted_upstream(hosted_received, 2.0)), base_url="http://hosted")

    async def sink(ev):
        events.append(ev)

    app = create_app(upstream_client=upstream, event_sink=sink, fallback_client=hosted, fallback_model="claude-sonnet-5-5", first_token_timeout=0.1)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    import time
    t0 = time.perf_counter()
    r = await client.post("/v1/chat/completions", json=BODY, headers={"X-R3-Call-Id": "h1"})
    elapsed = time.perf_counter() - t0
    assert r.headers["X-R3-Model"] == "r3-voice-base"
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in parse_sse(r.text) if c != "[DONE]") == "late"
    assert elapsed < 1.5, f"local should win at ~0.5 s, not wait for hosted: {elapsed:.2f}s"
    assert len(hosted_received) == 1  # hosted was tried (fallback event), then cancelled
    kinds = [e["event"] for e in events]
    assert "fallback" in kinds and events[-1]["event"] == "llm_done" and events[-1]["model_id"] == "r3-voice-base"
    ft = [e for e in events if e["event"] == "llm_first_token"]
    assert ft and ft[0]["payload"]["source"] == "local"
