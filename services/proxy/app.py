"""Proxy on 8001 between every LLM consumer and the model server on 8000.

- Streams `/v1/chat/completions` through `StreamTransformer` (dedupes vllm-mlx's repeated tool-call
  deltas, strips `<think>`, converts leaked raw `<tool_call>` text).
- Emits `llm_request`, `llm_first_token`, `llm_done` turn events (CONTRACT.md §6), keyed by the
  `X-R3-Call-Id` header the agent sets; turn index increments per call id.
- Hosted fallback at a turn boundary (plan T+13.5): if the local server errors or produces no first
  token within `first_token_timeout` seconds, the same request is re-issued to `fallback_client` as
  `fallback_model` before any byte has reached the client, and a `fallback` event is logged. The agent
  never notices; its chat context is the same, so call state is preserved.
- Passes everything else through (`/v1/models`, non-streaming completions).
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from services.proxy.transform import StreamTransformer

log = logging.getLogger("r3.proxy")
EventSink = Callable[[dict[str, Any]], Awaitable[None]]


def now_ms() -> int:
    """Wall-clock ms: comparable across the agent and proxy processes (contract §6 ts_ms)."""
    return time.time_ns() // 1_000_000


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


# Claude 5.5's OpenAI-compatible endpoint returns 400 for sampling knobs ("`temperature` is deprecated for this model"),
# and its hidden reasoning consumes the token budget (a max_tokens=96 request ended with finish_reason=length and no
# visible token), so a request re-issued to the hosted model goes without them; the local server keeps its own defaults.
HOSTED_DROP = ("temperature", "top_p", "max_tokens", "max_completion_tokens")


class HostedBreaker:
    """Hosted model availability. A 4xx from the hosted endpoint (bad key, no credit, rejected params) means
    retrying it is pointless: open for `cooldown_s`, during which the proxy waits on local instead of falling back."""

    def __init__(self, cooldown_s: float = 600.0) -> None:
        self.cooldown_s, self.until, self.reason = cooldown_s, 0.0, None

    def available(self) -> bool:
        return time.monotonic() >= self.until

    def trip(self, reason: str) -> None:
        self.until, self.reason = time.monotonic() + self.cooldown_s, reason


class UpstreamStatus(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"upstream status {status}")
        self.status, self.body = status, body


class _Stream:
    """One upstream SSE response being normalized. `prime()` reads up to the first token so the
    caller can decide to fall back before anything reaches the client; `rest()` replays what was
    buffered and continues."""

    def __init__(self, resp: httpx.Response, model_id: str) -> None:
        self.resp, self.model_id = resp, model_id
        self.lines = resp.aiter_lines()
        self.tr = StreamTransformer()
        self.usage: dict = {}
        self.first_token_ms: int | None = None
        self.done = False
        self.buffered: list[str] = []

    def process(self, line: str) -> list[str]:
        if not line.startswith("data:"):
            return []
        payload = line[5:].strip()
        if payload == "[DONE]":
            self.done = True
            return [sse(c) for c in self.tr.finish()] + ["data: [DONE]\n\n"]
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return [f"{line}\n\n"]
        if obj.get("usage"):
            self.usage = obj["usage"]
        out = []
        for c in self.tr.feed(obj):
            if self.first_token_ms is None and c.get("choices"):
                d = c["choices"][0].get("delta") or {}
                if d.get("content") or d.get("tool_calls"):
                    self.first_token_ms = now_ms()
            out.append(sse(c))
        return out

    async def prime(self) -> None:
        while self.first_token_ms is None and not self.done:
            try:
                line = await self.lines.__anext__()
            except StopAsyncIteration:
                self.done = True
                break
            self.buffered.extend(self.process(line))

    async def rest(self):
        for s in self.buffered:
            yield s
        if not self.done:
            async for line in self.lines:
                for s in self.process(line):
                    yield s
                if self.done:
                    break


async def open_stream(client: httpx.AsyncClient, body: dict, model_id: str, timeout: float | None) -> _Stream:
    """POST the request, check the status, read up to the first token. `timeout` bounds the whole
    thing (connect + headers + first token); None means no bound beyond the client's own."""

    async def go() -> _Stream:
        resp = await client.send(client.build_request("POST", "/v1/chat/completions", json=body), stream=True)
        if resp.status_code != 200:
            text = (await resp.aread()).decode(errors="replace")
            await resp.aclose()
            raise UpstreamStatus(resp.status_code, text)
        s = _Stream(resp, model_id)
        try:
            await s.prime()
        except BaseException:
            await resp.aclose()
            raise
        return s

    return await asyncio.wait_for(go(), timeout)


def create_app(
    upstream_client: httpx.AsyncClient,
    event_sink: EventSink | None = None,
    fallback_client: httpx.AsyncClient | None = None,
    fallback_model: str | None = None,
    first_token_timeout: float = 2.0,
    admission_limit: int = 0,
) -> Starlette:
    """admission_limit: concurrent local generations the engine serves within budget (load test); a streamed turn
    arriving above it is routed to the hosted model when that is available, instead of queueing. 0 disables."""
    turn_counters: dict[str, int] = collections.defaultdict(int)
    breaker = HostedBreaker()
    inflight = {"local": 0}

    async def emit(call_id: str, turn_idx: int, model_id: str, event: str, payload: dict | None = None) -> None:
        ev = {"call_id": call_id, "turn_idx": turn_idx, "model_id": model_id, "event": event, "ts_ms": now_ms(), "payload": payload or {}}
        log.info(json.dumps(ev, separators=(",", ":")))
        if event_sink is not None:
            await event_sink(ev)

    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        call_id = request.headers.get("x-r3-call-id") or f"anon-{uuid.uuid4().hex[:8]}"
        turn_idx = turn_counters[call_id]
        turn_counters[call_id] += 1
        model_id = str(body.get("model", "unknown"))
        t0 = now_ms()
        await emit(call_id, turn_idx, model_id, "llm_request", {
            "messages": len(body.get("messages") or []), "tools": len(body.get("tools") or []), "stream": bool(body.get("stream")),
        })

        if not body.get("stream"):
            try:
                r = await upstream_client.post("/v1/chat/completions", json=body)
            except httpx.HTTPError as e:
                await emit(call_id, turn_idx, model_id, "llm_error", {"error": repr(e)})
                return JSONResponse({"error": {"message": f"upstream unavailable: {e}", "type": "upstream_error"}}, status_code=502)
            data = r.json()
            usage = data.get("usage") or {}
            await emit(call_id, turn_idx, model_id, "llm_first_token", {"ms": now_ms() - t0})
            await emit(call_id, turn_idx, model_id, "llm_done", {"ms": now_ms() - t0, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "stream": False})
            return JSONResponse(data, status_code=r.status_code)

        hosted_ok = fallback_client is not None and bool(fallback_model) and breaker.available()
        attempts: list[tuple[httpx.AsyncClient, str, str]]
        if admission_limit and hosted_ok and inflight["local"] >= admission_limit:
            # above the measured concurrency limit: this turn goes to the hosted model instead of waiting a full turn
            await emit(call_id, turn_idx, model_id, "admission", {"in_flight": inflight["local"], "limit": admission_limit, "routed_to": fallback_model})
            attempts = [(fallback_client, fallback_model, "hosted")]
        else:
            attempts = [(upstream_client, model_id, "local")]
            if hosted_ok:
                attempts.append((fallback_client, fallback_model, "hosted"))

        s: _Stream | None = None
        served_source = "local"
        i = 0
        while i < len(attempts):
            client, mid, source = attempts[i]
            has_next = i + 1 < len(attempts)
            is_local = source != "hosted"
            if is_local:
                inflight["local"] += 1
            try:
                sent = {**body, "model": mid} if is_local else {k: v for k, v in body.items() if k not in HOSTED_DROP} | {"model": mid}
                s = await open_stream(client, sent, mid, first_token_timeout if has_next else None)
                served_source = source
                break
            except (httpx.HTTPError, UpstreamStatus, asyncio.TimeoutError) as e:
                if is_local:
                    inflight["local"] -= 1
                reason = "first_token_timeout" if isinstance(e, asyncio.TimeoutError) else "upstream_error"
                detail = {"status": e.status, "body": e.body[:500]} if isinstance(e, UpstreamStatus) else {"error": repr(e)}
                await emit(call_id, turn_idx, mid, "llm_error", {"reason": reason, "source": source, **detail})
                if source == "hosted" and isinstance(e, UpstreamStatus) and 400 <= e.status < 500:
                    # the hosted side rejects us outright: stop trying it, and give the local turn back its time
                    breaker.trip(f"{e.status}: {e.body[:160]}")
                    await emit(call_id, turn_idx, mid, "fallback_unavailable", {"status": e.status, "cooldown_s": breaker.cooldown_s, "body": e.body[:300]})
                    attempts.append((upstream_client, model_id, "local_retry"))
                    has_next = True
                if has_next:
                    await emit(call_id, turn_idx, mid, "fallback", {"reason": reason, "from_model": mid, "to_model": attempts[i + 1][1], "ms": now_ms() - t0})
                    i += 1
                    continue
                if isinstance(e, UpstreamStatus):
                    return Response(e.body, status_code=e.status, media_type="application/json")
                return JSONResponse({"error": {"message": f"upstream unavailable: {e}", "type": "upstream_error"}}, status_code=502)
        assert s is not None
        served = s
        if served.first_token_ms is not None:
            await emit(call_id, turn_idx, served.model_id, "llm_first_token", {"ms": served.first_token_ms - t0, "source": served_source})

        async def stream():
            try:
                async for piece in served.rest():
                    yield piece
            finally:
                if served_source != "hosted":
                    inflight["local"] -= 1
                await served.resp.aclose()
                await emit(call_id, turn_idx, served.model_id, "llm_done", {
                    "ms": now_ms() - t0, "prompt_tokens": served.usage.get("prompt_tokens"), "completion_tokens": served.usage.get("completion_tokens"),
                    "stream": True, "first_token": served.first_token_ms is not None, **served.tr.stats,
                })

        headers = {"Cache-Control": "no-cache", "X-R3-Call-Id": call_id, "X-R3-Turn": str(turn_idx), "X-R3-Model": served.model_id}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    async def passthrough(request: Request) -> Response:
        try:
            r = await upstream_client.request(request.method, request.url.path, params=request.query_params, content=await request.body())
        except httpx.HTTPError as e:
            return JSONResponse({"error": {"message": f"upstream unavailable: {e}"}}, status_code=502)
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    async def health(_: Request) -> Response:
        try:
            r = await upstream_client.get("/v1/models", timeout=5)
            up = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except httpx.HTTPError as e:
            up = f"down: {type(e).__name__}"
        return JSONResponse({
            "status": "ok", "upstream": up, "upstream_base_url": str(upstream_client.base_url),
            "fallback": {"model": fallback_model, "first_token_timeout_s": first_token_timeout, "available": breaker.available(), "unavailable_reason": breaker.reason} if fallback_client is not None else None,
        })

    return Starlette(routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", passthrough, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ])
