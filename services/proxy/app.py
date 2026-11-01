"""Proxy on 8001 between every LLM consumer and the model server on 8000.

- Streams `/v1/chat/completions` through `StreamTransformer` (dedupes vllm-mlx's repeated tool-call
  deltas, strips `<think>`, converts leaked raw `<tool_call>` text).
- Emits `llm_request`, `llm_first_token`, `llm_done` turn events (CONTRACT.md §6), keyed by the
  `X-R3-Call-Id` header the agent sets; turn index increments per call id.
- Passes everything else through (`/v1/models`, non-streaming completions).
"""
from __future__ import annotations

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


def create_app(upstream_client: httpx.AsyncClient, event_sink: EventSink | None = None) -> Starlette:
    turn_counters: dict[str, int] = collections.defaultdict(int)

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

        try:
            upstream_req = upstream_client.build_request("POST", "/v1/chat/completions", json=body)
            upstream_resp = await upstream_client.send(upstream_req, stream=True)
        except httpx.HTTPError as e:
            await emit(call_id, turn_idx, model_id, "llm_error", {"error": repr(e)})
            return JSONResponse({"error": {"message": f"upstream unavailable: {e}", "type": "upstream_error"}}, status_code=502)
        if upstream_resp.status_code != 200:
            text = (await upstream_resp.aread()).decode(errors="replace")
            await upstream_resp.aclose()
            await emit(call_id, turn_idx, model_id, "llm_error", {"status": upstream_resp.status_code, "body": text[:500]})
            return Response(text, status_code=upstream_resp.status_code, media_type=upstream_resp.headers.get("content-type", "application/json"))

        async def stream():
            tr = StreamTransformer()
            first = False
            usage: dict = {}
            try:
                async for line in upstream_resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        for c in tr.finish():
                            yield sse(c)
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        yield f"{line}\n\n"
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for c in tr.feed(obj):
                        if not first and c.get("choices"):
                            d = c["choices"][0].get("delta") or {}
                            if d.get("content") or d.get("tool_calls"):
                                first = True
                                await emit(call_id, turn_idx, model_id, "llm_first_token", {"ms": now_ms() - t0})
                        yield sse(c)
            finally:
                await upstream_resp.aclose()
                await emit(call_id, turn_idx, model_id, "llm_done", {
                    "ms": now_ms() - t0, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "stream": True, "first_token": first, **tr.stats,
                })

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-R3-Call-Id": call_id, "X-R3-Turn": str(turn_idx)})

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
        return JSONResponse({"status": "ok", "upstream": up, "upstream_base_url": str(upstream_client.base_url)})

    return Starlette(routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", passthrough, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ])
