"""R3 voice agent on LiveKit Agents.

Run:  uv run python -m services.agent.main dev            # connect to LiveKit Cloud, auto-join new rooms
      uv run python -m services.agent.main console        # local mic/speaker, no browser
      uv run python -m services.agent.main download-files # turn-detector + Silero model files

Pipeline per call: Deepgram STT -> LLM at LLM_BASE_URL (via the proxy) -> contract tools on an
in-memory sandbox -> Cartesia TTS. Every stage writes CONTRACT.md §6 events to turn_events with the
room name as call_id; the proxy adds llm_* events for the same call_id via the X-R3-Call-Id header.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import psycopg
from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli
from livekit.agents.metrics import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics
from livekit.plugins import cartesia, deepgram, silero
from livekit.plugins import openai as lk_openai
from livekit.plugins.turn_detector.english import EnglishModel
from openai import AsyncOpenAI
from psycopg.types.json import Jsonb

from services.agent.state import CallState
from services.common.events import PostgresSink
from services.tools.executor import ToolExecutor
from services.tools.livekit_tools import build_livekit_tools
from services.tools.sandbox import Sandbox

ROOT = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
log = logging.getLogger("r3.agent")

SEED = json.loads((ROOT / "eval/sandbox/seed.json").read_text())
SYSTEM_PROMPT = (ROOT / "schemas/system_prompt.md").read_text().replace("{today}", SEED["clock_spoken"])
GREETING = "Thanks for calling Marlow Mutual Bank. Can I get the phone number on your account?"
STRUCTURAL = ("<tool_call>", "</tool_call>", "<think>", "</think>", '{"name"', "<|im_", "[Calling tool", "[calling tool")


@dataclass
class CallData:
    call_id: str
    model_id: str
    state: CallState
    sandbox: Sandbox
    executor: ToolExecutor
    sink: PostgresSink | None
    turn_idx: int = 0
    structural_leaks: int = 0
    started: float = field(default_factory=time.time)


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm
TOOLS = build_livekit_tools(lambda ctx: ctx.userdata.executor)


async def db_exec(sql: str, params: tuple) -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return
    try:
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            await conn.execute(sql, params)
    except Exception as e:  # never let bookkeeping take down a call
        log.warning("sessions write failed: %r", e)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    call_id = ctx.room.name
    model_id = os.environ.get("LLM_MODEL", "r3-voice-base")
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8001/v1")
    dsn = os.environ.get("DATABASE_URL")
    sink = PostgresSink(dsn) if dsn else None
    state = CallState(call_id=call_id, model_id=model_id)
    sandbox = Sandbox(SEED)
    executor = ToolExecutor(sandbox=sandbox, state=state, event_sink=sink)
    data = CallData(call_id=call_id, model_id=model_id, state=state, sandbox=sandbox, executor=executor, sink=sink)

    async def emit(event: str, payload: dict[str, Any] | None = None) -> None:
        ev = {"call_id": call_id, "turn_idx": data.turn_idx, "model_id": model_id, "event": event, "ts_ms": time.time_ns() // 1_000_000, "payload": payload or {}}
        log.info(json.dumps(ev, separators=(",", ":"), default=str))
        if sink is not None:
            await sink(ev)

    def fire(event: str, payload: dict[str, Any] | None = None) -> None:
        asyncio.create_task(emit(event, payload))

    await db_exec(
        "insert into sessions(call_id, model_id, started_at, state) values (%s, %s, now(), %s) on conflict (call_id) do nothing",
        (call_id, model_id, Jsonb(state.to_dict())),
    )

    api_key = os.environ.get("HOSTED_LLM_API_KEY", "local") if "anthropic" in base_url else "local"
    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers={"X-R3-Call-Id": call_id}, timeout=httpx.Timeout(60.0, connect=5.0))

    session: AgentSession[CallData] = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en-US", interim_results=True, punctuate=True, smart_format=False, filler_words=True),
        # no sampling knobs: the local server defaults to temperature 0.2 (restart.sh) and Claude 5.5's endpoint rejects temperature/top_p
        llm=lk_openai.LLM(model=model_id, client=llm_client, parallel_tool_calls=False),
        tts=cartesia.TTS(model=os.environ.get("CARTESIA_MODEL", "sonic-3"), voice=os.environ["CARTESIA_VOICE"]),
        vad=ctx.proc.userdata["vad"],
        turn_detection=EnglishModel(),
        userdata=data,
        max_tool_steps=3,
    )

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        if ev.old_state == "speaking" and ev.new_state == "listening":
            data.turn_idx += 1
            executor.turn_idx = data.turn_idx
            fire("speech_end")

    @session.on("user_input_transcribed")
    def _transcribed(ev) -> None:
        if ev.is_final:
            fire("stt_final", {"transcript": ev.transcript})

    @session.on("agent_state_changed")
    def _agent_state(ev) -> None:
        if ev.old_state == "speaking" and ev.new_state != "speaking":
            fire("playback_end", {"next_state": ev.new_state})
        elif ev.new_state == "speaking":
            fire("agent_speaking")

    @session.on("metrics_collected")
    def _metrics(ev) -> None:
        m = ev.metrics
        if isinstance(m, EOUMetrics):
            fire("eou_metrics", {"end_of_utterance_delay": m.end_of_utterance_delay, "transcription_delay": m.transcription_delay, "on_user_turn_completed_delay": m.on_user_turn_completed_delay})
        elif isinstance(m, LLMMetrics):
            fire("llm_metrics", {"ttft": m.ttft, "duration": m.duration, "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens, "cancelled": m.cancelled})
        elif isinstance(m, TTSMetrics):
            fire("tts_first_audio", {"ttfb": m.ttfb, "duration": m.duration, "audio_duration": m.audio_duration, "characters": m.characters_count, "cancelled": m.cancelled})
        elif isinstance(m, STTMetrics):
            fire("stt_metrics", {"duration": m.duration, "audio_duration": m.audio_duration})

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        item = ev.item
        if getattr(item, "type", None) != "message":
            return
        text = item.text_content or ""
        leak = any(t in text for t in STRUCTURAL)
        if leak and item.role == "assistant":
            data.structural_leaks += 1
        fire("transcript", {"role": item.role, "text": text, "structural": leak, "interrupted": bool(getattr(item, "interrupted", False))})
        if item.role == "assistant" and getattr(item, "interrupted", False):
            fire("interruption", {"truncated_text": text})

    @session.on("speech_created")
    def _speech(ev) -> None:
        fire("speech_created", {"source": ev.source, "user_initiated": ev.user_initiated})

    @session.on("error")
    def _error(ev) -> None:
        fire("error", {"error": repr(getattr(ev, "error", ev)), "source": repr(getattr(ev, "source", None))})

    async def on_shutdown() -> None:
        outcome = "handoff" if state.handoff else ("completed" if state.completed else "abandoned")
        await emit("call_end", {"outcome": outcome, "turns": data.turn_idx, "structural_leaks": data.structural_leaks, "duration_s": round(time.time() - data.started, 1)})
        await db_exec("update sessions set ended_at = now(), outcome = %s, state = %s where call_id = %s", (outcome, Jsonb(state.to_dict()), call_id))

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(agent=Agent(instructions=SYSTEM_PROMPT, tools=TOOLS), room=ctx.room)
    await ctx.connect()
    await emit("call_start", {"llm_base_url": base_url, "stt": "deepgram/nova-3", "tts": os.environ.get("CARTESIA_MODEL", "sonic-3")})
    session.say(GREETING, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
