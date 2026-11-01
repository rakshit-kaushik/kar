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
from typing import Any, Callable

import httpx
import psycopg
from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli
from livekit.agents import EndpointingOptions, PreemptiveGenerationOptions, TurnHandlingOptions
import numpy as np
from livekit.agents.metrics import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics
from livekit.plugins import deepgram, silero
from livekit.plugins import openai as lk_openai
from livekit.plugins.turn_detector.english import EnglishModel
from openai import AsyncOpenAI
from psycopg.types.json import Jsonb

from services.agent.state import CallState
from services.agent.faults import faults_from_room_name
from services.agent.filler import FillerPolicy, filler_text
from services.agent.claimguard import ClaimGuard, claim_guarded_stream, handoff_reason, state_view
from services.agent.loopguard import LoopGuard, loop_guarded
from services.agent.policy import HandoffPolicy, policy_stream
from services.agent.tts import make_tts, tts_vendor
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
# Turn-taking knobs (AGENT_* in .env). Defaults are LiveKit's; .env carries the values chosen by measurement.
REPROMPT_LINE = os.environ.get("AGENT_REPROMPT_LINE", "Sorry, I didn't catch that. Could you say it again?")

TURN = {
    "vad_min_silence_s": float(os.environ.get("AGENT_VAD_MIN_SILENCE", "0.55")),
    "endpoint_min_s": float(os.environ.get("AGENT_ENDPOINT_MIN", "0.5")),
    "endpoint_max_s": float(os.environ.get("AGENT_ENDPOINT_MAX", "3.0")),
    "preemptive": os.environ.get("AGENT_PREEMPTIVE", "0") == "1",
    # caller speech that VAD heard but STT never transcribed (two rehearsal stalls, Nov 1): after this many seconds the session
    # emits user_transcription_timeout and the agent asks the caller to say it again; 0 disables (LiveKit docs: "use it to prompt the user")
    "transcription_timeout_s": float(os.environ.get("AGENT_TRANSCRIPTION_TIMEOUT", "2.5")),
}
AUDIBLE_RMS = 300.0


class R3Agent(Agent):
    """Agent with one measurement hook: the moment the first audible TTS frame leaves the TTS node, so the
    agent-side clock has a 'first audio' timestamp to compare with what the caller hears."""

    def __init__(self, *, on_first_audible, guard: LoopGuard, on_loop, claim_guard: ClaimGuard, handoff_done, on_claim, completed_tools=None, on_action_claim=None,
                 policy: HandoffPolicy | None = None, policy_state=None, on_policy=None, **kw) -> None:
        super().__init__(**kw)
        self._on_first_audible = on_first_audible
        self._guard, self._on_loop = guard, on_loop
        self._claim_guard, self._handoff_done, self._on_claim = claim_guard, handoff_done, on_claim
        self._completed_tools, self._on_action_claim = completed_tools, on_action_claim
        self._policy, self._policy_state, self._on_policy = policy, policy_state, on_policy

    async def llm_node(self, chat_ctx, tools, model_settings):
        # services/agent/claimguard.py: a transfer announced without request_handoff gets the handoff recorded first;
        # services/agent/loopguard.py: an exact repeat of the previous line is swallowed (rephrase, then handoff); a regeneration
        # the caller never heard (preemptive generation discarded by the pipeline, or cancelled by the caller talking on) is not a repeat
        last = next((it for it in reversed(chat_ctx.items) if getattr(it, "role", None) == "assistant"), None)
        self._guard.begin_turn(getattr(last, "text_content", None) if last is not None else None, interrupted=bool(getattr(last, "interrupted", False)))
        source = claim_guarded_stream(Agent.default.llm_node(self, chat_ctx, tools, model_settings), self._claim_guard, self._handoff_done, self._on_claim,
                                      completed_tools=self._completed_tools, on_action_claim=self._on_action_claim)
        if self._policy is not None and self._policy_state is not None:
            source = policy_stream(source, self._policy, self._policy_state, self._on_policy)  # contract rule: a fraud ticket ends in a transfer
        async for chunk in loop_guarded(source, self._guard, self._on_loop, self._handoff_done):
            yield chunk

    async def tts_node(self, text, model_settings):
        t0, fired, frames = time.perf_counter(), False, 0
        async for frame in Agent.default.tts_node(self, text, model_settings):
            frames += 1
            if not fired:
                x = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
                if len(x) and float(np.sqrt(np.mean(x * x))) >= AUDIBLE_RMS:
                    fired = True
                    self._on_first_audible({"ms_since_tts_start": round((time.perf_counter() - t0) * 1000), "frames_before": frames - 1})
            yield frame
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
    filler: FillerPolicy = field(default_factory=lambda: FillerPolicy(filler_text()))
    guard: LoopGuard = field(default_factory=LoopGuard.from_env)
    claim_guard: ClaimGuard = field(default_factory=ClaimGuard.from_env)
    policy: HandoffPolicy = field(default_factory=HandoffPolicy.from_env)
    fire: Callable[..., None] | None = None  # set per call: the turn-event emitter


server = AgentServer(num_idle_processes=1)  # a warm job process: the agent joins ~1 s sooner


def prewarm(proc: JobProcess) -> None:
    # VAD end-of-speech fires min_silence_duration after the caller stops; the LiveKit default 0.55 s is a large
    # share of the 1.0 s end-of-speech -> first-audio budget. Tuned by measurement with scripts/sim_caller.py.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=float(os.environ.get("AGENT_VAD_MIN_SILENCE", "0.55")))


server.setup_fnc = prewarm
def _tool_start(ctx, tool_name: str) -> None:
    """The moment a tool starts: speak the filler (services/agent/filler.py) and log it as `filler_start`. Not added to
    the chat context, so the model never sees it. Never lets an error reach the tool call."""
    d: CallData = ctx.userdata
    try:
        text = d.filler.decide(d.turn_idx, str(ctx.session.agent_state), tool_name)
        if text and d.fire is not None:
            d.fire("filler_start", {"tool": tool_name, "text": text})
            ctx.session.say(text, allow_interruptions=True, add_to_chat_ctx=False)
    except Exception as e:  # a filler must never cost the tool call
        log.warning("filler skipped: %r", e)


TOOLS = build_livekit_tools(lambda ctx: ctx.userdata.executor, on_tool_start=_tool_start)


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
    faults = faults_from_room_name(call_id)  # live demo: room name '...--fault-check_availability-timeout' (services/agent/faults.py)
    if faults:
        sandbox.inject(faults)
    executor = ToolExecutor(sandbox=sandbox, state=state, event_sink=sink)
    data = CallData(call_id=call_id, model_id=model_id, state=state, sandbox=sandbox, executor=executor, sink=sink)

    async def emit(event: str, payload: dict[str, Any] | None = None) -> None:
        ev = {"call_id": call_id, "turn_idx": data.turn_idx, "model_id": model_id, "event": event, "ts_ms": time.time_ns() // 1_000_000, "payload": payload or {}}
        log.info(json.dumps(ev, separators=(",", ":"), default=str))
        if sink is not None:
            await sink(ev)

    def fire(event: str, payload: dict[str, Any] | None = None) -> None:
        asyncio.create_task(emit(event, payload))

    data.fire = fire

    async def on_loop(action: str, repeated: str) -> None:
        """The loop guard replaced a repeated line: log it; on the second repeat record the handoff through the executor."""
        await emit("loop_guard", {"action": action, "repeated": repeated[:200], "repeats": data.guard.repeats})
        if action == "handoff":
            await executor.execute("request_handoff", {"reason": "out_of_scope", "summary": f"agent repeated itself: {repeated[:120]}"})

    async def on_claim(text: str) -> None:
        """The model announced a transfer with no request_handoff in the call: record the handoff first, then let it speak."""
        reason = handoff_reason(state_view(state))
        await emit("claim_guard", {"kind": "transfer", "claimed": text[:200], "reason": reason})
        await executor.execute("request_handoff", {"reason": reason, "summary": f"agent announced a transfer without calling request_handoff: {text[:120]}"})

    async def on_policy(reason: str) -> None:
        """The contract requires a transfer the model did not make (fraud-category ticket, no handoff): record it, then the agent says so."""
        await emit("policy_handoff", {"reason": reason, "rule": "fraud_ticket"})
        await executor.execute("request_handoff", {"reason": reason, "summary": "policy: a fraud-category ticket requires a transfer; the model closed without one"})

    async def on_action_claim(kind: str, tool: str, sentence: str) -> None:
        """The model claimed a finished write (booked, moved, cancelled, ticket) with no successful result of that tool in the call:
        the sentence was held back and the agent asks instead. Counted in the claim_guard column, never as the model's recovery."""
        await emit("claim_guard", {"kind": kind, "tool": tool, "claimed": sentence[:200], "replaced": True})

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
        tts=make_tts(),  # services/agent/tts.py: AGENT_TTS=cartesia|deepgram
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=EnglishModel(),
            endpointing=EndpointingOptions(min_delay=TURN["endpoint_min_s"], max_delay=TURN["endpoint_max_s"]),
            preemptive_generation=PreemptiveGenerationOptions(enabled=TURN["preemptive"]),
        ),
        userdata=data,
        max_tool_steps=3,
        transcription_timeout=TURN["transcription_timeout_s"] or None,
    )

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        if ev.new_state == "speaking":
            fire("speech_start")  # VAD onset: with the caller's own capture time this measures caller -> agent transport
        if ev.old_state == "speaking" and ev.new_state == "listening":
            data.turn_idx += 1
            executor.turn_idx = data.turn_idx
            fire("speech_end")

    @session.on("user_transcription_timeout")
    def _on_transcription_timeout(ev) -> None:
        # VAD heard the caller, STT produced nothing final: without this both sides wait (rehearsal interruption calls 3 and 5).
        # Not in the chat context, so the model and the loop guard never see it; identical for every model; its own event and count.
        fire("stt_timeout_reprompt", {"speech_duration": round(float(getattr(ev, "speech_duration", 0.0)), 2), "text": REPROMPT_LINE})
        session.say(REPROMPT_LINE, allow_interruptions=True, add_to_chat_ctx=False)

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
        await emit("call_end", {"outcome": outcome, "tts_vendor": tts_vendor(), "turns": data.turn_idx, "structural_leaks": data.structural_leaks, "duration_s": round(time.time() - data.started, 1)})
        await db_exec("update sessions set ended_at = now(), outcome = %s, state = %s where call_id = %s", (outcome, Jsonb(state.to_dict()), call_id))
        # the room dies with the call: a rejoin under the same name must get a fresh room and a fresh agent, never a leftover one
        try:
            await ctx.delete_room()
        except Exception as e:
            log.info("delete_room: %r", e)

    ctx.add_shutdown_callback(on_shutdown)

    @session.on("close")
    def _session_closed(ev) -> None:
        fire("session_closed", {"reason": str(getattr(ev, "reason", "")), "error": repr(getattr(ev, "error", None))})
        ctx.shutdown(reason=f"session closed: {getattr(ev, 'reason', '')}")

    @ctx.room.on("participant_disconnected")
    def _caller_left(p) -> None:
        if getattr(p, "kind", None) != 4:  # 4 = agent; anything else is the caller
            fire("caller_left", {"identity": p.identity})
            ctx.shutdown(reason="caller left")

    await session.start(agent=R3Agent(instructions=SYSTEM_PROMPT, tools=TOOLS, on_first_audible=lambda p: fire("tts_first_audible", p), guard=data.guard, on_loop=on_loop, claim_guard=data.claim_guard, handoff_done=lambda: bool(state.handoff), on_claim=on_claim,
                                      completed_tools=lambda: {c.get("tool") for c in state.completed}, on_action_claim=on_action_claim,
                                      policy=data.policy, policy_state=lambda: state_view(state), on_policy=on_policy), room=ctx.room)
    await ctx.connect()
    await emit("call_start", {"llm_base_url": base_url, "stt": "deepgram/nova-3", "tts": tts_vendor(), "filler": filler_text(), "loop_guard": data.guard.enabled, "claim_guard": data.claim_guard.enabled, "policy_handoff": data.policy.enabled, "transcription_timeout_s": TURN["transcription_timeout_s"], "turn": TURN})
    session.say(GREETING, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
