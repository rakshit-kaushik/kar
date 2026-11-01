#!/usr/bin/env python
"""Audio-level simulated caller: joins a LiveKit room like a browser would, publishes a mic-like
48 kHz track (TTS'd caller speech, silence otherwise), listens to the agent, and measures what a
listener hears: end of caller speech -> agent's first audio, per turn. Drives the full pipeline
(Deepgram STT -> model -> tools -> Cartesia TTS) without a human. Used for G1 and the load tests.

  uv run python scripts/sim_caller.py --prepare                # synthesize and cache the caller lines
  uv run python scripts/sim_caller.py --calls 10               # 10 sequential calls, fixed Dana script
  uv run python scripts/sim_caller.py --calls 8 --concurrency 4
  uv run python scripts/sim_caller.py --calls 3 --script llm   # hosted model plays the caller, adaptive

Caller-side events go to turn_events under the room name (call_id): caller_speech_end, caller_first_audio,
caller_summary; the agent writes its own stages for the same call_id, so scripts/g1_report.py joins both.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

from services.common.events import PostgresSink
from services.sim.naming import room_name as make_room_name
from services.sim.recorder import CallRecorder
from services.sim.timeline import CallerTimeline, pct

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SR, FRAME_MS = 48000, 10
SAMPLES = SR * FRAME_MS // 1000
CACHE = pathlib.Path(os.environ.get("R3_ARTIFACTS", "~/r3-artifacts")).expanduser() / "sim_audio"
CALLER_VOICE = os.environ.get("SIM_CALLER_VOICE", "47c38ca4-5f35-497b-b1a3-415245fb35e1")  # Cartesia "Daniel"; the agent is Skylar
DEEPGRAM_CALLER_VOICE = os.environ.get("SIM_DEEPGRAM_VOICE", "aura-2-orion-en")  # the fallback caller voice; the agent speaks aura-2-thalia-en
_tts_fallback_announced = False
AGENT_STATE_ATTR = "lk.agent.state"
from services.demo.caller_script import role_for  # noqa: E402

SEED = json.loads((ROOT / "eval/sandbox/seed.json").read_text())

# Dana (C-1001) books a mortgage consultation; the agent's greeting asks for the phone number first.
DANA_FIXED = [
    "five five five, zero one four, two two one nine",
    "march twelfth, nineteen eighty eight",
    "i'd like to book a mortgage consultation for tuesday morning at the market street branch",
    "nine thirty is fine",
    "yes, that's right",
    "no, that's everything. thank you, goodbye",
]
DANA_GOAL = ("You are Dana Whitfield calling Marlow Mutual Bank. Your phone number is 555-014-2219 (say it as digits: 'five five five, "
             "zero one four, two two one nine'), your date of birth is March 12, 1988. You want to book a mortgage consultation on "
             "Tuesday morning (November 3rd) at the Market Street branch; take the earliest time the agent offers at or after nine thirty, "
             "accept the read-back with a clear 'yes', then end the call with a short goodbye. Give exactly one piece of information per turn, "
             "answer only what was asked, speak like a person on the phone (no lists, no formatting, at most 25 words). If the agent says the "
             "booking is confirmed, say thank you and goodbye. If the agent transfers you or says goodbye, reply only: goodbye.")


def now_ms() -> int:
    return time.time_ns() // 1_000_000


# ---- caller audio -------------------------------------------------------------------------------------------
def _post(url: str, **kw):  # one seam for tests
    return httpx.post(url, **kw)


def _synth_cartesia(text: str, voice: str) -> bytes:
    h = {"X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": os.environ.get("CARTESIA_VERSION", "2026-08-14")}
    body = {"model_id": os.environ.get("CARTESIA_MODEL", "sonic-3"), "transcript": text, "voice": {"mode": "id", "id": voice},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": SR}, "language": "en"}
    r = _post("https://api.cartesia.ai/tts/bytes", headers=h, json=body, timeout=60)
    r.raise_for_status()
    return r.content


def _synth_deepgram(text: str, voice: str) -> bytes:
    """Deepgram Aura: raw linear16 at SR (the agent uses the same vendor since Cartesia ran out of credits, Nov 1 04:24)."""
    r = _post("https://api.deepgram.com/v1/speak", params={"model": voice, "encoding": "linear16", "sample_rate": SR, "container": "none"},
              headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": "application/json"}, json={"text": text}, timeout=60)
    r.raise_for_status()
    return r.content


def _cache_file(voice: str, text: str) -> pathlib.Path:
    d = CACHE / voice
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(text.encode()).hexdigest()[:16] + ".pcm")


def tts_cached(text: str, voice: str = CALLER_VOICE) -> np.ndarray:
    """Caller audio as raw pcm_s16le 48 kHz mono, cached by voice + text hash. SIM_TTS=cartesia|deepgram|auto (default auto:
    Cartesia first so cached lines keep their voice; on a Cartesia error Deepgram speaks the line, cached under its own voice)."""
    global _tts_fallback_announced
    vendor = os.environ.get("SIM_TTS", "auto")
    f = _cache_file(DEEPGRAM_CALLER_VOICE if vendor == "deepgram" else voice, text)
    if vendor == "auto" and not f.exists() and _cache_file(DEEPGRAM_CALLER_VOICE, text).exists():
        f = _cache_file(DEEPGRAM_CALLER_VOICE, text)  # already spoken through the fallback: no second Cartesia attempt
    if not f.exists():
        if vendor == "deepgram":
            f.write_bytes(_synth_deepgram(text, DEEPGRAM_CALLER_VOICE))
        else:
            try:
                f.write_bytes(_synth_cartesia(text, voice))
            except httpx.HTTPError as e:
                if vendor != "auto" or not os.environ.get("DEEPGRAM_API_KEY"):
                    raise
                if not _tts_fallback_announced:
                    print(f"sim caller TTS: Cartesia failed ({e}); speaking new lines through Deepgram {DEEPGRAM_CALLER_VOICE}", flush=True)
                    _tts_fallback_announced = True
                f = _cache_file(DEEPGRAM_CALLER_VOICE, text)
                if not f.exists():
                    f.write_bytes(_synth_deepgram(text, DEEPGRAM_CALLER_VOICE))
    pcm = np.frombuffer(f.read_bytes(), dtype=np.int16)
    # trim the trailing TTS silence so "speech end" means the last audible sample, then keep 120 ms of tail
    thr = 200
    nz = np.nonzero(np.abs(pcm.astype(np.int32)) > thr)[0]
    if len(nz):
        pcm = pcm[: min(len(pcm), int(nz[-1]) + SR * 120 // 1000)]
    return pcm


class MicPump:
    """Publishes 10 ms frames continuously like a real microphone: speech when queued, silence otherwise."""

    def __init__(self, source: rtc.AudioSource) -> None:
        self.source = source
        self.queue: list[np.ndarray] = []
        self.last_speech_capture_ms: int | None = None
        self._speaking = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def speak(self, pcm: np.ndarray) -> int:
        """Queue an utterance; returns the wall-clock ms when its last frame was captured."""
        pad = np.zeros(SAMPLES * 30, dtype=np.int16)  # 300 ms of lead-in silence, like drawing breath
        buf = np.concatenate([pad, pcm])
        frames = [buf[i:i + SAMPLES] for i in range(0, len(buf) - len(buf) % SAMPLES, SAMPLES)]
        self._speaking.clear()
        self.first_audible_capture_ms = None
        self.queue.extend(frames)
        await self._speaking.wait()
        assert self.last_speech_capture_ms is not None
        return self.last_speech_capture_ms

    async def _run(self) -> None:
        silence = np.zeros(SAMPLES, dtype=np.int16)
        t_next = time.perf_counter()
        while True:
            if self.queue:
                chunk = self.queue.pop(0)
                is_last = not self.queue
            else:
                chunk, is_last = silence, False
            frame = rtc.AudioFrame(data=chunk.tobytes(), sample_rate=SR, num_channels=1, samples_per_channel=len(chunk))
            if chunk is not silence and getattr(self, "first_audible_capture_ms", None) is None and float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) >= 300:
                self.first_audible_capture_ms = now_ms()
            if chunk is not silence and getattr(self, "on_frame", None) is not None:
                self.on_frame(now_ms(), chunk)
            await self.source.capture_frame(frame)
            if is_last:
                self.last_speech_capture_ms = now_ms()
                self._speaking.set()
            t_next += FRAME_MS / 1000
            delay = t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                t_next = time.perf_counter()


# ---- caller brains -----------------------------------------------------------------------------------------
class FixedScript:
    def __init__(self, lines: list[str]) -> None:
        self.lines, self.i = lines, 0

    async def next_line(self, transcript: list[tuple[str, str]]) -> str | None:
        if self.i >= len(self.lines):
            return None
        line = self.lines[self.i]
        self.i += 1
        return line


class LLMCaller:
    """The hosted model plays the caller from the goal and the running transcript (seeded, no temperature)."""

    def __init__(self, goal: str, seed: int, max_turns: int = 12) -> None:
        from openai import AsyncOpenAI
        self.goal, self.seed, self.max_turns, self.turns = goal, seed, max_turns, 0
        self.client = AsyncOpenAI(base_url=os.environ["HOSTED_LLM_BASE_URL"], api_key=os.environ["HOSTED_LLM_API_KEY"])
        self.model = os.environ["HOSTED_LLM_MODEL"]

    async def next_line(self, transcript: list[tuple[str, str]]) -> str | None:
        if self.turns >= self.max_turns:
            return None
        last_agent = next((t for r, t in reversed(transcript) if r == "agent"), "")
        if self.turns > 0 and any(w in last_agent.lower() for w in ("goodbye", "transferring", "connect you", "have a great day")):
            self.turns += 1
            return "goodbye" if "goodbye" not in [t for r, t in transcript if r == "caller"][-1:] else None
        msgs = [{"role": "system", "content": self.goal + " Reply with the caller's next spoken words only."}]
        for role, text in transcript:
            msgs.append({"role": "assistant" if role == "caller" else "user", "content": text})
        if not transcript:
            msgs.append({"role": "user", "content": "(the line is ringing; the agent will greet you)"})
        # the hosted model's hidden reasoning consumes the token budget (a 60-token budget returned finish_reason=length
        # and empty content on Nov 1), so the budget is generous and an empty reply is retried once with a larger one
        line = ""
        for budget in (400, 1500):
            r = await self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=budget, seed=self.seed)
            line = (r.choices[0].message.content or "").strip().strip('"')
            if line:
                break
        self.turns += 1
        return line or None


# ---- one call ------------------------------------------------------------------------------------------------
def token(room: str, identity: str = "caller") -> str:
    return (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]).with_identity(identity).with_name(identity.title())
            .with_ttl(dt.timedelta(minutes=30)).with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)).to_jwt())


async def run_call(room_name: str, brain, args, sink: PostgresSink | None, model_id: str) -> dict[str, Any]:
    tl = CallerTimeline(silence_rms=args.silence_rms)
    transcript: list[tuple[str, str]] = []
    room = rtc.Room()
    agent_joined: asyncio.Future = asyncio.get_running_loop().create_future()
    audio_tasks: list[asyncio.Task] = []
    t_connect = now_ms()
    result: dict[str, Any] = {"call_id": room_name, "ok": False, "turns": 0, "notes": []}
    rec = CallRecorder(SR) if getattr(args, "record_dir", None) else None
    frames_dump: list | None = [] if args.dump_frames else None
    caller_starts: list[int] = []

    async def emit(event: str, payload: dict, turn_idx: int = 0) -> None:
        ev = {"call_id": room_name, "turn_idx": turn_idx, "model_id": model_id, "event": event, "ts_ms": now_ms(), "payload": payload}
        if sink is not None:
            try:
                await sink(ev)
            except Exception as e:  # measurement must not kill the call
                result["notes"].append(f"sink: {e!r}")

    @room.on("participant_connected")
    def _joined(p: rtc.RemoteParticipant) -> None:
        if not agent_joined.done():
            agent_joined.set_result((p.identity, now_ms() - t_connect))
        state = p.attributes.get(AGENT_STATE_ATTR)
        if state:
            tl.agent_state(now_ms(), state)

    @room.on("participant_attributes_changed")
    def _attrs(changed: dict, p: rtc.RemoteParticipant) -> None:
        if AGENT_STATE_ATTR in changed:
            tl.agent_state(now_ms(), changed[AGENT_STATE_ATTR])

    @room.on("track_subscribed")
    def _sub(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            async def listen() -> None:
                stream = rtc.AudioStream(track, sample_rate=SR, num_channels=1)
                async for ev in stream:
                    x = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32)
                    rms = float(math.sqrt(np.mean(x * x))) if len(x) else 0.0
                    ts = now_ms()
                    tl.agent_audio(ts, rms)
                    if rec is not None:
                        rec.add("agent", ts, np.frombuffer(ev.frame.data, dtype=np.int16))
                    if frames_dump is not None:
                        frames_dump.append((ts, len(x), int(rms)))
            audio_tasks.append(asyncio.create_task(listen()))

    # Agents 1.x publishes transcripts as text streams on lk.transcription (legacy transcription_received is gone).
    own_track_sids: set[str] = set()  # filled once the mic track is published; a transcript of one of these tracks is the caller's

    def _on_text(reader, participant_identity: str) -> None:
        async def read() -> None:
            try:
                text = (await reader.read_all()).strip()
            except Exception:
                return
            attrs = reader.info.attributes or {}
            if not text or attrs.get("lk.transcription_final", "true") == "false":
                return
            transcript.append((role_for(attrs, own_track_sids), text))
        asyncio.create_task(read())

    room.register_text_stream_handler("lk.transcription", _on_text)

    listener: rtc.Room | None = None
    heard_by_listener: list[int] = []  # ts of the first audible frame of each caller utterance, as a second client hears it
    if args.listener:
        listener = rtc.Room()
        lst_state = {"quiet_since": None}

        @listener.on("track_subscribed")
        def _lsub(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO and p.identity == "caller":
                async def llisten() -> None:
                    stream = rtc.AudioStream(track, sample_rate=SR, num_channels=1)
                    last_audible = None
                    async for ev in stream:
                        x = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32)
                        rms = float(math.sqrt(np.mean(x * x))) if len(x) else 0.0
                        ts = now_ms()
                        if rms >= args.silence_rms:
                            if last_audible is None or ts - last_audible > 1500:  # a new utterance after >1.5 s of silence
                                heard_by_listener.append(ts)
                            last_audible = ts
                audio_tasks.append(asyncio.create_task(llisten()))

        await listener.connect(os.environ["LIVEKIT_URL"], token(room_name, identity="listener"), options=rtc.RoomOptions(auto_subscribe=True))

    await room.connect(os.environ["LIVEKIT_URL"], token(room_name), options=rtc.RoomOptions(auto_subscribe=True))
    source = rtc.AudioSource(SR, 1, queue_size_ms=100)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    pub = await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    own_track_sids.add(pub.sid)
    pump = MicPump(source)
    if rec is not None:
        pump.on_frame = lambda ts, chunk: rec.add("caller", ts, chunk)
    pump.start()
    try:
        try:
            identity, dispatch_ms = await asyncio.wait_for(asyncio.shield(agent_joined), args.dispatch_timeout)
        except asyncio.TimeoutError:
            result["notes"].append("no agent joined"); return result
        result["dispatch_ms"] = dispatch_ms
        await emit("caller_agent_joined", {"ms": dispatch_ms, "identity": identity})

        async def wait_listening(timeout: float) -> bool:
            """Wait until the agent is listening AND its audio has been silent for a moment: the state attribute
            flips to listening before the last buffered audio has played out, and talking over the agent's tail
            registers as an interruption that can lose the caller's first words."""
            t_end = time.perf_counter() + timeout
            while time.perf_counter() < t_end:
                if tl.agent_quiet(now_ms(), quiet_ms=args.quiet_ms):
                    return True
                await asyncio.sleep(0.02)
            return False

        turn = 0
        # a caller lets the greeting finish before speaking: wait for the agent to have said something
        t_end = time.perf_counter() + args.reply_timeout
        while tl.last_audible_ms is None and time.perf_counter() < t_end:
            await asyncio.sleep(0.02)
        if tl.last_audible_ms is None:
            result["notes"].append("no greeting heard"); return result
        while True:
            interrupt_info = None
            if args.interrupt_turn and turn + 1 == args.interrupt_turn:
                # cut into the agent's reply once it has been audibly speaking for interrupt_after_ms
                t_end = time.perf_counter() + args.reply_timeout
                while tl.agent_audible_for_ms(now_ms()) < args.interrupt_after_ms and time.perf_counter() < t_end:
                    await asyncio.sleep(0.02)
                spoke_before = tl.agent_audible_for_ms(now_ms())
                interrupt_info = {"turn": turn + 1, "agent_spoke_ms_before": spoke_before, "transcript_idx": len(transcript)}
                if spoke_before < args.interrupt_after_ms:
                    result["notes"].append(f"turn {turn + 1}: agent reply too short to interrupt ({spoke_before} ms)")
            else:
                # quiet -> human pause -> still quiet (the agent may have started a new sentence meanwhile)
                while True:
                    if not await wait_listening(args.reply_timeout):
                        result["notes"].append(f"turn {turn}: agent never finished speaking / never replied"); return result
                    await asyncio.sleep(args.pause)
                    if tl.agent_quiet(now_ms(), quiet_ms=args.quiet_ms):
                        break
            line = await brain.next_line(transcript)
            if line is None:
                break
            turn += 1
            result["turns"] = turn
            pcm = await asyncio.to_thread(tts_cached, line)

            async def watch_stop() -> int | None:
                """ms from the caller's first audible frame to the agent's last audible frame (a 400 ms gap = stopped)."""
                while pump.first_audible_capture_ms is None:
                    await asyncio.sleep(0.01)
                onset = pump.first_audible_capture_ms
                t_end = time.perf_counter() + 20
                while time.perf_counter() < t_end:
                    la = tl.last_audible_ms
                    if la is not None and now_ms() - la > 400:
                        return max(0, la - onset)
                    await asyncio.sleep(0.02)
                return None

            stop_task = asyncio.create_task(watch_stop()) if interrupt_info else None
            eos = await pump.speak(pcm)
            if stop_task is not None:
                interrupt_info["agent_stop_ms"] = await stop_task
                interrupt_info["line"] = line
                result["interruption"] = interrupt_info
                await emit("caller_interrupted", {k: v for k, v in interrupt_info.items() if k != "transcript_idx"}, turn)
            tl.caller_speech_end(eos)
            if pump.first_audible_capture_ms is not None:
                caller_starts.append(pump.first_audible_capture_ms)
                await emit("caller_speech_start", {"turn": turn, "ts_capture_ms": pump.first_audible_capture_ms}, turn)
            transcript.append(("caller", line))
            reask = getattr(brain, "last_reask", None)
            await emit("caller_speech_end", {"turn": turn, "text": line, "audio_ms": int(len(pcm) * 1000 / SR), **({"reask": reask} if reask else {})}, turn)
            if reask:
                result.setdefault("reasks", []).append({"turn": turn, "fact": reask, "line": line})
            # wait for the agent's first audio (or its state to flip) before the next line; the loop top waits for listening
            t_end = time.perf_counter() + args.reply_timeout
            while time.perf_counter() < t_end and tl.turns[-1].first_audio_ms is None and not tl.agent_is_listening():
                await asyncio.sleep(0.02)
            lat = tl.latencies()[-1]
            if lat["eos_to_first_audio_ms"] is not None:
                await emit("caller_first_audio", {"turn": turn, **lat}, turn)
            else:
                await emit("caller_no_reply", {"turn": turn, **lat}, turn)
        result["turns"] = turn
        # let the agent finish its closing line
        await wait_listening(args.reply_timeout)
        await asyncio.sleep(0.5)
        result["ok"] = True
    finally:
        await pump.stop()
        for t in audio_tasks:
            t.cancel()
        summary = tl.summary()
        result["summary"] = summary
        result["latencies"] = tl.latencies()
        result["transcript"] = transcript
        if result.get("interruption"):
            idx = result["interruption"].pop("transcript_idx", 0)
            result["interruption"]["transcript_after"] = transcript[idx:]
        if listener is not None:
            pairs = list(zip(caller_starts, heard_by_listener))
            result["loopback_ms"] = [h - s for s, h in pairs]
            await listener.disconnect()
        if frames_dump is not None:
            result["frames"] = frames_dump
        if rec is not None:
            wav = pathlib.Path(args.record_dir) / f"{room_name}.wav"
            rec.save(wav)
            result["recording"] = str(wav)
        await emit("caller_summary", {**summary, "turns_spoken": result["turns"], "dispatch_ms": result.get("dispatch_ms"), "notes": result["notes"]})
        await room.disconnect()
    return result


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="synthesize and cache the fixed script lines, then exit")
    ap.add_argument("--calls", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--script", choices=["fixed", "llm"], default="fixed")
    ap.add_argument("--prefix", default="r3-sim")
    ap.add_argument("--pause", type=float, default=0.6, help="seconds the caller waits after the agent stops before answering")
    ap.add_argument("--reply-timeout", type=float, default=45.0, help="the base model can talk for 26 s (it reads out every slot)")
    ap.add_argument("--dispatch-timeout", type=float, default=45.0)
    ap.add_argument("--silence-rms", type=float, default=300.0)
    ap.add_argument("--quiet-ms", type=int, default=700, help="agent audio must be silent this long before the caller speaks")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--json", default=None, help="write per-call results to this file")
    ap.add_argument("--lines-file", default=None, help="JSON list of caller lines to speak instead of the fixed Dana script (demo/turns/*.json)")
    ap.add_argument("--facts", default=None, help="JSON {phone,dob,day,time,branch}: the caller repeats a fact the agent asks for again instead of its next line (services/demo/caller_script.py); rows labelled -adaptive")
    ap.add_argument("--interrupt-turn", type=int, default=0, help="caller turn that cuts into the agent's reply instead of waiting for quiet (0 = never)")
    ap.add_argument("--interrupt-after-ms", type=int, default=2000, help="how long the agent must have been audibly speaking before the caller cuts in")
    ap.add_argument("--listener", action="store_true", help="second participant measures capture->heard delay of the caller's own speech (transport + receive buffer)")
    ap.add_argument("--record-dir", default=None, help="write a two-channel wav per call (left caller, right agent) into this directory")
    ap.add_argument("--dump-frames", action="store_true", help="record (ts_ms, samples, rms) of every agent audio frame into the JSON")
    args = ap.parse_args()

    if args.prepare:
        lines = json.loads(pathlib.Path(args.lines_file).read_text()) if args.lines_file else DANA_FIXED
        for line in lines:
            pcm = tts_cached(line)
            print(f"  cached {len(pcm) / SR:5.2f}s  {line}")
        return 0

    sink = None if args.no_db or not os.environ.get("DATABASE_URL") else PostgresSink(os.environ["DATABASE_URL"])
    model_id = os.environ.get("LLM_MODEL", "r3-voice-base")
    batch = time.strftime("%H%M%S")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i: int) -> dict:
        async with sem:
            if args.lines_file and args.facts:
                from services.demo.caller_script import AnsweringScript
                brain = AnsweringScript(json.loads(pathlib.Path(args.lines_file).read_text()), json.loads(pathlib.Path(args.facts).read_text()))
            elif args.lines_file:
                brain = FixedScript(json.loads(pathlib.Path(args.lines_file).read_text()))
            else:
                brain = FixedScript(DANA_FIXED) if args.script == "fixed" else LLMCaller(DANA_GOAL, seed=i)
            room_name = make_room_name(args.prefix, batch, i + 1)  # keeps a --fault-... spec at the end, where the agent parses it
            t0 = time.perf_counter()
            try:
                r = await run_call(room_name, brain, args, sink, model_id)
            except Exception as e:  # e.g. the LLM caller's hosted model refusing (credits): end this call, keep the batch
                r = {"call_id": room_name, "ok": False, "turns": 0, "notes": [f"call aborted: {e!r}"[:200]], "transcript": []}
            r["wall_s"] = round(time.perf_counter() - t0, 1)
            s = r.get("summary", {}).get("eos_to_first_audio_ms")
            print(f"[{room_name}] ok={r['ok']} turns={r['turns']} dispatch={r.get('dispatch_ms')} ms  first-audio p50={s['p50']:.0f} max={s['max']:.0f} ms" + (f"  notes={r['notes']}" if r['notes'] else "") if s else f"[{room_name}] ok={r['ok']} turns={r['turns']} notes={r['notes']}", flush=True)
            return r

    results = await asyncio.gather(*(one(i) for i in range(args.calls)))
    all_lat = [l["eos_to_first_audio_ms"] for r in results for l in r.get("latencies", []) if l["eos_to_first_audio_ms"] is not None]
    print(f"\ncalls={len(results)} ok={sum(r['ok'] for r in results)} concurrency={args.concurrency} script={args.script}")
    if all_lat:
        print(f"end of caller speech -> agent first audio: p50 {pct(all_lat, 0.5):.0f} ms  p95 {pct(all_lat, 0.95):.0f} ms  max {max(all_lat):.0f} ms  n={len(all_lat)}  (target p50 ≤ 1000, p95 ≤ 1600)")
    for r in results:
        if r.get("interruption"):
            i = r["interruption"]
            print(f"interruption [{r['call_id']}]: caller cut in after the agent had spoken {i['agent_spoke_ms_before']} ms; agent audio stopped {i.get('agent_stop_ms')} ms after the caller's onset; "
                  f"heard after it: {[t for role, t in i.get('transcript_after', []) if role == 'caller'][:3]}")
    disp = [r["dispatch_ms"] for r in results if r.get("dispatch_ms") is not None]
    if disp:
        print(f"agent joined after: p50 {pct(disp, 0.5):.0f} ms  max {max(disp):.0f} ms")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2, default=str))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
