#!/usr/bin/env python
"""Dispatch probe: join a fresh room as a caller (no audio), confirm the agent worker is dispatched,
joins, and publishes its audio track (the greeting). Measures both delays. Use before handing anyone
a call link, and after any agent restart; a stale worker connection shows up here as a timeout.

Usage: uv run python scripts/probe_dispatch.py [--room probe-<ts>] [--timeout 20] [--keep]
Exit 0 = agent joined and published audio within the timeout.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import pathlib
import sys
import time

from dotenv import load_dotenv
from livekit import api, rtc

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def token(room: str) -> str:
    return (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity("probe").with_name("Probe").with_ttl(dt.timedelta(minutes=10))
            .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)).to_jwt())


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default=f"probe-{int(time.time())}")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--keep", action="store_true", help="stay in the room until Ctrl-C (lets you watch the agent's log)")
    args = ap.parse_args()

    room = rtc.Room()
    joined: asyncio.Future = asyncio.get_running_loop().create_future()
    audio: asyncio.Future = asyncio.get_running_loop().create_future()
    t0 = time.perf_counter()

    @room.on("participant_connected")
    def _on_join(p: rtc.RemoteParticipant) -> None:
        if not joined.done():
            joined.set_result((p.identity, time.perf_counter() - t0))

    @room.on("track_published")
    def _on_pub(pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant) -> None:
        if pub.kind == rtc.TrackKind.KIND_AUDIO and not audio.done():
            audio.set_result(time.perf_counter() - t0)

    waiting = True

    @room.on("disconnected")
    def _on_disc(reason) -> None:
        if waiting:
            for f in (joined, audio):
                if not f.done():
                    f.set_exception(RuntimeError(f"probe disconnected: {reason}"))

    await room.connect(os.environ["LIVEKIT_URL"], token(args.room))
    print(f"probe connected to room {args.room} in {time.perf_counter() - t0:.2f}s; waiting for the agent (timeout {args.timeout:.0f}s)")
    try:
        identity, t_join = await asyncio.wait_for(asyncio.shield(joined), args.timeout)
        print(f"agent joined: identity={identity} after {t_join:.2f}s")
        t_audio = await asyncio.wait_for(asyncio.shield(audio), args.timeout)
        print(f"agent published audio (greeting) after {t_audio:.2f}s")
        ok = True
    except (asyncio.TimeoutError, RuntimeError) as e:
        print(f"FAIL: {e if str(e) else 'no agent within the timeout'}")
        ok = False
    waiting = False
    for f in (joined, audio):
        if f.done() and not f.cancelled():
            f.exception()  # mark retrieved
    if args.keep and ok:
        print("holding the room; Ctrl-C to leave")
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    await room.disconnect()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
