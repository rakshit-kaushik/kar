#!/usr/bin/env python
"""List LiveKit rooms and participants; --busy exits 0 if any room has a non-agent participant (a caller), else 1.
Used by scripts/agent_watchdog.sh so the agent is never restarted under a live caller."""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

from dotenv import load_dotenv
from livekit import api

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
AGENT_KIND = 4


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--busy", action="store_true")
    args = ap.parse_args()
    lk = api.LiveKitAPI(os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    try:
        rooms = (await lk.room.list_rooms(api.ListRoomsRequest())).rooms
        busy = False
        for r in rooms:
            parts = (await lk.room.list_participants(api.ListParticipantsRequest(room=r.name))).participants
            callers = [p.identity for p in parts if p.kind != AGENT_KIND]
            agents = [p.identity for p in parts if p.kind == AGENT_KIND]
            busy = busy or bool(callers)
            if not args.busy:
                print(f"{r.name}: callers={callers} agents={agents}")
        if args.busy:
            print("busy" if busy else "idle")
            return 0 if busy else 1
        if not rooms:
            print("no rooms")
        return 0
    finally:
        await lk.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
