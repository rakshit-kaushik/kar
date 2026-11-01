#!/usr/bin/env python
"""Mint a browser join token for a call room and print a one-click meeting link.

Usage: uv run python scripts/join_token.py [room-name]
The agent worker (`services.agent.main dev`) auto-joins every new room in the project.
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys
import time
from urllib.parse import quote

from dotenv import load_dotenv
from livekit import api

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

room = sys.argv[1] if len(sys.argv) > 1 else f"r3-call-{int(time.time()) % 100000}"
token = (
    api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    .with_identity("caller")
    .with_name("Caller")
    .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
    .with_ttl(dt.timedelta(hours=12))
    .to_jwt()
)
url = os.environ["LIVEKIT_URL"]
print(f"room:  {room}")
print(f"url:   {url}")
print(f"token: {token}")
print()
print("open:  " + f"https://meet.livekit.io/custom?liveKitUrl={quote(url, safe='')}&token={token}")
