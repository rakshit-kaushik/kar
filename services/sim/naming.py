"""Room names for simulated calls. A fault spec in the prefix (`--fault-<tool>-<error>`, services/agent/faults.py) must stay
at the end of the room name, after the batch suffix, or the agent's parser does not see it."""
from __future__ import annotations


def room_name(prefix: str, batch: str, i: int) -> str:
    head, sep, tail = prefix.partition("--fault-")
    return f"{head}-{batch}-{i}{sep}{tail}"
