"""Is a caller reply a goodbye and nothing else? Shared by the simulator (a farewell-only reply ends
the call even without the [hangup] marker, which the local base-model caller never emits) and the
analysis side (a caller that keeps saying goodbye is a harness fault, not the agent's)."""

from __future__ import annotations

import re

_FAREWELL = re.compile(
    r"\b(good\s?bye|bye(?:-|\s)?bye|bye|take care|have a (?:good|great|nice|lovely) "
    r"(?:day|one|evening|afternoon|weekend)|talk to you later|see you)\b",
    re.IGNORECASE,
)
_STILL_TALKING = re.compile(
    r"\b(wait|actually|one more|also|before you go|another|question|can you|could you|please|hold on)\b|\?",
    re.IGNORECASE,
)
MAX_FAREWELL_WORDS = 12


def is_farewell(text: str) -> bool:
    """True for a short reply that says goodbye and asks for nothing more."""
    words = (text or "").split()
    if not words or len(words) > MAX_FAREWELL_WORDS:
        return False
    return bool(_FAREWELL.search(text)) and not _STILL_TALKING.search(text)
