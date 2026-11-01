"""Deterministic STT shaping of caller text: what Deepgram would hand the model.

Always: lowercase, punctuation stripped (apostrophes kept), numbers spoken as words
(phones and card digits digit by digit, years as years, times as "two thirty").
Noise levels 1–3 add fillers, homophone swaps, dropped and repeated words with a seeded RNG,
so the same (text, level, seed) always yields the same transcript.
"""

from __future__ import annotations

import random
import re

ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINAL_ONES = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
}
ORDINAL_TENS = {
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}

FILLERS = ["um", "uh", "like", "you know", "so", "yeah"]
HOMOPHONES = {
    "two": ["to", "too"],
    "to": ["two", "too"],
    "too": ["to", "two"],
    "for": ["four"],
    "four": ["for"],
    "fourth": ["forth"],
    "there": ["their"],
    "their": ["there"],
    "by": ["buy"],
    "right": ["write"],
    "know": ["no"],
    "no": ["know"],
    "won": ["one"],
    "one": ["won"],
    "eight": ["ate"],
    "ate": ["eight"],
    "hear": ["here"],
    "here": ["hear"],
}
RATES = {  # per word
    0: {"filler": 0.0, "homophone": 0.0, "drop": 0.0, "repeat": 0.0},
    1: {"filler": 0.05, "homophone": 0.0, "drop": 0.0, "repeat": 0.0},
    2: {"filler": 0.10, "homophone": 0.08, "drop": 0.02, "repeat": 0.0},
    3: {"filler": 0.15, "homophone": 0.15, "drop": 0.05, "repeat": 0.04},
}


def _under_100(n: int) -> str:
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] if ones == 0 else f"{TENS[tens]} {ONES[ones]}"


def cardinal(n: int) -> str:
    """0..9999 in words."""
    if n < 100:
        return _under_100(n)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = f"{ONES[hundreds]} hundred"
        return head if rest == 0 else f"{head} {_under_100(rest)}"
    thousands, rest = divmod(n, 1000)
    head = f"{_under_100(thousands)} thousand"
    return head if rest == 0 else f"{head} {cardinal(rest)}"


def ordinal(n: int) -> str:
    if n in ORDINAL_ONES:
        return ORDINAL_ONES[n]
    if n in ORDINAL_TENS:
        return ORDINAL_TENS[n]
    tens, ones = divmod(n, 10)
    return f"{TENS[tens]} {ORDINAL_ONES[ones]}"


def year_words(n: int) -> str:
    if 1900 <= n <= 1999:
        rest = n - 1900
        return "nineteen hundred" if rest == 0 else f"nineteen {_under_100(rest)}"
    if 2000 <= n <= 2009:
        return "two thousand" if n == 2000 else f"two thousand {ONES[n - 2000]}"
    return f"twenty {_under_100(n - 2000)}"


def spell_digits(s: str) -> str:
    return " ".join(ONES[int(ch)] for ch in s if ch.isdigit())


def number_words(tok: str) -> str:
    """A run of digits as a caller would say it."""
    if not tok.isdigit():
        return tok
    n = int(tok)
    if len(tok) == 4 and 1900 <= n <= 2099:
        return year_words(n)
    if len(tok) >= 4 or (len(tok) > 1 and tok[0] == "0"):
        return spell_digits(tok)
    return cardinal(n)


def time_words(h: int, m: int) -> str:
    hour = cardinal(h if h <= 12 else h - 12) if h != 0 else "twelve"
    if m == 0:
        return f"{hour} o'clock"
    if m < 10:
        return f"{hour} oh {ONES[m]}"
    return f"{hour} {_under_100(m)}"


_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_ORDINAL = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b")
_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s']")


def normalize(text: str) -> str:
    """Lowercase, numbers to words, punctuation out. No randomness."""
    t = text.lower()
    t = _PHONE.sub(lambda m: spell_digits(m.group(0)), t)
    t = _TIME.sub(lambda m: time_words(int(m.group(1)), int(m.group(2))), t)
    t = _ORDINAL.sub(lambda m: ordinal(int(m.group(1))), t)
    t = _DIGITS.sub(lambda m: number_words(m.group(0)), t)
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)  # stray quotes, keep it's / o'clock
    return re.sub(r"\s+", " ", t).strip()


def shape(text: str, level: int = 0, seed: int = 0) -> str:
    """STT-shaped caller text at a noise level, deterministic for (text, level, seed)."""
    base = normalize(text)
    if level <= 0:
        return base
    rates = RATES[min(level, 3)]
    rng = random.Random(f"{seed}:{text}")
    out: list[str] = []
    for w in base.split():
        if rates["drop"] and rng.random() < rates["drop"] and w not in ONES:
            continue
        if rates["filler"] and rng.random() < rates["filler"]:
            out.append(rng.choice(FILLERS))
        if rates["homophone"] and w in HOMOPHONES and rng.random() < rates["homophone"]:
            w = rng.choice(HOMOPHONES[w])
        out.append(w)
        if rates["repeat"] and rng.random() < rates["repeat"]:
            out.append(w)
    return " ".join(out)
