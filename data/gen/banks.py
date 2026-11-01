"""Phrasing seeds mined from Taskmaster-1 and SGD appointment-style dialogues (phrasing only).

The bank holds short user utterances (3–18 words) with scheduling vocabulary, lower-cased, with any
long digit runs removed. Personas in generated scenarios quote two seeds so the hosted caller
borrows real callers' phrasing instead of the generator's voice. Utterances used for the human's
test paraphrases never enter this bank (the test set was hand-written, not mined).

    uv run python -m data.gen.banks --sgd $R3_ARTIFACTS/datasets/sgd --tm1 $R3_ARTIFACTS/datasets/taskmaster/TM-1-2019 \
        --out data/gen/phrasing_bank.json --n 600
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

BANK_PATH = Path(__file__).with_name("phrasing_bank.json")
SCHEDULING = re.compile(
    r"\b(appointment|book|booking|schedule|reschedule|cancel|move|slot|time|morning|afternoon|evening|tomorrow|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|next week|this week|o'?clock|am|pm|available|open|earlier|later)\b",
    re.IGNORECASE,
)
TM1_IDS = ("auto-repair", "restaurant-table", "coffee", "pizza", "movie", "uber")
SGD_SERVICES = ("Services_", "Restaurants_", "Events_", "Travel_", "RentalCars_")


def _clean(u: str) -> str | None:
    u = re.sub(r"\s+", " ", u).strip().lower()
    if re.search(r"\d{7,}", u) or "http" in u or "@" in u:
        return None
    n = len(u.split())
    if n < 3 or n > 18 or not SCHEDULING.search(u):
        return None
    return u


def mine_tm1(root: str | Path) -> list[str]:
    out = []
    p = Path(root) / "self-dialogs.json"
    if not p.exists():
        return out
    for d in json.loads(p.read_text()):
        if not any(k in str(d.get("instruction_id", "")) for k in TM1_IDS):
            continue
        for u in d.get("utterances", []):
            if u.get("speaker") == "USER":
                c = _clean(u.get("text", ""))
                if c:
                    out.append(c)
    return out


def mine_sgd(root: str | Path) -> list[str]:
    out = []
    for f in sorted(glob.glob(str(Path(root) / "train" / "dialogues_*.json"))):
        for dlg in json.loads(Path(f).read_text()):
            if not any(s.startswith(SGD_SERVICES) for s in dlg.get("services", [])):
                continue
            for t in dlg.get("turns", []):
                if t.get("speaker") == "USER":
                    c = _clean(t.get("utterance", ""))
                    if c:
                        out.append(c)
    return out


def build_bank(sgd: str | Path, tm1: str | Path, n: int, seed: int = 0) -> list[str]:
    seen = set()
    pool = []
    for u in mine_tm1(tm1) + mine_sgd(sgd):
        if u not in seen:
            seen.add(u)
            pool.append(u)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:n])


def load_bank(path: str | Path = BANK_PATH) -> list[str]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sgd", required=True)
    p.add_argument("--tm1", required=True)
    p.add_argument("--out", default=str(BANK_PATH))
    p.add_argument("--n", type=int, default=600)
    a = p.parse_args(argv)
    bank = build_bank(a.sgd, a.tm1, a.n)
    Path(a.out).write_text(json.dumps(bank, indent=0, ensure_ascii=False) + "\n")
    print(f"{len(bank)} phrasing seeds -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
