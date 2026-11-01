"""Twenty-example synthetic voice dataset for dry runs (mlx_lm `messages` + `tools` format).

Not training data. Session E generates the real data at T+5 to T+9; this exists so the loader,
driver, fuse and eval hooks can be exercised end to end on CPU before the T+7 window. Shapes follow
CONTRACT.md §1-§3: STT-shaped caller turns, nothing account-specific before `lookup_account`
returns verified, read-back before every write, tool calls as OpenAI `tool_calls` with JSON-string
arguments, tool results as `tool` messages, at most three targets per conversation (a tool call,
a clarification, the closing turn), each example = prefix -> that assistant turn.

Measured on Qwen3-4B-8bit: the eight tool schemas render to ~1,470 tokens and the system prompt
to ~70, leaving ~500 tokens of dialogue under seq 2048, so these conversations are compressed
(one read-back per booking, no slot-selection exchange) and two scenarios keep only two targets.

    uv run python -m train.synth --out $R3_ARTIFACTS/datasets/synth20
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import uuid
from pathlib import Path

from train.env import repo_root

TOOLS_DIR = repo_root() / "schemas" / "tools"
TODAY = dt.date(2026, 10, 31)

SYSTEM_PROMPT = (
    "You are Marlow Mutual Bank's phone line. One or two short sentences, one question per turn. "
    "Nothing account-specific before lookup_account verifies the caller. Read back and get a yes "
    "before any write. Say numbers and times in words; never mention tools or JSON. Hand off on "
    "fraud, two failed verifications, or a second ask for a person."
)

BRANCHES = {
    "market_st": "Market Street",
    "marlow_st": "Marlow Street",
    "marker_st": "Marker Street",
    "mill_rd": "Mill Road",
}
SERVICES = {
    "account_opening": "account opening",
    "mortgage_consultation": "mortgage consultation",
    "loan_application": "loan application",
    "financial_review": "financial review",
    "notary": "notary appointment",
    "safe_deposit_box": "safe deposit box visit",
}
ASKS = {
    "account_opening": "i want to open a new account",
    "mortgage_consultation": "i need to see someone about a mortgage",
    "loan_application": "i need to apply for a car loan",
    "financial_review": "i'd like to sit down for a financial review",
    "notary": "i need something notarized",
    "safe_deposit_box": "i need to get into my safe deposit box",
}
FIRST_NAMES = ["Dana", "Priya", "Marcus", "Elena", "Tom", "Aisha", "Jonah", "Mei", "Luis", "Grace"]
DOBS = [
    ("1988-03-12", "march twelfth nineteen eighty eight"),
    ("1975-07-04", "july fourth nineteen seventy five"),
    ("1992-11-30", "november thirtieth nineteen ninety two"),
    ("1969-01-21", "january twenty first nineteen sixty nine"),
    ("1984-09-09", "september ninth nineteen eighty four"),
    ("1990-05-16", "may sixteenth nineteen ninety"),
]
DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
    8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
    14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 21: "twenty first", 22: "twenty second", 23: "twenty third",
    24: "twenty fourth", 25: "twenty fifth", 26: "twenty sixth", 27: "twenty seventh",
    28: "twenty eighth", 29: "twenty ninth", 30: "thirtieth", 31: "thirty first",
}
MONTHS = {11: "November", 12: "December"}
TIMES = {
    "09:00": ("nine in the morning", "nine"),
    "09:30": ("nine thirty in the morning", "nine thirty"),
    "10:00": ("ten in the morning", "ten"),
    "11:00": ("eleven in the morning", "eleven"),
    "14:00": ("two in the afternoon", "two"),
    "14:30": ("two thirty in the afternoon", "two thirty"),
    "15:00": ("three in the afternoon", "three"),
}


def load_tools() -> list[dict]:
    """The eight contract tools, sorted by name (schemas/tools/*.json is the source of truth)."""
    return [json.loads(p.read_text()) for p in sorted(TOOLS_DIR.glob("*.json"))]


def spoken_digits(s: str) -> str:
    return " ".join(DIGITS[c] for c in s)


def spoken_date(d: dt.date) -> str:
    return f"{d.strftime('%A')}, {MONTHS[d.month]} {ORDINALS[d.day]}"


def spoken_time(hhmm: str) -> str:
    return TIMES[hhmm][0]


def short_time(hhmm: str) -> str:
    return TIMES[hhmm][1]


def _weekday(rng: random.Random, offset_days: int = 3, span: int = 12) -> dt.date:
    while True:
        d = TODAY + dt.timedelta(days=rng.randint(offset_days, offset_days + span))
        if d.weekday() < 5:
            return d


def _ikey(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _call_id(rng: random.Random) -> str:
    return f"call_{rng.getrandbits(32):08x}"


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def tool_calls(rng: random.Random, *calls: tuple[str, dict], content: str = "") -> dict:
    """An assistant turn made of tool calls. `content` is a string, never null: the Qwen3 chat
    template runs string operations on assistant content."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": _call_id(rng),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, separators=(",", ":"))},
            }
            for name, args in calls
        ],
    }


def tool_results(call_turn: dict, *payloads: dict) -> list[dict]:
    return [
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(payload, separators=(",", ":")),
        }
        for call, payload in zip(call_turn["tool_calls"], payloads)
    ]


class Caller:
    def __init__(self, rng: random.Random, appointments: list[dict] | None = None, with_cards: bool = False):
        self.name = rng.choice(FIRST_NAMES)
        self.phone = "555" + "".join(rng.choice("0123456789") for _ in range(7))
        self.dob, self.dob_spoken = rng.choice(DOBS)
        self.customer_id = f"C-{rng.randint(1000, 9999)}"
        self.branch = rng.choice(list(BRANCHES))
        self.last4 = "".join(rng.choice("0123456789") for _ in range(4))
        self.appointments = appointments or []
        self.with_cards = with_cards

    def profile(self) -> dict:
        out = {
            "verified": True,
            "customer_id": self.customer_id,
            "first_name": self.name,
            "branch_id": self.branch,
            "appointments": self.appointments,
        }
        if self.with_cards:
            out["cards"] = [{"card_last4": self.last4, "status": "active"}]
        return out


def _opening(rng: random.Random, c: Caller, lookup_result: dict) -> list[dict]:
    lookup = tool_calls(rng, ("lookup_account", {"phone": c.phone, "date_of_birth": c.dob}))
    return [
        assistant("Thanks for calling Marlow Mutual. What's the phone number on your account?"),
        user(f"yeah it's {spoken_digits(c.phone)}"),
        assistant("Thanks. And your date of birth?"),
        user(c.dob_spoken),
        lookup,
        *tool_results(lookup, lookup_result),
    ]


def _verified_opening(rng: random.Random, c: Caller) -> list[dict]:
    return _opening(rng, c, c.profile()) + [
        assistant(f"Thanks {c.name}, you're verified. How can I help?")
    ]


# Each scenario returns (turns, targets) where targets = [(index into turns, kind)], <= 3 per conversation.

def ordinary_booking(rng: random.Random):
    c = Caller(rng)
    service = rng.choice([s for s in SERVICES if s != "notary"])
    day = _weekday(rng)
    slots = sorted(rng.sample(["09:00", "09:30", "10:00", "11:00"], 3))
    pick = rng.choice(slots)
    branch = c.branch
    turns = _verified_opening(rng, c)
    turns += [
        user(f"{ASKS[service]} next week"),
        assistant("Happy to. Which day next week works for you?"),
        user(f"{day.strftime('%A').lower()} morning {BRANCHES[branch].lower()} is fine"),
    ]
    avail = tool_calls(rng, ("check_availability", {"branch_id": branch, "service_type": service, "date": day.isoformat()}))
    turns += [avail, *tool_results(avail, {"slots": slots})]
    turns += [
        assistant(f"{spoken_time(pick).capitalize()} is open. So that's a {SERVICES[service]} at {BRANCHES[branch]} on {spoken_date(day)} at {spoken_time(pick)}. Shall I book it?"),
        user("yes"),
    ]
    book = tool_calls(rng, ("book_appointment", {
        "customer_id": c.customer_id, "branch_id": branch, "service_type": service,
        "slot_start": f"{day.isoformat()}T{pick}", "idempotency_key": _ikey(rng)}))
    appt = f"A-{rng.randint(1000, 9999)}"
    turns += [book, *tool_results(book, {"appointment_id": appt})]
    turns += [
        assistant(f"Booked. See you {day.strftime('%A')} at {short_time(pick)}, {c.name}. Goodbye."),
    ]
    targets = [(turns.index(book), "tool_call"), (len(turns) - 1, "closing")]
    clar = next(i for i, t in enumerate(turns) if t.get("content", "").startswith("Happy to."))
    return "ordinary", turns, [(clar, "clarification")] + targets


def correction(rng: random.Random):
    c = Caller(rng)
    day = _weekday(rng)
    turns = _verified_opening(rng, c)
    turns += [
        user(f"book me a notary appointment {day.strftime('%A').lower()} at two at marlow street"),
        assistant("Just to check, did you mean Marlow Street or Market Street?"),
        user("marlow street"),
    ]
    avail = tool_calls(rng, ("check_availability", {"branch_id": "marlow_st", "service_type": "notary", "date": day.isoformat()}))
    turns += [avail, *tool_results(avail, {"slots": ["14:00", "14:30"]})]
    turns += [
        assistant(f"Two in the afternoon is open. A notary appointment at Marlow Street on {spoken_date(day)} at two. Shall I book it?"),
        user("actually make it two thirty"),
        assistant(f"Two thirty it is. Notary at Marlow Street, {spoken_date(day)} at two thirty in the afternoon. Book it?"),
        user("yes"),
    ]
    book = tool_calls(rng, ("book_appointment", {
        "customer_id": c.customer_id, "branch_id": "marlow_st", "service_type": "notary",
        "slot_start": f"{day.isoformat()}T14:30", "idempotency_key": _ikey(rng)}))
    turns += [book, *tool_results(book, {"appointment_id": f"A-{rng.randint(1000, 9999)}"})]
    turns += [
        assistant(f"Booked for two thirty at Marlow Street. Thanks {c.name}, goodbye."),
    ]
    clar = next(i for i, t in enumerate(turns) if t.get("content", "").startswith("Just to check"))
    return "correction", turns, [(clar, "clarification"), (turns.index(book), "tool_call")]


def lost_card_fraud(rng: random.Random):
    c = Caller(rng, with_cards=True)
    turns = _verified_opening(rng, c)
    turns += [
        user("i lost my card and i see a charge i didn't make"),
        assistant("Which card is it, the last four digits?"),
        user(spoken_digits(c.last4)),
        assistant(f"I'll block the card ending in {spoken_digits(c.last4)} right now. Go ahead?"),
        user("yes please"),
    ]
    block = tool_calls(rng, ("block_card", {
        "customer_id": c.customer_id, "card_last4": c.last4, "reason": "lost", "idempotency_key": _ikey(rng)}))
    turns += [block, *tool_results(block, {"blocked": True})]
    ticket = f"T-{rng.randint(1000, 9999)}"
    summary = f"Lost card {c.last4}, unrecognized charge reported"
    escalate = tool_calls(
        rng,
        ("create_ticket", {"customer_id": c.customer_id, "category": "unauthorized_transaction", "summary": summary, "idempotency_key": _ikey(rng)}),
        ("request_handoff", {"reason": "suspected_fraud", "summary": f"Card {c.last4} blocked, fraud ticket opened"}),
    )
    turns += [escalate, *tool_results(escalate, {"ticket_id": ticket}, {"ok": True})]
    turns += [assistant(f"The card is blocked and I've opened a fraud case, number {spoken_digits(ticket[2:])}. I'm connecting you to our fraud team now, please hold.")]
    return "lost_card_fraud", turns, [(turns.index(block), "tool_call"), (turns.index(escalate), "tool_call")]


def reschedule(rng: random.Random):
    old_day = _weekday(rng, 2, 4)
    new_day = _weekday(rng, 7, 6)
    appt = {"appointment_id": f"A-{rng.randint(1000, 9999)}", "service_type": "notary", "branch_id": "mill_rd", "slot_start": f"{old_day.isoformat()}T14:00"}
    c = Caller(rng, appointments=[appt])
    turns = _verified_opening(rng, c)
    turns += [
        user(f"i need to move my appointment on {old_day.strftime('%A').lower()}"),
        assistant(f"That's your notary at Mill Road on {spoken_date(old_day)} at two. What day would you like instead?"),
        user(f"{new_day.strftime('%A').lower()} morning"),
    ]
    avail = tool_calls(rng, ("check_availability", {"branch_id": "mill_rd", "service_type": "notary", "date": new_day.isoformat()}))
    turns += [avail, *tool_results(avail, {"slots": ["09:00", "10:00"]})]
    turns += [
        assistant(f"Ten in the morning is open. Moving your notary at Mill Road to {spoken_date(new_day)} at ten. Go ahead?"),
        user("yes"),
    ]
    move = tool_calls(rng, ("reschedule_appointment", {"appointment_id": appt["appointment_id"], "new_slot_start": f"{new_day.isoformat()}T10:00", "idempotency_key": _ikey(rng)}))
    turns += [move, *tool_results(move, {"ok": True, "appointment_id": appt["appointment_id"]})]
    turns += [
        assistant(f"Done, it's now {new_day.strftime('%A')} at ten. Thanks {c.name}, goodbye."),
    ]
    clar = next(i for i, t in enumerate(turns) if t.get("content", "").startswith("That's your notary"))
    return "reschedule", turns, [(clar, "clarification"), (turns.index(move), "tool_call"), (len(turns) - 1, "closing")]


def cancel(rng: random.Random):
    day = _weekday(rng)
    appt = {"appointment_id": f"A-{rng.randint(1000, 9999)}", "service_type": "financial_review", "branch_id": "market_st", "slot_start": f"{day.isoformat()}T10:00"}
    c = Caller(rng, appointments=[appt])
    turns = _verified_opening(rng, c)
    turns += [
        user("i want to cancel my appointment next week"),
        assistant(f"That's your financial review at Market Street on {spoken_date(day)} at ten in the morning. Shall I cancel it?"),
        user("yes cancel it"),
    ]
    cxl = tool_calls(rng, ("cancel_appointment", {"appointment_id": appt["appointment_id"], "idempotency_key": _ikey(rng)}))
    turns += [cxl, *tool_results(cxl, {"ok": True})]
    turns += [
        assistant(f"Cancelled. Thanks {c.name}, goodbye."),
    ]
    clar = next(i for i, t in enumerate(turns) if t.get("content", "").startswith("That's your"))
    return "cancel", turns, [(clar, "clarification"), (turns.index(cxl), "tool_call"), (len(turns) - 1, "closing")]


def unknown_account(rng: random.Random):
    c = Caller(rng)
    not_found = {"error": "not_found", "message": "No account matches"}
    turns = _opening(rng, c, not_found)
    turns += [
        assistant("I couldn't find an account with those details. Could you say the phone number once more?"),
    ]
    phone2 = "555" + "".join(rng.choice("0123456789") for _ in range(7))
    turns += [user(spoken_digits(phone2))]
    again = tool_calls(rng, ("lookup_account", {"phone": phone2, "date_of_birth": c.dob}))
    turns += [again, *tool_results(again, not_found)]
    handoff = tool_calls(rng, ("request_handoff", {"reason": "verification_failed", "summary": "Two failed account lookups; nothing verified"}))
    turns += [handoff, *tool_results(handoff, {"ok": True})]
    turns += [assistant("I'm not able to verify the account from here, so I'm connecting you to a banker now. Please hold.")]
    clar = next(i for i, t in enumerate(turns) if t.get("content", "").startswith("I couldn't find"))
    return "unknown_account", turns, [(clar, "clarification"), (turns.index(handoff), "tool_call"), (len(turns) - 1, "closing")]


def no_availability(rng: random.Random):
    c = Caller(rng)
    day = _weekday(rng, 3, 8)
    next_day = day + dt.timedelta(days=1 if day.weekday() < 4 else 3)
    turns = _verified_opening(rng, c)
    turns += [user(f"i'd like to open an account {day.strftime('%A').lower()} at marker street")]
    first = tool_calls(rng, ("check_availability", {"branch_id": "marker_st", "service_type": "account_opening", "date": day.isoformat()}))
    turns += [first, *tool_results(first, {"slots": []})]
    turns += [
        assistant(f"Nothing is open at Marker Street on {day.strftime('%A')}. Would {next_day.strftime('%A')} work instead?"),
        user(f"sure {next_day.strftime('%A').lower()}"),
    ]
    second = tool_calls(rng, ("check_availability", {"branch_id": "marker_st", "service_type": "account_opening", "date": next_day.isoformat()}))
    turns += [second, *tool_results(second, {"slots": ["09:00", "11:00"]})]
    turns += [
        assistant(f"An account opening at Marker Street on {spoken_date(next_day)} at eleven in the morning is open. Shall I book it?"),
        user("yes"),
    ]
    book = tool_calls(rng, ("book_appointment", {
        "customer_id": c.customer_id, "branch_id": "marker_st", "service_type": "account_opening",
        "slot_start": f"{next_day.isoformat()}T11:00", "idempotency_key": _ikey(rng)}))
    turns += [book, *tool_results(book, {"appointment_id": f"A-{rng.randint(1000, 9999)}"})]
    turns += [assistant(f"Booked for {next_day.strftime('%A')} at eleven. Thanks {c.name}, goodbye.")]
    clar = next(i for i, t in enumerate(turns) if t.get("content", "").startswith("Nothing is open"))
    return "no_availability", turns, [(clar, "clarification"), (turns.index(second), "tool_call")]


SCENARIOS = [ordinary_booking, correction, lost_card_fraud, reschedule, cancel, unknown_account, no_availability]


def _conversations(rng: random.Random, tools: list[dict], start: int):
    i = start
    while True:
        scenario, turns, targets = SCENARIOS[i % len(SCENARIOS)](rng)
        conv_id = f"synth-{i:04d}"
        examples = []
        for idx, kind in targets[:3]:
            assert turns[idx]["role"] == "assistant", (scenario, idx)
            examples.append({
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns[: idx + 1],
                "tools": tools,
                "meta": {"conversation_id": conv_id, "scenario": scenario, "target_kind": kind, "target_index": idx},
            })
        yield examples
        i += 1


def build_examples(n_train: int = 20, n_valid: int = 4, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Whole conversations go to one split; the counts are trimmed to exactly n_train / n_valid."""
    rng = random.Random(seed)
    tools = load_tools()
    gen = _conversations(rng, tools, 0)
    train: list[dict] = []
    while len(train) < n_train:
        train += next(gen)
    valid: list[dict] = []
    while len(valid) < n_valid:
        valid += next(gen)
    return train[:n_train], valid[:n_valid]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(out_dir: Path, n_train: int = 20, n_valid: int = 4, seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, valid = build_examples(n_train, n_valid, seed)
    files = {"train": "train.jsonl", "valid": "valid.jsonl"}
    for split, rows in (("train", train), ("valid", valid)):
        with (out_dir / files[split]).open("w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    manifest = {
        "name": out_dir.name,
        "version": "0.1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "train/synth.py",
        "seed": seed,
        "format": "mlx_lm.messages+tools",
        "enable_thinking": False,
        "files": files,
        "sha256": {name: sha256_file(out_dir / name) for name in files.values()},
        "counts": {"train": len(train), "valid": len(valid)},
        "targets_per_conversation": 3,
        "note": "synthetic dry-run data from the contract's example dialogues; never train a candidate on it",
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-valid", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    path = write_dataset(a.out, a.n_train, a.n_valid, a.seed)
    print(f"wrote {a.n_train} train + {a.n_valid} valid examples; manifest {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
