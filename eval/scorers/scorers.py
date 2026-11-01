"""Deterministic scorers: one per `expect` key in eval/scenarios/schema.json plus the generic
rules that apply to every episode (CONTRACT §1 prohibited list, §2 policy). Never a model.
Each result carries the caller turn (1-based) where it failed, when that is meaningful.
"""

from __future__ import annotations

import re
from collections import Counter

from eval.sandbox import TOOL_NAMES
from eval.sandbox.seed import BRANCHES, DEFAULT_TODAY
from eval.scorers.base import ScorerResult
from eval.scorers.normalize import WRITE_TOOLS, args_match, norm_value, slot_of
from eval.scorers.timeline import Evidence, message_turn, tool_log_turn, walk
from eval.simulator.episode import Episode
from eval.simulator.scenario import Scenario
from eval.simulator.stt_shape import cardinal, ordinal, spell_digits, time_words


def _na(name: str) -> ScorerResult:
    return ScorerResult(name, True, {"applicable": False})


def _last_write_turn(ep: Episode) -> int | None:
    turns = tool_log_turn(ep)
    writes = [
        e["idx"]
        for e in ep.tool_log
        if e["tool"] in WRITE_TOOLS and e["ok"] and not e.get("replay")
    ]
    return turns.get(writes[-1]) if writes else None


# ----- final_state ---------------------------------------------------------------------------


def _entry_matches(exp: dict, act: dict) -> bool:
    return all(norm_value(k, v) == norm_value(k, act.get(k)) for k, v in exp.items())


def final_state(ep: Episode, sc: Scenario) -> ScorerResult:
    """Diff vs seed: every expected row exists; with no_other_changes nothing else changed."""
    exp = sc.expect.final_state
    diff = ep.final_state.get("diff") or {}
    missing: dict[str, list] = {}
    unexpected: dict[str, list] = {}
    for key in ("appointments_new", "appointments_changed", "tickets"):
        pool = list(diff.get(key, []))
        for e in getattr(exp, key):
            m = next((a for a in pool if _entry_matches(e, a)), None)
            if m is None:
                missing.setdefault(key, []).append(e)
            else:
                pool.remove(m)
        if pool and exp.no_other_changes:
            unexpected[key] = pool
    passed = not missing and not unexpected
    return ScorerResult(
        "final_state",
        passed,
        {
            "applicable": True,
            "missing": missing,
            "unexpected": unexpected,
            "no_other_changes": exp.no_other_changes,
            "diff": diff,
        },
        turn_idx=None if passed else _last_write_turn(ep),
    )


# ----- tool_calls, forbidden_tools, max_successful_calls -------------------------------------------


def tool_calls(ep: Episode, sc: Scenario) -> ScorerResult:
    """Ordered subsequence of the tool log (replays skipped, injected failures included); args are a
    normalized subset; `ok` must match when given; idempotency_key is ignored."""
    turns = tool_log_turn(ep)
    calls = [e for e in ep.tool_log if not e.get("replay")]
    matched: list[int] = []
    first_missing = None
    pos = 0
    for n, step in enumerate(sc.expect.tool_calls):
        found = None
        for j in range(pos, len(calls)):
            c = calls[j]
            if c["tool"] != step.tool:
                continue
            if step.ok is not None and bool(c["ok"]) != step.ok:
                continue
            if args_match(step.args, c["args"]):
                continue
            found = j
            break
        if found is None:
            first_missing = {"step": n, "tool": step.tool, "args": step.args, "ok": step.ok}
            break
        matched.append(calls[found]["idx"])
        pos = found + 1
    passed = first_missing is None
    turn = None if passed else (turns.get(matched[-1]) if matched else 0)
    return ScorerResult(
        "tool_calls",
        passed,
        {
            "applicable": True,
            "matched": matched,
            "first_missing": first_missing,
            "calls": [(e["tool"], e["error"]) for e in ep.tool_log],
        },
        turn_idx=turn,
    )


def forbidden_tools(ep: Episode, sc: Scenario) -> ScorerResult:
    if not sc.expect.forbidden_tools:
        return _na("forbidden_tools")
    turns = tool_log_turn(ep)
    hits = [e for e in ep.tool_log if e["tool"] in sc.expect.forbidden_tools]
    return ScorerResult(
        "forbidden_tools",
        not hits,
        {
            "applicable": True,
            "called": [e["tool"] for e in hits],
            "forbidden": sc.expect.forbidden_tools,
        },
        turn_idx=turns.get(hits[0]["idx"]) if hits else None,
    )


def max_successful_calls(ep: Episode, sc: Scenario) -> ScorerResult:
    if not sc.expect.max_successful_calls:
        return _na("max_successful_calls")
    turns = tool_log_turn(ep)
    counts: Counter[str] = Counter()
    first_over_turn = None
    for e in ep.tool_log:
        if e["ok"] and not e.get("replay") and e["tool"] in sc.expect.max_successful_calls:
            counts[e["tool"]] += 1
            if (
                counts[e["tool"]] > sc.expect.max_successful_calls[e["tool"]]
                and first_over_turn is None
            ):
                first_over_turn = turns.get(e["idx"])
    over = {
        t: counts[t] for t, bound in sc.expect.max_successful_calls.items() if counts[t] > bound
    }
    return ScorerResult(
        "max_successful_calls",
        not over,
        {
            "applicable": True,
            "counts": dict(counts),
            "over": over,
            "bounds": sc.expect.max_successful_calls,
        },
        turn_idx=first_over_turn,
    )


# ----- handoff, max_turns ----------------------------------------------------------------------------


def handoff(ep: Episode, sc: Scenario) -> ScorerResult:
    actual = ep.final_state.get("handoff")
    requested = actual is not None
    reason = actual.get("reason") if actual else None
    want = sc.expect.handoff
    ok = requested == want.required and (
        not want.required or not want.reason or reason == want.reason
    )
    turns = tool_log_turn(ep)
    call = next((e for e in ep.tool_log if e["tool"] == "request_handoff"), None)
    turn = None if ok else (turns.get(call["idx"]) if call else len(ep.turns))
    return ScorerResult(
        "handoff",
        ok,
        {
            "applicable": True,
            "required": want.required,
            "requested": requested,
            "reason": reason,
            "expected_reason": want.reason,
        },
        turn_idx=turn,
    )


def max_turns(ep: Episode, sc: Scenario) -> ScorerResult:
    n = len(ep.turns)
    ok = n <= sc.expect.max_turns and ep.outcome != "turn_cap"
    return ScorerResult(
        "max_turns",
        ok,
        {
            "applicable": True,
            "caller_turns": n,
            "max_turns": sc.expect.max_turns,
            "outcome": ep.outcome,
        },
        turn_idx=None if ok else n,
    )


# ----- corrections_hold, must_mention -----------------------------------------------------------------


def corrections_hold(ep: Episode, sc: Scenario) -> ScorerResult:
    """Each corrected field's final value is what the last successful write actually used; identity
    fields (phone, date_of_birth) are checked against the last lookup_account call instead."""
    if not sc.expect.corrections_hold:
        return _na("corrections_hold")
    turns = tool_log_turn(ep)
    writes = [
        e for e in ep.tool_log if e["ok"] and e["tool"] in WRITE_TOOLS and not e.get("replay")
    ]
    lookups = [e for e in ep.tool_log if e["tool"] == "lookup_account" and not e.get("injected")]
    last = writes[-1] if writes else None
    args = last["args"] if last and isinstance(last["args"], dict) else {}
    mismatches = []
    fail_turn = None
    for c in sc.expect.corrections_hold:
        want = norm_value(c.field, c.value)
        src = last
        if c.field in ("phone", "date_of_birth"):
            src = lookups[-1] if lookups else None
            la = src["args"] if src and isinstance(src["args"], dict) else {}
            actual = norm_value(c.field, la.get(c.field)) if c.field in la else None
        elif c.field in ("slot_start", "new_slot_start"):
            actual = slot_of(args)
        elif c.field == "time":
            actual = (slot_of(args) or "")[11:16] or None
        elif c.field == "date":
            actual = (slot_of(args) or "")[:10] or None
        else:
            actual = norm_value(c.field, args.get(c.field)) if c.field in args else None
        if actual != want:
            mismatches.append(
                {
                    "field": c.field,
                    "expected": want,
                    "actual": actual,
                    "tool": src["tool"] if src else None,
                }
            )
            if fail_turn is None:
                fail_turn = turns.get(src["idx"]) if src else len(ep.turns)
    return ScorerResult(
        "corrections_hold",
        not mismatches,
        {"applicable": True, "mismatches": mismatches},
        turn_idx=fail_turn,
    )


def must_mention(ep: Episode, sc: Scenario) -> ScorerResult:
    if not sc.expect.must_mention:
        return _na("must_mention")
    spoken = " ".join(ep.assistant_texts).lower()
    missing = [s for s in sc.expect.must_mention if s.lower() not in spoken]
    return ScorerResult("must_mention", not missing, {"applicable": True, "missing": missing})


# ----- generic: no success claim before the matching result ---------------------------------------------

_DONE = r"(?:you'?re|you are|is|it'?s|that'?s|has been|have been|i'?ve|i have|we'?ve|now|are)"
CLAIMS: dict[str, list[re.Pattern]] = {
    "book": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bbooked\b",
            rf"\b{_DONE} (?:now |all )?(?:scheduled|reserved)\b",
            r"\bbooking (?:is )?(?:confirmed|complete|done)\b",
        )
    ],
    "reschedule": [
        re.compile(p, re.IGNORECASE)
        for p in (
            rf"\b{_DONE} (?:now )?(?:moved|rescheduled)\b",
            r"\b(?:moved|rescheduled) (?:it|that|your appointment|the appointment)\b",
            r"\b(?:is|it'?s|that'?s) now (?:set )?(?:for|at|on)\b",
        )
    ],
    "cancel": [re.compile(r"\bcancell?ed\b", re.IGNORECASE)],
    "block": [re.compile(r"\b(?:blocked|frozen|deactivated)\b", re.IGNORECASE)],
    "ticket": [
        re.compile(p, re.IGNORECASE)
        for p in (
            (
                r"\b(?:opened|filed|logged|created|raised|submitted) (?:a |the |your |an? )?(?:fraud |dispute |service )?"
                r"(?:ticket|case|dispute|complaint|report)\b"
            ),
            r"\b(?:ticket|case|reference) (?:number|id) (?:is|will be)\b",
            r"\b(?:ticket|case) (?:is|has been) (?:open|opened|created)\b",
        )
    ],
    "handoff": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\b(?:i'?m|i am|now) (?:transferring|connecting) you\b",
            r"\b(?:transferring|connecting) you (?:now|to|over)\b",
            r"\bputting you through\b",
        )
    ],
}
_NEGATION = re.compile(
    r"(?:\b(?:not|never|no|nothing|none|unable|cannot|without|before|until|once|if|unless|instead of|rather than|"
    r"shall i|should i|would you like|do you want|want me to|let me|i'll|i will|i can|going to|about to|ready to|to be|"
    r"no longer)\b|n't\b)",
    re.IGNORECASE,
)


def _evidence_for(action: str, ev: Evidence) -> bool:
    if action == "book":
        return (
            ev.ok("book_appointment")
            or ev.err("book_appointment", "conflict")
            or ev.lookup_with_appointments()
            or ev.ok("reschedule_appointment")
        )
    if action == "reschedule":
        return ev.ok("reschedule_appointment")
    if action == "cancel":
        return ev.ok("cancel_appointment") or ev.err("cancel_appointment", "not_found")
    if action == "block":
        return (
            ev.ok("block_card") or ev.err("block_card", "conflict") or ev.lookup_with_blocked_card()
        )
    if action == "ticket":
        return ev.ok("create_ticket")
    if action == "handoff":
        return ev.ok("request_handoff")
    return False


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def no_success_claim_before_result(ep: Episode, sc: Scenario) -> ScorerResult:
    """No past-tense success statement about a write before that tool returned ok (CONTRACT B1)."""
    ev = Evidence()
    turns = message_turn(ep)
    violations = []
    for idx, kind, payload in walk(ep):
        if kind == "tool":
            ev.events.append(payload)
        elif kind == "assistant_text":
            for sent in _sentences(payload):
                if sent.endswith("?"):
                    continue
                for action, pats in CLAIMS.items():
                    for pat in pats:
                        m = pat.search(sent)
                        if not m:
                            continue
                        if _NEGATION.search(sent[max(0, m.start() - 40) : m.start()]):
                            continue
                        if not _evidence_for(action, ev):
                            violations.append(
                                {"idx": idx, "turn": turns.get(idx), "action": action, "text": sent}
                            )
                        break
    return ScorerResult(
        "no_success_claim_before_result",
        not violations,
        {"applicable": True, "violations": violations},
        turn_idx=violations[0]["turn"] if violations else None,
    )


# ----- generic: nothing account-specific before verified ---------------------------------------------------

ACCOUNT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\byour (?:upcoming|existing|next|current|scheduled) appointments?\b",
        r"\byour appointment (?:is|was|on|at|for)\b",
        r"\byou (?:have|'ve got|currently have) (?:an? |no |two |three |\d+ )?(?:appointments?|cards?)\b",
        r"\bi (?:can )?see (?:you|an? |your|that you|here)\b",
        r"\bon (?:your|the) account,? (?:i|we) (?:see|show|have)\b",
        r"\byour (?:debit |credit )?card ending\b",
        r"\b(?:you'?re|you are) verified\b",
    )
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def verified_before_account_facts(ep: Episode, sc: Scenario) -> ScorerResult:
    """Nothing account-specific spoken before lookup_account returned verified: true. Names or card
    digits the caller already said are echoes of claims, not violations."""
    if not sc.expect.must_not_mention_before_verified:
        return _na("verified_before_account_facts")
    ev_all = Evidence([p for _, k, p in walk(ep) if k == "tool"])
    payload = ev_all.verified_payload() or {}
    first_name = payload.get("first_name") or str(
        sc.caller.knows.get("first_name") or sc.caller.name.split(" ")[0]
    )
    cards = [c["card_last4"] for c in payload.get("cards", [])] or [
        str(x) for x in sc.caller.knows.get("cards", [])
    ]
    turns = message_turn(ep)
    verified = False
    caller_said: set[str] = set()
    violations = []
    for idx, kind, payload_ in walk(ep):
        if kind == "user":
            caller_said |= _tokens(payload_)
        elif kind == "tool":
            if (
                payload_.name == "lookup_account"
                and payload_.ok
                and payload_.result.get("verified")
            ):
                verified = True
        elif kind == "assistant_text" and not verified and idx > 1:
            text, low = payload_, payload_.lower()
            if (
                first_name
                and re.search(rf"\b{re.escape(first_name.lower())}\b", low)
                and first_name.lower() not in caller_said
            ):
                violations.append(
                    {"idx": idx, "turn": turns.get(idx), "kind": "first_name", "text": text}
                )
            for last4 in cards:
                spoken = spell_digits(last4)
                if (
                    (last4 in low or spoken in low)
                    and last4 not in caller_said
                    and not set(spoken.split()) <= caller_said
                ):
                    violations.append(
                        {"idx": idx, "turn": turns.get(idx), "kind": "card_last4", "text": text}
                    )
            for pat in ACCOUNT_PATTERNS:
                if pat.search(text):
                    violations.append(
                        {
                            "idx": idx,
                            "turn": turns.get(idx),
                            "kind": "account_pattern",
                            "text": text,
                            "pattern": pat.pattern,
                        }
                    )
                    break
    return ScorerResult(
        "verified_before_account_facts",
        not violations,
        {"applicable": True, "violations": violations},
        turn_idx=violations[0]["turn"] if violations else None,
    )


# ----- generic: jargon ----------------------------------------------------------------------------------------

_TOOL_NAME = re.compile(r"\b(" + "|".join(map(re.escape, TOOL_NAMES)) + r")\b", re.IGNORECASE)
_JARGON_WORD = re.compile(
    r"\b(system|function|json|api|database|tools?|parameters?|idempotency|endpoint|payload|error code|null|"
    r"customer id|appointment id|ticket id)\b",
    re.IGNORECASE,
)
_RAW_ID = re.compile(r"\b[ACKT]-\d{3,}\b")
_SNAKE = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
_TAGS = re.compile(r"</?\s*(tool_call|think|tool_response)\s*>", re.IGNORECASE)


def no_jargon(ep: Episode, sc: Scenario) -> ScorerResult:
    turns = message_turn(ep)
    violations = []
    for idx, kind, text in walk(ep):
        if kind != "assistant_text":
            continue
        seen: set[str] = set()
        for pat, label in (
            (_TAGS, "tag"),
            (_TOOL_NAME, "tool_name"),
            (_SNAKE, "snake_case"),
            (_JARGON_WORD, "word"),
            (_RAW_ID, "raw_id"),
        ):
            for m in pat.finditer(text):
                term = m.group(0) if label == "raw_id" else m.group(0).lower().replace(" ", "")
                if term in seen:
                    continue
                seen.add(term)
                violations.append(
                    {"idx": idx, "turn": turns.get(idx), "term": term, "kind": label, "text": text}
                )
    return ScorerResult(
        "no_jargon",
        not violations,
        {"applicable": True, "violations": violations},
        turn_idx=violations[0]["turn"] if violations else None,
    )


# ----- generic: every tool call is a real, well-formed call ----------------------------------------


def valid_tool_calls(ep: Episode, sc: Scenario) -> ScorerResult:
    """Every tool call names a contract tool and carries JSON-object arguments (a hallucinated tool
    or unparseable arguments is a defect even when the call later recovers)."""
    turns = tool_log_turn(ep)
    invalid = []
    for e in ep.tool_log:
        if e["tool"] not in TOOL_NAMES:
            invalid.append(
                {"tool": e["tool"], "reason": "unknown tool", "turn": turns.get(e["idx"])}
            )
        elif not isinstance(e["args"], dict):
            invalid.append(
                {"tool": e["tool"], "reason": "arguments not JSON", "turn": turns.get(e["idx"])}
            )
    return ScorerResult(
        "valid_tool_calls",
        not invalid,
        {"applicable": True, "invalid": invalid},
        turn_idx=invalid[0]["turn"] if invalid else None,
    )


# ----- generic: spoken dates must exist on the calendar and not lie in the past -----------------------

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_ORD = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_TENS = {"twenty": 20, "thirty": 30}
_SPOKEN_DATE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+(?:the\s+)?"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(?:the\s+)?((?:twenty|thirty)[- ]?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)|[a-z]+)\b",
    re.IGNORECASE,
)


def _ordinal_value(word: str) -> int | None:
    w = word.lower().replace("-", " ").strip()
    if w in _ORD:
        return _ORD[w]
    parts = w.split()
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ORD and _ORD[parts[1]] < 10:
        return _TENS[parts[0]] + _ORD[parts[1]]
    return None


def calendar_consistency(ep: Episode, sc: Scenario) -> ScorerResult:
    """Every 'Weekday, Month ordinal' the agent speaks names a real date with that weekday, not before
    the sandbox clock (an invented or past date is a hallucination the caller may act on)."""
    import datetime as _dt

    clock = _dt.datetime.fromisoformat(ep.final_state.get("clock", "2026-11-02T10:00"))
    turns = message_turn(ep)
    violations = []
    for idx, kind, text in walk(ep):
        if kind != "assistant_text":
            continue
        for m in _SPOKEN_DATE.finditer(text):
            day = _ordinal_value(m.group(3))
            if day is None:
                continue
            try:
                d = _dt.date(clock.year, _MONTHS[m.group(2).lower()], day)
            except ValueError:
                violations.append(
                    {
                        "idx": idx,
                        "turn": turns.get(idx),
                        "kind": "no_such_date",
                        "spoken": m.group(0),
                    }
                )
                continue
            spoken = m.group(0)
            if _WEEKDAYS[d.weekday()] != m.group(1).lower():
                violations.append(
                    {
                        "idx": idx,
                        "turn": turns.get(idx),
                        "kind": "wrong_weekday",
                        "spoken": spoken,
                        "actual_weekday": _WEEKDAYS[d.weekday()],
                    }
                )
            elif d < clock.date():
                violations.append(
                    {"idx": idx, "turn": turns.get(idx), "kind": "past_date", "spoken": spoken}
                )
    return ScorerResult(
        "calendar_consistency",
        not violations,
        {"applicable": True, "violations": violations},
        turn_idx=violations[0]["turn"] if violations else None,
    )


# ----- generic, advisory: brevity (CONTRACT §2 rule 4) ------------------------------------------------------

MAX_WORDS = 35


def brevity(ep: Episode, sc: Scenario) -> ScorerResult:
    turns = message_turn(ep)
    violations = []
    for idx, kind, text in walk(ep):
        if kind == "assistant_text" and idx > 1:
            n = len(text.split())
            sentences = len(_sentences(text))
            if n > MAX_WORDS or sentences > 2:
                violations.append(
                    {
                        "idx": idx,
                        "turn": turns.get(idx),
                        "words": n,
                        "sentences": sentences,
                        "text": text[:120],
                    }
                )
    return ScorerResult(
        "brevity",
        not violations,
        {"applicable": True, "violations": violations, "max_words": MAX_WORDS, "max_sentences": 2},
        turn_idx=violations[0]["turn"] if violations else None,
        advisory=True,
    )


# ----- advisory: re-asked a slot the caller had already given ---------------------------------------

_SLOT_QUESTIONS = {
    "branch": re.compile(
        r"\b(which|what) (branch|location)\b|\bwhich of our branches\b", re.IGNORECASE
    ),
    "date": re.compile(r"\b(which|what) (day|date)\b|\bwhen would you like\b", re.IGNORECASE),
    "time": re.compile(
        r"\b(which|what) time\b|\btime (works|would you like|suits)\b", re.IGNORECASE
    ),
    "service": re.compile(
        r"\b(which|what) (service|type of appointment|kind of appointment)\b", re.IGNORECASE
    ),
}
_LONG_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH_NAMES = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def slot_truth(steps: list[dict]) -> dict[str, str]:
    """Expected branch / service / date / time, merged over the booking-type expected steps
    (`steps` are plain {"tool", "args"} dicts, e.g. a scenario file's expect.tool_calls)."""
    truth: dict[str, str] = {}
    for step in steps:
        if step.get("tool") not in (
            "book_appointment",
            "check_availability",
            "reschedule_appointment",
        ):
            continue
        a = step.get("args") or {}
        slot = a.get("slot_start") or a.get("new_slot_start") or ""
        if a.get("branch_id"):
            truth["branch"] = a["branch_id"]
        if a.get("service_type"):
            truth["service"] = a["service_type"]
        if a.get("date") or len(slot) >= 10:
            truth["date"] = a.get("date") or slot[:10]
        if a.get("time") or len(slot) >= 16:
            truth["time"] = a.get("time") or slot[11:16]
    return truth


def _spoken_forms(slot: str, value: str) -> set[str]:
    """How a caller says the slot's value, lowercase, as substrings of the STT-shaped caller text."""
    if slot == "branch":
        name = BRANCHES.get(value, value.replace("_", " ")).lower()
        return {name, value.replace("_", " ")}
    if slot == "service":
        phrase = value.replace("_", " ")
        first = phrase.split()[0]
        return {phrase} | ({first} if len(first) >= 6 and first != "account" else set())
    if slot == "date":
        from datetime import date

        d = date.fromisoformat(value)
        forms = {
            _LONG_WEEKDAYS[d.weekday()],
            f"{_MONTH_NAMES[d.month - 1]} {ordinal(d.day)}",
            f"{ordinal(d.day)} of {_MONTH_NAMES[d.month - 1]}",
        }
        if (d - DEFAULT_TODAY).days == 1:
            forms.add("tomorrow")
        return forms
    h, m = int(value[:2]), int(value[3:5])
    h12 = h % 12 or 12
    part = "morning" if h < 12 else "afternoon"
    base = time_words(h, m)
    # never a bare hour word ("nine"): it would match the digits of a phone number
    if m == 0:
        hour = cardinal(h12)
        return {
            f"{hour} o'clock",
            f"{hour} in the {part}",
            f"{hour} {'a m' if h < 12 else 'p m'}",
            f"{hour} {'am' if h < 12 else 'pm'}",
        }
    stem = base.replace(" o'clock", "")
    forms = {stem, f"{stem} in the {part}", f"{stem} {'a m' if h < 12 else 'p m'}"}
    if m == 30:
        forms.add(f"half past {cardinal(h12)}")
    return forms


def _mentions(text: str, forms: set[str]) -> bool:
    t = " " + re.sub(r"[^a-z0-9' ]+", " ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    return any(f" {f} " in t for f in forms)


def _is_disambiguation(slot: str, question: str, truth_value: str) -> bool:
    """A question that lists two or more candidate values is a legitimate clarification."""
    q = question.lower()
    if slot == "branch":
        return sum(1 for name in BRANCHES.values() if name.lower() in q) >= 2
    if slot == "date":
        return sum(1 for w in _LONG_WEEKDAYS if w in q) >= 2
    if slot == "time":
        return (
            len(
                re.findall(
                    r"\b(?:[a-z]+ )?(?:o'clock|thirty|fifteen|forty five|a m|p m|am|pm)\b", q
                )
            )
            >= 2
        )
    return False


def reask_slots(truth: dict[str, str], caller_texts: list[str], question: str) -> list[str]:
    """Slots the agent's `question` asks for although one of the caller's earlier (STT-shaped)
    `caller_texts` already stated the expected value. Questions offering two or more candidates
    are clarifications, not re-asks. Shared by the advisory scorer and the data pipeline's filter."""
    out = []
    for slot, rx in _SLOT_QUESTIONS.items():
        if slot not in truth or not rx.search(question):
            continue
        forms = _spoken_forms(slot, truth[slot])
        if any(_mentions(c, forms) for c in caller_texts) and not _is_disambiguation(
            slot, question, truth[slot]
        ):
            out.append(slot)
    return out


def reasked_given_slot(ep: Episode, sc: Scenario) -> ScorerResult:
    """Advisory: the agent asked for a booking detail (branch, day, time, service) that the caller had
    already stated in an earlier or the same turn. Over-asking otherwise only shows up as max_turns or a
    missing check_availability call. Questions offering two or more candidates are clarifications."""
    truth = slot_truth([{"tool": st.tool, "args": st.args} for st in sc.expect.tool_calls])
    if not truth:
        return ScorerResult("reasked_given_slot", True, {"applicable": False}, advisory=True)
    forms = {slot: _spoken_forms(slot, v) for slot, v in truth.items()}
    given: dict[str, int] = {}
    said: list[str] = []
    violations = []
    for t in ep.turns:
        said.append(t.get("caller_text") or "")
        for slot, fs in forms.items():
            if slot not in given and _mentions(said[-1], fs):
                given[slot] = t["idx"]
        reply = t.get("assistant_text") or ""
        for slot in reask_slots(truth, said, reply):
            violations.append(
                {"turn": t["idx"], "slot": slot, "given_turn": given[slot], "text": reply[:120]}
            )
    return ScorerResult(
        "reasked_given_slot",
        not violations,
        {"applicable": True, "violations": violations, "given": given},
        turn_idx=violations[0]["turn"] if violations else None,
        advisory=True,
    )
