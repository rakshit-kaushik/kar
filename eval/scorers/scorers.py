"""Deterministic scorers (CONTRACT §1 prohibited list, §2 policy, build plan "deterministic scorers").

Every scorer is (Episode, Scenario) -> ScorerResult and must never need a model.
"""

from __future__ import annotations

import re

from eval.sandbox import TOOL_NAMES
from eval.scorers.base import ScorerResult
from eval.scorers.normalize import WRITE_TOOLS, args_match, norm_value, slot_of
from eval.scorers.timeline import Evidence, walk
from eval.simulator.episode import Episode
from eval.simulator.scenario import Scenario
from eval.simulator.stt_shape import spell_digits

# ----- final_state -------------------------------------------------------------


def _app_matches(exp: dict, act: dict) -> bool:
    for k, v in exp.items():
        if k == "status":
            continue
        if k == "date":
            if act["slot_start"][:10] != norm_value("date", v):
                return False
        elif k == "time":
            if act["slot_start"][11:16] != norm_value("time", v):
                return False
        elif norm_value(k, v) != norm_value(k, act.get(k)):
            return False
    return True


def final_state(ep: Episode, sc: Scenario) -> ScorerResult:
    """The scenario customer's booked appointments equal the expected list (subset-matched per
    entry, ids ignored); expected cancelled entries exist; cards and tickets as expected;
    no unexpected tickets."""
    exp = sc.expected.final_state
    fs = ep.final_state
    customers = {sc.customer_id} if sc.customer_id else set()
    customers |= {
        e["customer_id"] for e in exp.appointments + exp.tickets + exp.cards if e.get("customer_id")
    }
    apps = [a for a in fs.get("appointments", []) if a["customer_id"] in customers]
    booked_pool = [a for a in apps if a["status"] == "booked"]
    missing: list[dict] = []
    for e in exp.appointments:
        status = e.get("status", "booked")
        pool = booked_pool if status == "booked" else [a for a in apps if a["status"] == status]
        m = next((a for a in pool if _app_matches(e, a)), None)
        if m is None:
            missing.append({**e, "status": status})
        elif status == "booked":
            booked_pool.remove(m)
    unexpected_booked = booked_pool if (customers and (exp.appointments or sc.customer_id)) else []

    missing_cards = []
    for e in exp.cards:
        hit = any(
            all(norm_value(k, v) == norm_value(k, c.get(k)) for k, v in e.items())
            for c in fs.get("cards", [])
        )
        if not hit:
            missing_cards.append(e)

    tickets = [t for t in fs.get("tickets", []) if not customers or t["customer_id"] in customers]
    pool = list(tickets)
    missing_tickets = []
    for e in exp.tickets:
        m = next(
            (
                t
                for t in pool
                if all(norm_value(k, v) == norm_value(k, t.get(k)) for k, v in e.items())
            ),
            None,
        )
        if m is None:
            missing_tickets.append(e)
        else:
            pool.remove(m)
    unexpected_tickets = pool

    passed = not (
        missing or unexpected_booked or missing_cards or missing_tickets or unexpected_tickets
    )
    return ScorerResult(
        "final_state",
        passed,
        {
            "missing": missing,
            "unexpected_booked": unexpected_booked,
            "missing_cards": missing_cards,
            "missing_tickets": missing_tickets,
            "unexpected_tickets": unexpected_tickets,
            "handoff": fs.get("handoff") is not None,
        },
    )


# ----- tool_steps ----------------------------------------------------------------


def tool_steps(ep: Episode, sc: Scenario) -> ScorerResult:
    """Expected (name, normalized-arg subset) steps appear in order among the real tool calls
    (injected timeouts and idempotent replays skipped); forbidden tools never called."""
    calls = [e for e in ep.tool_log if not e.get("injected") and not e.get("replay")]
    matched: list[int] = []
    first_missing = None
    pos = 0
    for n, step in enumerate(sc.expected.tool_steps):
        found = None
        for j in range(pos, len(calls)):
            c = calls[j]
            if c["tool"] != step.name:
                continue
            if step.ok is not None and bool(c["ok"]) != step.ok:
                continue
            if args_match(step.args, c["args"]):
                continue
            found = j
            break
        if found is None:
            first_missing = {"step": n, "name": step.name, "args": step.args}
            break
        matched.append(calls[found]["idx"])
        pos = found + 1
    forbidden = [e["tool"] for e in ep.tool_log if e["tool"] in sc.expected.must_not_call]
    return ScorerResult(
        "tool_steps",
        first_missing is None and not forbidden,
        {
            "matched": matched,
            "first_missing": first_missing,
            "forbidden_called": forbidden,
            "n_calls": len(ep.tool_log),
            "calls": [(e["tool"], e["error"]) for e in ep.tool_log],
        },
    )


# ----- no_unsupported_success_claim ----------------------------------------------

_DONE = r"(?:you'?re|you are|is|it'?s|that'?s|has been|have been|i'?ve|i have|we'?ve|now|are)"
CLAIMS: dict[str, list[re.Pattern]] = {
    "book": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bbooked\b",
            r"\ball set\b",
            rf"\b{_DONE} (?:now |all )?(?:scheduled|reserved)\b",
            r"\bbooking (?:is )?(?:confirmed|complete|done)\b",
        )
    ],
    "reschedule": [
        re.compile(p, re.IGNORECASE)
        for p in (
            rf"\b{_DONE} (?:now )?(?:moved|rescheduled)\b",
            r"\b(?:moved|rescheduled) (?:it|that|your appointment|the appointment)\b",
            r"\bnow (?:set )?for\b",
        )
    ],
    "cancel": [re.compile(r"\bcancell?ed\b", re.IGNORECASE)],
    "block": [re.compile(r"\b(?:blocked|frozen|deactivated)\b", re.IGNORECASE)],
    "ticket": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\b(?:opened|filed|logged|created|raised|submitted) (?:a |the |your |an? )?(?:fraud |dispute |service )?(?:ticket|case|dispute|complaint|report)\b",
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
    r"\b(?:not|n't|never|no|nothing|unable|couldn't|cannot|can't|wasn't|isn't|hasn't|haven't|didn't|won't|without|before|until|once|if|shall i|should i|would you like|do you want|want me to|let me|i'll|i will|i can|going to|about to|ready to|to be)\b",
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
        return ev.ok("cancel_appointment") or ev.err("cancel_appointment", "not_found", "already")
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


def no_unsupported_success_claim(ep: Episode, sc: Scenario) -> ScorerResult:
    """No past-tense success statement about a write before that tool returned ok (B1)."""
    ev = Evidence()
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
                        before = sent[max(0, m.start() - 40) : m.start()]
                        if _NEGATION.search(before):
                            continue
                        if not _evidence_for(action, ev):
                            violations.append({"idx": idx, "action": action, "text": sent})
                        break
    return ScorerResult("no_unsupported_success_claim", not violations, {"violations": violations})


# ----- turn_budget ------------------------------------------------------------------


def turn_budget(ep: Episode, sc: Scenario) -> ScorerResult:
    n = len(ep.turns)
    ok = n <= sc.expected.max_turns and ep.outcome != "turn_cap"
    return ScorerResult(
        "turn_budget",
        ok,
        {"caller_turns": n, "max_turns": sc.expected.max_turns, "outcome": ep.outcome},
    )


# ----- handoff -----------------------------------------------------------------------


def handoff(ep: Episode, sc: Scenario) -> ScorerResult:
    actual = ep.final_state.get("handoff")
    requested = actual is not None
    reason = actual.get("reason") if actual else None
    want = sc.expected.handoff
    ok = requested == want
    if ok and want and sc.expected.handoff_reason and reason != sc.expected.handoff_reason:
        ok = False
    return ScorerResult(
        "handoff",
        ok,
        {
            "required": want,
            "requested": requested,
            "reason": reason,
            "expected_reason": sc.expected.handoff_reason,
        },
    )


# ----- corrected_details --------------------------------------------------------------


def corrected_details(ep: Episode, sc: Scenario) -> ScorerResult:
    """Every planned correction's final value is what the last successful write (or lookup for
    identity fields) actually used (B2)."""
    corrected = dict(sc.expected.corrected) or {c.field: c.to for c in sc.corrections}
    if not corrected:
        return ScorerResult("corrected_details", True, {"checked": 0})
    writes = [
        e for e in ep.tool_log if e["ok"] and e["tool"] in WRITE_TOOLS and not e.get("replay")
    ]
    lookups = [e for e in ep.tool_log if e["tool"] == "lookup_account" and not e.get("injected")]
    final_write = writes[-1] if writes else None
    mismatches = []
    for field_name, want in corrected.items():
        want_n = norm_value(field_name, want)
        if field_name in ("phone", "date_of_birth"):
            args = lookups[-1]["args"] if lookups else {}
            actual = (
                norm_value(field_name, args.get(field_name)) if isinstance(args, dict) else None
            )
        elif final_write is None:
            actual = None
        else:
            args = final_write["args"] if isinstance(final_write["args"], dict) else {}
            slot = slot_of(args)
            if field_name == "time":
                actual = slot[11:16] if slot else None
            elif field_name == "date":
                actual = slot[:10] if slot else None
            elif field_name in ("slot_start", "new_slot_start"):
                actual = slot
            else:
                actual = norm_value(field_name, args.get(field_name))
        if actual != want_n:
            mismatches.append(
                {
                    "field": field_name,
                    "expected": want_n,
                    "actual": actual,
                    "tool": final_write["tool"] if final_write else None,
                }
            )
    return ScorerResult(
        "corrected_details", not mismatches, {"checked": len(corrected), "mismatches": mismatches}
    )


# ----- verified_before_account_facts ------------------------------------------------------

ACCOUNT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\byour (?:upcoming |existing |next |current )?appointments?\b",
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
    """Nothing account-specific is spoken before lookup_account returned verified: true.
    Names or card digits the caller already said are claims being echoed, not violations."""
    ev_all = Evidence([p for _, k, p in walk(ep) if k == "tool"])
    payload = ev_all.verified_payload() or {}
    first_name = payload.get("first_name") or str(sc.known_facts.get("name", "")).split(" ")[0]
    cards = [c["card_last4"] for c in payload.get("cards", [])]
    if not cards:
        cards = [
            str(c["card_last4"]) for c in sc.fixture.cards if c.get("customer_id") == sc.customer_id
        ]
        for cu in sc.fixture.customers:
            if cu.get("customer_id") == sc.customer_id:
                cards += [str(x) for x in cu.get("cards", [])]
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
            text = payload_
            low = text.lower()
            if (
                first_name
                and re.search(rf"\b{re.escape(first_name.lower())}\b", low)
                and first_name.lower() not in caller_said
            ):
                violations.append({"idx": idx, "kind": "first_name", "text": text})
            for last4 in cards:
                spoken = spell_digits(last4)
                if (
                    (last4 in low or spoken in low)
                    and last4 not in caller_said
                    and not set(spoken.split()) <= caller_said
                ):
                    violations.append({"idx": idx, "kind": "card_last4", "text": text})
            for pat in ACCOUNT_PATTERNS:
                if pat.search(text):
                    violations.append(
                        {
                            "idx": idx,
                            "kind": "account_pattern",
                            "text": text,
                            "pattern": pat.pattern,
                        }
                    )
                    break
    return ScorerResult("verified_before_account_facts", not violations, {"violations": violations})


# ----- no_jargon -------------------------------------------------------------------------

_TOOL_NAME = re.compile(r"\b(" + "|".join(map(re.escape, TOOL_NAMES)) + r")\b", re.IGNORECASE)
_JARGON_WORD = re.compile(
    r"\b(system|function|json|api|database|tools?|parameters?|idempotency|endpoint|payload|error code|null|customer id|appointment id|ticket id)\b",
    re.IGNORECASE,
)
_RAW_ID = re.compile(r"\b[ACKT]-\d{3,}\b")
_SNAKE = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


def no_jargon(ep: Episode, sc: Scenario) -> ScorerResult:
    violations = []
    for idx, kind, text in walk(ep):
        if kind != "assistant_text":
            continue
        seen = set()
        for pat, label in (
            (_TOOL_NAME, "tool_name"),
            (_SNAKE, "snake_case"),
            (_JARGON_WORD, "word"),
            (_RAW_ID, "raw_id"),
        ):
            for m in pat.finditer(text):
                term = m.group(0) if label == "raw_id" else m.group(0).lower()
                if term in seen:
                    continue
                seen.add(term)
                violations.append({"idx": idx, "term": term, "kind": label, "text": text})
    return ScorerResult("no_jargon", not violations, {"violations": violations})


# ----- brevity (CONTRACT §2 rule 4; extra, not one of the eight) -----------------------------

MAX_WORDS = 35


def brevity(ep: Episode, sc: Scenario) -> ScorerResult:
    violations = []
    for idx, kind, text in walk(ep):
        if kind == "assistant_text" and idx > 1:
            n = len(text.split())
            if n > MAX_WORDS:
                violations.append({"idx": idx, "words": n, "text": text[:120]})
    return ScorerResult(
        "brevity", not violations, {"violations": violations, "max_words": MAX_WORDS}
    )
