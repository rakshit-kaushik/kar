"""Bank-driven generation scenarios (eval/scenarios/schema.json shape) for the data phase.

Every scenario is built against the seed sandbox so its `expect` block is exact (open slots,
existing appointments, cards), validated with the same schema and seed checks as the test set,
and never reuses a test scenario's combination: (customer, branch, service, date) of a new
booking, (customer, category) of a ticket, or a changed appointment id. Personas come from a
hand-written bank; phrasing seeds mined from Taskmaster/SGD are appended by data/gen/banks.py.
"""

from __future__ import annotations

import json
import random
import string
from datetime import date, timedelta
from pathlib import Path
from typing import ClassVar

from eval.sandbox import Sandbox
from eval.sandbox import seed as S
from eval.sandbox.store import HANDOFF_REASONS, TICKET_CATEGORIES
from eval.simulator.scenario import FAMILIES, load_scenarios
from eval.simulator.stt_shape import ONES, ordinal, spell_digits, time_words

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "eval" / "scenarios" / "test"

SERVICE_WORDS = {
    "account_opening": "account opening",
    "mortgage_consultation": "mortgage consultation",
    "loan_application": "loan application",
    "financial_review": "financial review",
    "notary": "notary appointment",
    "safe_deposit_box": "safe deposit box appointment",
}
BRANCH_WORDS = {
    "market_st": "Market Street",
    "marlow_st": "Marlow Street",
    "marker_st": "Marker Street",
    "mill_rd": "Mill Road",
}
CONFUSABLE = ("market_st", "marlow_st", "marker_st")
TICKET_GOALS = {
    "fee_dispute": "dispute a thirty-five dollar overdraft fee from last week that you think is unfair",
    "complaint": "complain that you waited forty minutes at the branch on Saturday and nobody apologised",
    "statement_document": "ask for a copy of last year's statements to be mailed to you",
    "account_access": "report that you are locked out of the mobile app and want it looked at",
    "transfer_payment": "report that a transfer you sent on Friday has been pending for three days",
}
PERSONAS = [
    "Brisk and businesslike; answers in as few words as possible",
    "Chatty and warm; adds small talk but gives correct facts",
    "Nervous first-time caller; asks the agent to repeat things once",
    "Elderly and unhurried; speaks slowly and sometimes repeats the question back",
    "Distracted parent; occasionally trails off mid-sentence and then finishes",
    "Impatient; wants it done fast and says so",
    "Polite and formal; thanks the agent after every answer",
    "Mumbling commuter on a train; short clipped answers",
    "Cheerful student; informal, says 'cool' and 'yeah'",
    "Skeptical; asks the agent to confirm details before agreeing",
    "Apologetic; keeps saying 'sorry to bother you'",
    "Confident regular customer who knows the branch staff by name",
    "Soft-spoken; gives information only when directly asked",
    "Talkative retiree with time to spare; wanders but comes back to the point",
    "Harried shift worker calling on a break; checks the time",
    "Non-native speaker; simple sentences, very clear about numbers",
    "Dry humour; jokes once, then cooperates",
    "Anxious about money; wants reassurance that nothing costs extra",
    "Methodical; reads details from notes, one at a time",
    "Easygoing; happy with whatever time the agent offers first",
    "Slightly hard of hearing; asks 'pardon?' once early on",
    "Efficient professional; confirms with 'correct' and 'yes, go ahead'",
    "Forgetful; needs a second to recall the date of birth, then gets it right",
    "Friendly but vague about details until the agent pins them down",
]
STT_QUIRKS = [
    "say 'oh' for every zero in the phone number",
    "add 'um' and 'like' fillers",
    "drop a small word now and then",
    "say the birth year in two parts, like 'nineteen' then 'seventy nine'",
    "repeat the last word of a sentence once",
    "run two short sentences together without a pause",
]
WRITE_TOOLS = ["book_appointment", "reschedule_appointment", "cancel_appointment", "create_ticket"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def spoken_date(d: date) -> str:
    return f"{WEEKDAYS[d.weekday()]}, November {ordinal(d.day)}"


def spoken_time(hhmm: str) -> str:
    h, m = int(hhmm[:2]), int(hhmm[3:])
    words = time_words(h, m).replace(" o'clock", "")
    return f"{words} in the {'morning' if h < 12 else 'afternoon'}"


def spoken_year(y: int) -> str:
    from eval.simulator.stt_shape import year_words

    return year_words(y)


def test_signatures(test_dir: str | Path = TEST_DIR) -> set[tuple]:
    sigs: set[tuple] = set()
    for s in load_scenarios(test_dir):
        for a in s.expect.final_state.appointments_new:
            sigs.add(
                (a["customer_id"], a["branch_id"], a["service_type"], str(a["slot_start"])[:10])
            )
        for t in s.expect.final_state.tickets:
            sigs.add(("ticket", s.caller.customer_id, t["category"]))
        for c in s.expect.final_state.appointments_changed:
            sigs.add(("changed", c["appointment_id"]))
    return sigs


class _Gen:
    def __init__(
        self,
        seed: int,
        sb: Sandbox,
        sigs: set[tuple],
        short: bool = False,
        phrasing: list[str] | None = None,
    ):
        self.rng = random.Random(seed)
        self.short = short
        self.phrasing = phrasing or []
        self.sb = sb
        self.sigs = set(sigs)
        self.used: set[tuple] = set()
        self.days = [
            sb.today + timedelta(days=i)
            for i in range(1, 13)
            if S.hours_for(sb.today + timedelta(days=i), sb.seed)
        ]

    # ----- building blocks -----------------------------------------------------------------
    def customer(self, with_appointment: bool | None = None):
        custs = list(self.sb.customers.values())
        if with_appointment is True:
            custs = [c for c in custs if self.sb.customer_appointments(c.customer_id)]
        elif with_appointment is False:
            custs = [c for c in custs if not self.sb.customer_appointments(c.customer_id)]
        return self.rng.choice(custs)

    def caller(
        self, cust, noise: str | None = None, withholds=None, quirks=None, extra: dict | None = None
    ) -> dict:
        noise = noise or self.rng.choices(["none", "low", "high"], weights=[60, 25, 15])[0]
        knows = {
            "phone": cust.phone,
            "phone_spoken": spell_digits(cust.phone),
            "date_of_birth": cust.date_of_birth.isoformat(),
            "first_name": cust.first_name,
            "cards": [k.card_last4 for k in self.sb.customer_cards(cust.customer_id)],
        }
        apps = self.sb.customer_appointments(cust.customer_id)
        if apps:
            knows["appointments"] = [
                {
                    "appointment_id": a.appointment_id,
                    "what": f"{a.service_type} at {BRANCH_WORDS[a.branch_id]} on {a.slot}",
                }
                for a in apps
            ]
        if extra:
            knows.update(extra)
        persona = self.rng.choice(PERSONAS)
        if self.phrasing:
            a, b = self.rng.sample(self.phrasing, 2)
            persona += f". Phrasing seeds: '{a}' / '{b}'"
        c = {
            "customer_id": cust.customer_id,
            "name": f"{cust.first_name} {cust.last_name}",
            "persona": persona,
            "knows": knows,
            "noise": noise,
        }
        if withholds:
            c["withholds"] = list(withholds)
        if quirks:
            c["stt_quirks"] = list(quirks)
        return c

    def open_slot(self, cust, service: str, branch: str | None = None, exclude_days=()):
        for _ in range(40):
            b = branch or self.rng.choice(list(BRANCH_WORDS))
            d = self.rng.choice([x for x in self.days if x not in exclude_days])
            slots = self.sb.open_slots(b, d)
            if not slots:
                continue
            if any(
                a.service_type == service and a.slot_start.date() == d
                for a in self.sb.customer_appointments(cust.customer_id)
            ):
                continue
            sig = (cust.customer_id, b, service, d.isoformat())
            if sig in self.sigs or sig in self.used:
                continue
            return b, d, self.rng.choice(slots)
        return None

    def lookup_step(self, cust) -> dict:
        return {
            "tool": "lookup_account",
            "args": {"phone": cust.phone, "date_of_birth": cust.date_of_birth.isoformat()},
            "ok": True,
        }

    def base(
        self, family: str, title: str, tests: str, caller: dict, goal: str, expect: dict, **more
    ) -> dict:
        expect.setdefault("handoff", {"required": False, "reason": None})
        expect.setdefault("max_turns", 14)
        expect["final_state"] = {**expect.get("final_state", {}), "no_other_changes": True}
        d = {
            "id": "pending",
            "family": family,
            "title": title,
            "tests": tests,
            "caller": caller,
            "goal": goal,
            "expect": expect,
        }
        d.update({k: v for k, v in more.items() if v})
        return d

    # ----- family builders ---------------------------------------------------------------------
    def book_parts(self, cust, service=None, branch=None):
        service = service or self.rng.choice(list(SERVICE_WORDS))
        found = self.open_slot(cust, service, branch)
        if not found:
            return None
        b, d, hhmm = found
        self.used.add((cust.customer_id, b, service, d.isoformat()))
        slot = f"{d.isoformat()}T{hhmm}"
        steps = [
            self.lookup_step(cust),
            {
                "tool": "check_availability",
                "args": {"branch_id": b, "service_type": service, "date": d.isoformat()},
                "ok": True,
            },
            {
                "tool": "book_appointment",
                "args": {
                    "customer_id": cust.customer_id,
                    "branch_id": b,
                    "service_type": service,
                    "slot_start": slot,
                },
                "ok": True,
            },
        ]
        fs = {
            "appointments_new": [
                {
                    "customer_id": cust.customer_id,
                    "branch_id": b,
                    "service_type": service,
                    "slot_start": slot,
                }
            ]
        }
        return service, b, d, hhmm, slot, steps, fs

    def ordinary(self):
        roll = self.rng.random()
        if self.short:
            roll = self.rng.choice([0.1, 0.3, 0.9])  # status, cancel or a catalog question
        if roll < 0.2:
            cust = self.customer(with_appointment=True)
            return self.base(
                "ordinary",
                f"{cust.first_name} asks when the next appointment is",
                "Status question: lookup only, facts read back, no writes",
                self.caller(cust),
                "Ask when and where your upcoming appointment is. Once the agent tells you the day, time and branch, say thanks and hang up. Do not book, move or cancel anything.",
                {
                    "tool_calls": [self.lookup_step(cust)],
                    "forbidden_tools": WRITE_TOOLS,
                    "final_state": {},
                    "max_turns": 8,
                },
            )
        if roll < 0.35:
            cust = self.customer(with_appointment=True)
            ap = self.sb.customer_appointments(cust.customer_id)[0]
            if ("changed", ap.appointment_id) in self.sigs:
                return None
            return self.base(
                "ordinary",
                f"{cust.first_name} cancels an appointment",
                "Cancel after read-back; status cancelled; nothing else changes",
                self.caller(cust),
                f"Cancel your {SERVICE_WORDS[ap.service_type]} at {BRANCH_WORDS[ap.branch_id]} on {spoken_date(ap.slot_start.date())}; you no longer need it. Say yes when the agent reads it back, then say goodbye.",
                {
                    "tool_calls": [
                        self.lookup_step(cust),
                        {
                            "tool": "cancel_appointment",
                            "args": {"appointment_id": ap.appointment_id},
                            "ok": True,
                        },
                    ],
                    "forbidden_tools": ["book_appointment", "reschedule_appointment"],
                    "final_state": {
                        "appointments_changed": [
                            {"appointment_id": ap.appointment_id, "status": "cancelled"}
                        ]
                    },
                    "max_turns": 10,
                },
            )
        if roll < 0.5:
            cust = self.customer(with_appointment=True)
            ap = self.sb.customer_appointments(cust.customer_id)[0]
            if ("changed", ap.appointment_id) in self.sigs:
                return None
            for _ in range(30):
                d = self.rng.choice([x for x in self.days if x != ap.slot_start.date()])
                slots = self.sb.open_slots(ap.branch_id, d)
                if slots:
                    hhmm = self.rng.choice(slots)
                    break
            else:
                return None
            new = f"{d.isoformat()}T{hhmm}"
            return self.base(
                "ordinary",
                f"{cust.first_name} moves an appointment",
                "Reschedule after read-back; old slot freed; nothing else changes",
                self.caller(cust),
                f"Move your {SERVICE_WORDS[ap.service_type]} at {BRANCH_WORDS[ap.branch_id]} from {spoken_date(ap.slot_start.date())} to {spoken_date(d)} at {spoken_time(hhmm)}; that time is available, insist on it. Confirm when it is read back.",
                {
                    "tool_calls": [
                        self.lookup_step(cust),
                        {
                            "tool": "reschedule_appointment",
                            "args": {"appointment_id": ap.appointment_id, "new_slot_start": new},
                            "ok": True,
                        },
                    ],
                    "forbidden_tools": ["book_appointment", "cancel_appointment"],
                    "final_state": {
                        "appointments_changed": [
                            {
                                "appointment_id": ap.appointment_id,
                                "slot_start": new,
                                "status": "booked",
                            }
                        ]
                    },
                    "max_turns": 12,
                },
            )
        if self.short and roll >= 0.85:
            cust = self.customer()
            q, mention = self.rng.choice(
                [
                    ("what the Saturday opening hours are", "nine"),
                    ("how much a notary appointment costs", "ten dollars"),
                    ("how much a safe deposit box costs", "sixty dollars"),
                    ("whether a financial review costs anything", "free"),
                    ("what the weekday hours are", "nine to five"),
                ]
            )
            return self.base(
                "ordinary",
                f"{cust.first_name} asks a catalog question",
                "Catalog fact spoken correctly, no tool writes, short close",
                self.caller(cust),
                f"After verification ask {q}. Accept the answer, say thanks and hang up. Nothing else.",
                {
                    "tool_calls": [self.lookup_step(cust)],
                    "forbidden_tools": WRITE_TOOLS,
                    "must_mention": [mention],
                    "final_state": {},
                    "max_turns": 8,
                },
            )
        cust = self.customer()
        parts = self.book_parts(cust)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        return self.base(
            "ordinary",
            f"{cust.first_name} books a {SERVICE_WORDS[service]}",
            "Plain happy path: verify, check, read back, book, close with only what was done",
            self.caller(cust),
            f"Book a {SERVICE_WORDS[service]} at {BRANCH_WORDS[b]} on {spoken_date(d)} at {spoken_time(hhmm)}. That time is available; insist on it if other times are offered. Confirm when the agent reads it back.",
            {"tool_calls": steps, "final_state": fs, "max_turns": 12},
        )

    def missing_info(self):
        cust = self.customer()
        parts = self.book_parts(cust)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        return self.base(
            "missing_info",
            f"{cust.first_name} wants to 'come in' and says nothing more",
            "Agent must ask for service, branch, day and time one question at a time",
            self.caller(cust, withholds=["service type", "branch", "day", "time"]),
            f"Open with 'i need to come in and see someone' and nothing more. Give each detail only when asked: a {SERVICE_WORDS[service]}, {BRANCH_WORDS[b]}, {spoken_date(d)}, {spoken_time(hhmm)} (available; insist on it). Confirm the read-back.",
            {"tool_calls": steps, "final_state": fs, "max_turns": 14},
        )

    def correction(self):
        cust = self.customer()
        parts = self.book_parts(cust)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        slots = [s for s in self.sb.open_slots(b, d) if s != hhmm]
        if self.rng.random() < 0.6 and slots:
            new_hhmm = self.rng.choice(slots)
            new_slot = f"{d.isoformat()}T{new_hhmm}"
            steps[2]["args"]["slot_start"] = new_slot
            fs["appointments_new"][0]["slot_start"] = new_slot
            return self.base(
                "correction",
                f"{cust.first_name} changes the time at the read-back",
                "The corrected time must be the one booked and read back",
                self.caller(cust),
                f"Book a {SERVICE_WORDS[service]} at {BRANCH_WORDS[b]} on {spoken_date(d)} at {spoken_time(hhmm)}. When the agent reads the booking back the first time, say 'actually make it {spoken_time(new_hhmm)}' and confirm that instead (it is available).",
                {
                    "tool_calls": steps,
                    "final_state": fs,
                    "max_turns": 14,
                    "corrections_hold": [{"field": "slot_start", "value": new_slot}],
                },
                script=[
                    {
                        "trigger": f"agent reads back the {spoken_time(hhmm)} booking",
                        "do": f"say 'actually make it {spoken_time(new_hhmm)}', then confirm the corrected read-back",
                    }
                ],
            )
        others = [
            x
            for x in BRANCH_WORDS
            if x != b
            and hhmm in self.sb.open_slots(x, d)
            and (cust.customer_id, x, service, d.isoformat()) not in self.sigs
        ]
        if not others:
            return None
        nb = self.rng.choice(others)
        self.used.add((cust.customer_id, nb, service, d.isoformat()))
        steps[1]["args"]["branch_id"] = nb
        steps[2]["args"]["branch_id"] = nb
        fs["appointments_new"][0]["branch_id"] = nb
        return self.base(
            "correction",
            f"{cust.first_name} corrects the branch at the read-back",
            "The corrected branch must be the one booked",
            self.caller(cust),
            f"Book a {SERVICE_WORDS[service]} on {spoken_date(d)} at {spoken_time(hhmm)}. First ask for {BRANCH_WORDS[b]}; when the agent reads the booking back, say you meant {BRANCH_WORDS[nb]} and confirm {BRANCH_WORDS[nb]} at the same time (available).",
            {
                "tool_calls": [steps[0], steps[2]],
                "final_state": fs,
                "max_turns": 14,
                "corrections_hold": [{"field": "branch_id", "value": nb}],
            },
            script=[
                {
                    "trigger": f"agent reads back {BRANCH_WORDS[b]}",
                    "do": f"say 'sorry, {BRANCH_WORDS[nb]}, not {BRANCH_WORDS[b]}' and confirm",
                }
            ],
        )

    def ambiguous_branch(self):
        cust = self.customer()
        target = self.rng.choice(CONFUSABLE)
        parts = self.book_parts(cust, branch=target)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        return self.base(
            "ambiguous_branch",
            f"{cust.first_name} says 'the mar- street branch'",
            "Agent must ask which of Market, Marlow, Marker before checking or booking",
            self.caller(cust, extra={"branch_you_mean": BRANCH_WORDS[b]}),
            f"Book a {SERVICE_WORDS[service]} on {spoken_date(d)} at {spoken_time(hhmm)} at 'the mar street branch', said exactly like that. Only when the agent asks which branch, say {BRANCH_WORDS[b]}. The time is available; confirm the read-back.",
            {"tool_calls": steps, "final_state": fs, "max_turns": 14},
            script=[
                {"trigger": "agent asks which branch you mean", "do": f"say {BRANCH_WORDS[b]}"}
            ],
        )

    def tool_timeout(self):
        cust = self.customer()
        parts = self.book_parts(cust)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        tool = self.rng.choice(["check_availability", "book_appointment"])
        failed = {"tool": tool, "args": {"branch_id": b}, "ok": False}
        idx = 1 if tool == "check_availability" else 2
        steps = steps[:idx] + [failed] + steps[idx:]
        return self.base(
            "tool_timeout",
            f"{tool.replace('_', ' ')} times out once for {cust.first_name}",
            "Agent must say it could not complete the step, retry once, never invent a result",
            self.caller(cust),
            f"Book a {SERVICE_WORDS[service]} at {BRANCH_WORDS[b]} on {spoken_date(d)} at {spoken_time(hhmm)} (available). If the agent says something did not go through, say 'okay, try again'. Confirm the read-back.",
            {
                "tool_calls": steps,
                "final_state": fs,
                "max_turns": 14,
                "forbidden_tools": ["request_handoff"],
            },
            inject=[{"tool": tool, "call_index": 1, "error": "timeout"}],
        )

    def duplicate(self):
        if self.short or self.rng.random() < 0.5:
            cust = self.customer(with_appointment=True)
            ap = self.sb.customer_appointments(cust.customer_id)[0]
            return self.base(
                "duplicate",
                f"{cust.first_name} asks for a booking that already exists",
                "Same service same day: agent must surface the existing appointment, not book again",
                self.caller(cust),
                f"Ask to book a {SERVICE_WORDS[ap.service_type]} at {BRANCH_WORDS[ap.branch_id]} on {spoken_date(ap.slot_start.date())}, any time. When the agent points out you already have one that day, say 'oh right, keep that one' and finish the call.",
                {
                    "tool_calls": [self.lookup_step(cust)],
                    "max_successful_calls": {"book_appointment": 0},
                    "forbidden_tools": ["cancel_appointment", "reschedule_appointment"],
                    "final_state": {},
                    "max_turns": 10,
                },
            )
        cust = self.customer()
        parts = self.book_parts(cust)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        return self.base(
            "duplicate",
            f"{cust.first_name} repeats the request after it is booked",
            "After a successful booking the caller asks again; agent must recognise it, not book twice",
            self.caller(cust),
            f"Book a {SERVICE_WORDS[service]} at {BRANCH_WORDS[b]} on {spoken_date(d)} at {spoken_time(hhmm)} (available). After it is booked, ask again 'and can you book me the {SERVICE_WORDS[service]} {spoken_date(d).split(',')[0]} at {spoken_time(hhmm)}' as if you had not asked; accept the answer.",
            {
                "tool_calls": steps,
                "final_state": fs,
                "max_successful_calls": {"book_appointment": 1},
                "max_turns": 14,
            },
        )

    def multi_intent(self):
        cust = self.customer()
        parts = self.book_parts(cust)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        cats = [c for c in TICKET_GOALS if ("ticket", cust.customer_id, c) not in self.sigs]
        cat = self.rng.choice(cats)
        steps = [
            steps[0],
            {
                "tool": "create_ticket",
                "args": {"customer_id": cust.customer_id, "category": cat},
                "ok": True,
            },
        ] + steps[1:]
        fs["tickets"] = [{"customer_id": cust.customer_id, "category": cat}]
        return self.base(
            "multi_intent",
            f"{cust.first_name} has two things: a {cat.replace('_', ' ')} and a booking",
            "Two intents in one call; both completed, read back separately",
            self.caller(cust),
            f"Two things. First, {TICKET_GOALS[cat]}; agree when the agent offers to open a ticket. Second, book a {SERVICE_WORDS[service]} at {BRANCH_WORDS[b]} on {spoken_date(d)} at {spoken_time(hhmm)} (available). Confirm each read-back.",
            {"tool_calls": steps, "final_state": fs, "max_turns": 16},
        )

    def needs_person(self):
        roll = self.rng.random()
        if roll < 0.3:
            cust = self.customer()
            if ("ticket", cust.customer_id, "unauthorized_transaction") in self.sigs:
                return None
            card = self.sb.customer_cards(cust.customer_id)[0].card_last4
            return self.base(
                "needs_person",
                f"{cust.first_name} reports charges they did not make",
                "Fraud: ticket unauthorized_transaction and handoff suspected_fraud; no bookings",
                self.caller(cust),
                f"After verification, say there are two charges on your card ending {spell_digits(card)} that you did not make. Agree to a ticket and accept being transferred to the fraud team.",
                {
                    "tool_calls": [
                        self.lookup_step(cust),
                        {
                            "tool": "create_ticket",
                            "args": {
                                "customer_id": cust.customer_id,
                                "category": "unauthorized_transaction",
                            },
                            "ok": True,
                        },
                        {
                            "tool": "request_handoff",
                            "args": {"reason": "suspected_fraud"},
                            "ok": True,
                        },
                    ],
                    "handoff": {"required": True, "reason": "suspected_fraud"},
                    "forbidden_tools": [
                        "book_appointment",
                        "reschedule_appointment",
                        "cancel_appointment",
                    ],
                    "final_state": {
                        "tickets": [
                            {
                                "customer_id": cust.customer_id,
                                "category": "unauthorized_transaction",
                            }
                        ]
                    },
                    "max_turns": 12,
                },
            )
        if roll < 0.55:
            cust = self.customer()
            if ("ticket", cust.customer_id, "card_lost_stolen") in self.sigs:
                return None
            card = self.sb.customer_cards(cust.customer_id)[0].card_last4
            where = self.rng.choice(
                ["at a restaurant", "in a taxi", "at the gym", "somewhere on the bus"]
            )
            return self.base(
                "needs_person",
                f"{cust.first_name} lost a card",
                "Lost card: card_lost_stolen ticket then handoff; a banker blocks it; no block claim",
                self.caller(cust),
                f"You lost your debit card ending {spell_digits(card)} {where} and want it dealt with. No strange charges, do not mention any. Agree to the ticket and accept the transfer to a banker.",
                {
                    "tool_calls": [
                        self.lookup_step(cust),
                        {
                            "tool": "create_ticket",
                            "args": {
                                "customer_id": cust.customer_id,
                                "category": "card_lost_stolen",
                            },
                            "ok": True,
                        },
                        {"tool": "request_handoff", "args": {}, "ok": True},
                    ],
                    "handoff": {"required": True, "reason": None},
                    "forbidden_tools": [
                        "book_appointment",
                        "reschedule_appointment",
                        "cancel_appointment",
                    ],
                    "final_state": {
                        "tickets": [
                            {"customer_id": cust.customer_id, "category": "card_lost_stolen"}
                        ]
                    },
                    "max_turns": 12,
                },
            )
        if roll < 0.8:
            cust = self.customer()
            return self.base(
                "needs_person",
                f"{cust.first_name} asks for a person twice",
                "First ask: offer to help once; second ask: handoff caller_requested; no writes",
                self.caller(cust),
                "Say 'i'd like to speak to a person please' as your first request. If the agent offers to help, say 'no thank you, a person please'. Give your phone number and date of birth if asked but do not explain the issue. Accept the transfer.",
                {
                    "tool_calls": [
                        {
                            "tool": "request_handoff",
                            "args": {"reason": "caller_requested"},
                            "ok": True,
                        }
                    ],
                    "handoff": {"required": True, "reason": "caller_requested"},
                    "forbidden_tools": WRITE_TOOLS,
                    "final_state": {},
                    "max_turns": 8,
                },
            )
        phone = "5559" + "".join(str(self.rng.randint(0, 9)) for _ in range(6))
        dob = date(self.rng.randint(1950, 2000), self.rng.randint(1, 12), self.rng.randint(1, 28))
        return self.base(
            "needs_person",
            "A caller whose phone is not on file",
            "Two failed lookups then handoff verification_failed; nothing account-specific said",
            {
                "customer_id": None,
                "name": self.rng.choice(
                    ["Morgan Lee", "Sam Rivera", "Jordan Blake", "Casey Nguyen"]
                ),
                "persona": self.rng.choice(PERSONAS),
                "knows": {
                    "phone": phone,
                    "phone_spoken": spell_digits(phone),
                    "date_of_birth": dob.isoformat(),
                },
                "noise": "none",
            },
            f"Ask about your appointments. Give your phone number {spell_digits(phone)} and date of birth {dob.isoformat()} whenever asked, and repeat them if asked again. When offered a person, accept.",
            {
                "tool_calls": [
                    {"tool": "lookup_account", "args": {"phone": phone}, "ok": False},
                    {"tool": "lookup_account", "args": {"phone": phone}, "ok": False},
                    {
                        "tool": "request_handoff",
                        "args": {"reason": "verification_failed"},
                        "ok": True,
                    },
                ],
                "handoff": {"required": True, "reason": "verification_failed"},
                "forbidden_tools": WRITE_TOOLS,
                "final_state": {},
                "max_turns": 10,
            },
        )

    def noisy_transcript(self):
        cust = self.customer()
        parts = self.book_parts(cust)
        if not parts:
            return None
        service, b, d, hhmm, _slot, steps, fs = parts
        quirks = self.rng.sample(STT_QUIRKS, k=self.rng.randint(2, 3))
        return self.base(
            "noisy_transcript",
            f"{cust.first_name} books over a bad line",
            "Agent must normalise spoken digits and fillers before looking up and booking",
            self.caller(cust, noise="high", quirks=quirks),
            f"Book a {SERVICE_WORDS[service]} at {BRANCH_WORDS[b]} on {spoken_date(d)} at {spoken_time(hhmm)} (available; insist on it). If the agent asks you to repeat something, repeat it.",
            {"tool_calls": steps, "final_state": fs, "max_turns": 14},
        )

    BUILDERS: ClassVar[dict] = {
        "ordinary": ordinary,
        "missing_info": missing_info,
        "correction": correction,
        "ambiguous_branch": ambiguous_branch,
        "tool_timeout": tool_timeout,
        "duplicate": duplicate,
        "multi_intent": multi_intent,
        "needs_person": needs_person,
        "noisy_transcript": noisy_transcript,
    }


def _batch_letters(i: int) -> str:
    a, b = divmod(i, 26)
    return string.ascii_lowercase[a % 26] + string.ascii_lowercase[b]


def _quota(n: int, weights: dict[str, float]) -> list[str]:
    """Family for each of n slots, proportional to weights (largest remainder), in a fixed order."""
    fams = [f for f in FAMILIES if weights.get(f, 0) > 0]
    total = sum(weights[f] for f in fams)
    exact = {f: n * weights[f] / total for f in fams}
    counts = {f: int(exact[f]) for f in fams}
    for f in sorted(fams, key=lambda f: exact[f] - counts[f], reverse=True)[
        : n - sum(counts.values())
    ]:
        counts[f] += 1
    order: list[str] = []
    remaining = dict(counts)
    while any(remaining.values()):
        for f in fams:
            if remaining[f] > 0:
                order.append(f)
                remaining[f] -= 1
    return order


def generate(
    n: int,
    seed: int = 0,
    mix: dict[str, float] | None = None,
    test_dir: str | Path = TEST_DIR,
    profile: str = "default",
    phrasing: bool = True,
) -> list[dict]:
    """n schema-valid generation scenarios; `mix` = family weights (default: equal), met by quota."""
    sb = Sandbox()
    from data.gen.banks import load_bank

    gen = _Gen(
        seed,
        sb,
        test_signatures(test_dir),
        short=profile == "short",
        phrasing=load_bank() if phrasing else None,
    )
    if profile == "short" and mix is None:
        mix = {"ordinary": 5, "needs_person": 3, "duplicate": 1}
    weights = mix or dict.fromkeys(FAMILIES, 1)
    out: list[dict] = []
    counters: dict[str, int] = dict.fromkeys(FAMILIES, 0)
    goals: set[str] = set()
    for fam in _quota(n, weights):
        for _ in range(60):
            d = _Gen.BUILDERS[fam](gen)
            key = (d["goal"], d["caller"]["persona"]) if d else None
            if d is None or key in goals:
                continue
            goals.add(key)
            k = counters[fam]
            counters[fam] += 1
            d["id"] = f"gen_{fam}_{_batch_letters(k // 100)}_{k % 100:02d}"
            out.append(d)
            break
    return out


def write_dir(scenarios: list[dict], directory: str | Path) -> list[Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in scenarios:
        p = out / f"{s['id']}.json"
        p.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        paths.append(p)
    return paths


__all__ = [
    "HANDOFF_REASONS",
    "ONES",
    "PERSONAS",
    "TICKET_CATEGORIES",
    "generate",
    "test_signatures",
    "write_dir",
]
