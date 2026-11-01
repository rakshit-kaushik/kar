# Demo scenarios (five, scripted; plan T+20.5–22)

Sandbox clock: Monday, November 2, 2026, 10:00. Branches open 9:00–17:00 weekdays, 9:00–13:00 Saturday. Tuesday = Nov 3,
Wednesday = Nov 4. Marker Street is blocked Tuesday morning; Mill Road is blocked all of Nov 4. Customers (eval/sandbox/seed.json):
Dana Whitfield C-1001, phone 555-014-2219, DOB March 12 1988, Market Street, no appointments; Marcus Okafor C-1002,
phone 555-019-8823, DOB July 4 1975, cards ending 7781 and 0034, notary appointment A-2001 at Marlow Street on Nov 5 at 14:00.

Every stage run is labelled **live**, **recorded** or **replay**; a replay is never shown as a live call.

How to run each scenario:
- **Live** (the user speaks the lines): `uv run python scripts/join_token.py <room>` → browser link; the agent joins. For the
  timeout scenario the room name carries the fault, e.g. `demo-timeout--fault-book_appointment-timeout`.
- **Simulated audio call** (unattended): `uv run python scripts/sim_caller.py --calls 1 --lines-file demo/turns/<name>.json --prefix demo-<name>`
  (`--interrupt-turn 4 --interrupt-after-ms 2500` for the interruption; the fault spec goes in `--prefix` for the timeout).
- **Replay through any model** (text, side by side): `uv run python scripts/replay.py --turns-file demo/turns/<name>.json --model base=... --model adapter=... --model hosted=...`
- Latency per stage: `uv run python scripts/g1_report.py --like 'demo-<name>%'`; cost: `uv run python scripts/cost_report.py --like 'demo-%'`.
- **Unattended rehearsal** (the plan's T+22–24 run): `RUNS=5 scripts/rehearsal.sh` under lock S (~50 min for 25 calls): per scenario the
  simulated caller with `--record-dir` (two-channel wav per call, left caller / right agent), `scripts/demo_check.py` (PASS/FAIL per call
  against the final-state column below, plus no success claim without a tool result, no ids spoken, no structural text), g1_report latency
  and cost_report into the ledger under `rehearsal-<model>-<stamp>`; output `$R3_ARTIFACTS/runs/rehearsal-<model>-<stamp>/`.
- The interruption cut-in defaults to caller turn 5 (the read-back, `INT_TURN`): the run-1 candidate answers the slot question in one
  sentence, too short to cut into at turn 4 (measured Nov 1 04:36). A recording made by the simulated caller is labelled **recorded**.

| # | Scenario | Caller lines | What the agent must do | Final state | Must never |
|---|---|---|---|---|---|
| 1 | Ordinary booking | `demo/turns/ordinary.json` (Dana): phone; DOB; mortgage consultation Tuesday morning at Market Street; "nine thirty is fine"; "yes, that's right"; goodbye | `lookup_account(5550142219, 1988-03-12)` → verified; `check_availability(market_st, mortgage_consultation, 2026-11-03)`; offer slots in one or two sentences; read back Tuesday Nov 3, 9:30, Market Street, mortgage consultation; after "yes" `book_appointment(C-1001, market_st, mortgage_consultation, 2026-11-03T09:30)`; confirm in one sentence; close | one new appointment for C-1001 at market_st, 2026-11-03T09:30; no ticket; no handoff | say "booked" before book_appointment returns; speak ids or tool names; ask two questions in one turn |
| 2 | Correction mid-flow | `demo/turns/correction.json`: Tuesday requested, then "actually, sorry, i meant wednesday, not tuesday"; "nine thirty works"; "yes" | acknowledge the correction in one sentence; `check_availability(..., 2026-11-04)`; read back **Wednesday Nov 4** 9:30; `book_appointment(..., 2026-11-04T09:30)` | exactly one appointment, on 2026-11-04; none on Nov 3 | book Tuesday (stale detail); read back Tuesday after the correction |
| 3 | Transient write fault absorbed by the executor (one retry, same idempotency key) | `demo/turns/timeout.json` (…; "yes"; "yes, please try again"; goodbye); room/prefix `demo-timeout--fault-book_appointment-timeout` (the first book_appointment call returns `timeout`) | after the read-back and "yes", the first `book_appointment` answers `timeout`; the tool executor retries once with the same idempotency key (events: tool_end attempt 1 failed, `tool_retry`, tool_end attempt 2 ok; the scorecard shows `retry=1`) and the model sees only the final result, so it confirms the booking. This is the agent's recovery, not the model's: without the retry the final model asked "Would you like me to try again?" and then never retried when told yes (0/5 in the 08:40 rehearsal). A double fault still reaches the model, which must say it could not book and `request_handoff(tool_failure, …)` with no booking claimed; if the retry fails too, `request_handoff(tool_failure, ...)` with no booking claimed | booking on the retry path (or a handoff with no booking if both fail) | invent slots; say the booking is done after an error; stay silent for more than a sentence |
| 4 | Interruption | `demo/turns/interruption.json` with the caller cutting in at turn 4 ("nine thirty is fine") after ~2.5 s of the agent's slot read-out | stop speaking within ~0.5 s of the caller's onset; treat the half-spoken list as not confirmed; read back 9:30 and book after "yes" | as scenario 1 | keep talking over the caller; book without the read-back |
| 5b | Handoff (caller asks for a person, twice) | `demo/turns/handoff_person.json` (Dana): phone; DOB; "i'd like to speak to a person, please"; "no, i'd really rather talk to a person about it"; goodbye | `lookup_account(5550142219, 1988-03-12)` → verified; may offer to help once; on the second ask `request_handoff(caller_requested, <summary>)` and tell her a colleague will take over | handoff requested with reason caller_requested; no ticket; appointments unchanged | keep deflecting after the second ask; book or open anything; speak ids |
| 5 | Handoff (suspected fraud) | `demo/turns/handoff.json` (Marcus): phone; DOB; unrecognized ~$200 charge on the card ending 7781 yesterday; "yes please"; goodbye | `lookup_account(5550198823, 1975-07-04)` → verified; read back the ticket (unauthorized transaction, card ending 7781, about $200, yesterday); after "yes please" `create_ticket(C-1002, unauthorized_transaction, <summary>)` then `request_handoff(suspected_fraud, <summary>)`; tell him a specialist will take over | one ticket (unauthorized_transaction) for C-1002; handoff requested; appointments unchanged | say the charge is reversed, the card blocked or the money back (no tool does that); mention the ticket id |

Scoring on stage: the eval runner's deterministic scorers (final state, tool names and arguments, no unsupported success claim, turn
budget, handoff when required and never when not, corrected details retained) plus the three-line human rubric (usefulness of the
question asked, spoken brevity, naturalness). Each scenario is run five times on the frozen build and recorded (`$R3_ARTIFACTS/runs/demo-*`).

## Flows chosen, and why (Nov 1 07:55, reviewer)
Run 2 (the likely final model) holds the direct flows and is weak on the indirect ones: in the text eval the handoff after a
ticket is missed in most seeds when the request is indirect (a lost card already blocked, a lost card revealed slowly, a
second intent attached to a confirmation) and after a double tool fault, while the direct fraud report (an unrecognized
charge, ticket, then a suspected-fraud transfer) and the twice-asked "a person, please" hold. So the demo's handoff scenario
(#5) is the direct fraud report, and the timeout scenario (#3) is a single check_availability fault with recovery, not the
double fault. These are the flows chosen, labelled here as such; the weak families stay in the write-up's failure table.

Two changes after the first live calls on the final model (Nov 1 08:02-08:30). The timeout fault moved from check_availability to
book_appointment: run 2 books directly once the caller has named a time and did not call check_availability in any of its 7 live
bookings, so a check_availability fault was never consumed and the scenario measured nothing. And 5b was added as the reviewer's
alternative handoff flow (the person asked for twice): the direct fraud report (5) opened the ticket but did not transfer in the
one live call on run 2, which is E's main remaining failure family; the rehearsal runs both and the demo uses whichever passes.
The scenario checker now waits for a call's final state before scoring it (the agent writes it once, at call end).
