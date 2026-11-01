# R3 CONTRACT — retail-bank phone line, appointment and card service

Status: DRAFT v0.1 for review · Oct 31, 2026 · Every session reads this before writing a line. A session that needs a change here stops and asks; it never edits this file.

## 0. Fixed facts

| Item | Value |
| --- | --- |
| Machine | M5 Max, 36 GB unified memory (plan assumed 64; micro-batch stays 1, no spec-decode test until load tests say there is headroom) |
| Base model | `mlx-community/Qwen3-4B-8bit`, thinking disabled, served via vllm-mlx (oMLX fallback) |
| Hosted model | Anthropic `claude-sonnet-5-5` through the OpenAI-compatible endpoint; plays the caller, the teacher agent, the fallback, the baseline |
| Voice | Deepgram STT → model → Cartesia `sonic-3.6` TTS inside LiveKit Agents `AgentSession`; browser only |
| Ports | model server 8000 · proxy 8001 · UI 3000 · Postgres 5432 · agent (LiveKit worker) no port |
| Postgres | Homebrew PostgreSQL 15, native, already running; database `r3`; `DATABASE_URL` in `.env`; migrations via `scripts/db_migrate.sh` (no Docker needed) |
| Config keys | `LLM_BASE_URL` / `LLM_MODEL` = active endpoint (flipped by `scripts/gpu_flip.sh`); `LOCAL_LLM_*`, `HOSTED_LLM_*`, `DEEPGRAM_API_KEY`, `CARTESIA_*`, `LIVEKIT_*`, `DATABASE_URL`, `R3_ARTIFACTS` — see `.env.example` |
| Artifacts | `$R3_ARTIFACTS` = `~/r3-artifacts/{models,datasets,runs}`; worktrees hold code only |
| Secrets | `.env` in repo root, gitignored; `.env.example` lists every key |
| Model endpoint | always `LLM_BASE_URL` + `LLM_MODEL` from config; the launcher flips it to the hosted URL during training windows |
| Banned shortcuts | no vLLM (CUDA), no PEFT, no Docker for model processes, no packing, no thinking mode, no multi-LoRA serving |

## 1. Workflow spec

**Business.** Marlow Mutual Bank, a fictional retail bank in the fictional city of Marlow. Four branches, deliberately confusable names: Market Street, Marlow Street, Marker Street, Mill Road. Hours Mon–Fri 9:00–17:00, Sat 9:00–13:00, closed Sun. Appointments are 30-minute slots. All data is synthetic (Faker); no real PII anywhere, ever.

**Callers.** Existing customers calling the service line. They identify with phone number + date of birth. Non-customers get general hours and a handoff.

**Intents (one call may hold two).**

| Intent | What the agent may do | Done when |
| --- | --- | --- |
| book | find a slot, book an appointment for a listed service at a branch | `book_appointment` returned `appointment_id` and it was read back |
| reschedule | move an existing appointment | `reschedule_appointment` ok and read back |
| cancel | cancel an existing appointment | `cancel_appointment` ok and read back |
| status | say when and where an upcoming appointment is | facts from `lookup_account` spoken |
| lost or stolen card | block the card, offer a replacement | `block_card` ok and read back |
| dispute or complaint | open a ticket in the right category | `create_ticket` ok, ticket id read back |
| speak to a person | hand off | `request_handoff` ok, one closing sentence |

**Appointment services and catalog (the only facts the agent may state).**

| service_type | Duration | Fee |
| --- | --- | --- |
| account_opening | 30 min | free |
| mortgage_consultation | 30 min | free |
| loan_application | 30 min | free |
| financial_review | 30 min | free |
| notary | 30 min | $10 per document |
| safe_deposit_box | 30 min | $60 per year, small box |

Card replacement: standard mail 5–7 business days, free; express 2 business days, $25. No other prices, rates or fees exist; anything else is "I can't quote that, a banker can".

**Exceptions.** unknown account (two failed lookups → handoff); no availability (offer next two days or another branch); ambiguous branch (Market vs Marlow vs Marker → ask, never guess); duplicate booking (same service, same day → tool returns `conflict`, ask whether to keep or move); tool timeout (say so, one retry, then handoff); card already blocked (`conflict` → say it is already blocked, offer replacement status); suspected fraud (any mention of transactions the caller did not make → ticket `unauthorized_transaction` *and* handoff).

**Prohibited.** Stating anything account-specific before `lookup_account` returns `verified: true`. Confirming a booking, block, cancellation or ticket a tool has not returned. Prices, hours or policies outside the catalog. Booking outside hours or in the past. Blocking a card without an explicit yes after read-back. Saying tool names, ids as internal jargon, "system", "function", "JSON".

**Handoff rules.** Hand off on: suspected fraud; identity verification failed twice; caller asks for a person twice (first ask → offer to help once); any request outside the intents; a tool failing twice. Never hand off when a listed intent is still completable.

## 2. Conversation policy — ten rules

1. One question per turn. Never two.
2. Nothing account-specific is spoken until `lookup_account` has returned `verified: true`.
3. Before any write (`book`, `reschedule`, `cancel`, `block_card`, `create_ticket`) read back every argument in one sentence and get an explicit yes.
4. Replies are one or two sentences, at most 35 words.
5. Numbers as spoken: times "two thirty in the afternoon", dates "Tuesday, November fourth", card digits "ending in four two one nine", never "14:30" or "11/04".
6. On a correction: acknowledge in half a sentence, replace the fact, re-confirm the whole action before writing. A corrected fact wins over an earlier one everywhere.
7. On a tool failure: say what did not work in plain words, offer one retry or a person; never imply success.
8. Verified facts (from tools) and claimed facts (from the caller) are kept apart; only verified facts are stated as true.
9. Handoff per §1; after `request_handoff` say one sentence and stop.
10. Closing: one sentence summarizing only completed actions, then goodbye. Half-spoken confirmations that were interrupted count as unconfirmed.

## 3. Example dialogues (expected tool calls and final state)

Caller turns are shown STT-shaped (lowercase, no punctuation, digits as words where Deepgram would). Good = passes every scorer. Bad = what the scorers must catch.

**G1 — ordinary booking.**
A: Thanks for calling Marlow Mutual, this is the service line. Can I get the phone number on your account?
C: yeah it's five five five zero one four two two one nine
A: Thanks. And your date of birth?
C: march twelfth nineteen eighty eight
→ `lookup_account(phone="5550142219", date_of_birth="1988-03-12")` → `{verified:true, customer_id:"C-1042", first_name:"Dana", branch_id:"market_st", appointments:[]}`
A: Thanks Dana, you're verified. What can I help with today?
C: i need to see someone about a mortgage next week
A: Happy to. Which day next week works, and is Market Street still the best branch?
C: tuesday morning market street is fine
→ `check_availability(branch_id="market_st", service_type="mortgage_consultation", date="2026-11-04")` → `{slots:["09:00","09:30","11:00"]}`
A: Tuesday I have nine, nine thirty, or eleven in the morning. Which one?
C: nine thirty
A: So that's a mortgage consultation at Market Street on Tuesday, November fourth at nine thirty in the morning. Shall I book it?
C: yes
→ `book_appointment(customer_id="C-1042", branch_id="market_st", service_type="mortgage_consultation", slot_start="2026-11-04T09:30", idempotency_key=…)` → `{appointment_id:"A-7731"}`
A: Booked. You'll see a banker at Market Street on Tuesday at nine thirty. Anything else?
C: no that's it
A: Thanks Dana, see you Tuesday. Goodbye.
Final state: one appointment A-7731; completed_actions=[book]; handoff=false.

**G2 — correction mid-flow.**
…verified as above…
C: book me a notary appointment thursday at two at marlow street
A: Just to check, did you mean Marlow Street or Market Street?
C: marlow street
→ `check_availability(branch_id="marlow_st", service_type="notary", date="2026-11-06")` → `{slots:["14:00","14:30"]}`
A: Two in the afternoon is open. A notary appointment at Marlow Street on Thursday, November sixth at two. Shall I book it?
C: actually make it two thirty
A: Two thirty it is. Notary at Marlow Street, Thursday November sixth at two thirty in the afternoon. Book it?
C: yes
→ `book_appointment(…, slot_start="2026-11-06T14:30", …)` → ok
Final state: appointment at 14:30 (not 14:00); stale-detail scorer passes.

**G3 — lost card with fraud mention → handoff.**
…verified…
C: i lost my card and i see a charge i didn't make
A: I'm sorry to hear that. Which card is it, the last four digits?
C: four two one nine
A: I'll block the card ending in four two one nine right now so nothing else goes through. Go ahead?
C: yes please
→ `block_card(customer_id="C-1042", card_last4="4219", reason="lost", idempotency_key=…)` → `{blocked:true}`
→ `create_ticket(customer_id="C-1042", category="unauthorized_transaction", summary="Lost card 4219; caller reports a charge they did not make", idempotency_key=…)` → `{ticket_id:"T-5520"}`
→ `request_handoff(reason="suspected_fraud", summary="Card 4219 blocked, ticket T-5520 opened, caller reports unauthorized charge")`
A: The card is blocked and I've opened a fraud case, number five five two zero. I'm connecting you to our fraud team now, please hold.
Final state: card blocked; ticket category unauthorized_transaction; handoff=true.

**B1 — unsupported success claim (must fail).**
C: can you book me for friday at ten
A: Done, you're booked for Friday at ten.   ← no `check_availability`, no `book_appointment`, no read-back
Scorers: success-claim-before-tool-result fails; final-state mismatch fails.

**B2 — stale detail after correction (must fail).**
C: thursday at two … actually two thirty
A: Booking notary at Marlow Street Thursday at two. Shall I book?   ← reverted
→ `book_appointment(slot_start="…T14:00")`
Scorers: corrected-detail scorer fails; tool-argument scorer fails.

## 4. Tool schemas

Files in `schemas/tools/*.json` are the source of truth; the simulator, the agent and the eval runner all load the same files. Each is OpenAI function-calling format, kept near 60 tokens of description. Every write takes `idempotency_key` (string, caller-generated UUID); the sandbox returns the original result on a repeated key.

| Tool | Writes | Error codes |
| --- | --- | --- |
| `lookup_account(phone, date_of_birth)` | no | `not_found`, `not_verified`, `timeout` |
| `check_availability(branch_id, service_type, date)` | no | `invalid_argument`, `unavailable`, `timeout` |
| `book_appointment(customer_id, branch_id, service_type, slot_start, idempotency_key)` | yes | `conflict`, `unavailable`, `invalid_argument`, `timeout` |
| `reschedule_appointment(appointment_id, new_slot_start, idempotency_key)` | yes | `not_found`, `conflict`, `unavailable`, `timeout` |
| `cancel_appointment(appointment_id, idempotency_key)` | yes | `not_found`, `timeout` |
| `block_card(customer_id, card_last4, reason, idempotency_key)` | yes | `not_found`, `conflict`, `timeout` |
| `create_ticket(customer_id, category, summary, idempotency_key)` | yes | `invalid_argument`, `timeout` |
| `request_handoff(reason, summary)` | yes | none |

Error shape: `{"error": "<code>", "message": "<short text>"}`. The simulator injects `timeout` on a schedule for the tool-timeout family. `block_card` is the first tool cut if the schema block pushes turn-2 prefill over 150 tokens with the cache on.

## 5. Session state (`schemas/state.json`)

```
verified:   { customer_id, first_name, branch_id, appointments[], cards[] }      # tool-sourced only
claimed:    { phone, date_of_birth, service_type, branch_id, date, time, card_last4, appointment_ref }  # caller-sourced
stage:      greeting | identifying | intent | collecting | confirming | executing | closing | handoff
pending:    { tool, args, read_back_spoken: bool, confirmed: bool } | null
completed:  [ { tool, args, result, ts } ]
handoff:    { requested: bool, reason, ts } | null
model_id, call_id, started_at
```

A write tool may be called only when `pending.confirmed` is true for that tool and args. Checkpointed to Postgres after every tool result and at call end.

## 6. Turn-event schema (`schemas/events.json`)

One row per event, table `turn_events`: `call_id, turn_idx, model_id, event, ts_ms (monotonic), wall_ts, payload jsonb`.

Events, in the order they occur in a turn: `speech_end` · `stt_final` (payload: transcript) · `llm_request` (prompt_tokens) · `llm_first_token` · `tool_start` (name) · `tool_end` (name, ok, ms) · `llm_done` (completion_tokens) · `tts_first_audio` · `playback_end` · plus out-of-order `interruption` (truncated_text) · `fallback` (reason, from_model, to_model) · `call_start` · `call_end` (outcome).

Derived latencies: end-of-speech→first-audio = `tts_first_audio − speech_end`; TTFT = `llm_first_token − llm_request`.

## 7. Interfaces and storage

- The model is always an OpenAI-compatible chat-completions URL with `tools`; streaming; tool calls arrive as `tool_calls` deltas (the proxy on 8001 guarantees this and strips `<think>`).
- Simulator, agent and eval runner import tools from `schemas/tools/`, sandbox data from `eval/sandbox/`.
- Postgres tables: `customers`, `appointments`, `cards`, `tickets` (sandbox seed, reset per episode) · `sessions`, `turn_events`, `eval_results` (run_id, model_id, scenario_id, family, scorer, pass, detail), `cost_ledger` (run_id, model_id, item, qty, unit_cost, measured bool).
- Eval runner CLI: `r3-eval --base-url … --model … --scenarios eval/scenarios/test/ --out runs/<id>/` — scores any URL identically.

## 8. Scenario families (test set = 27, three per family, hand-written, never used for generation)

ordinary · missing information · correction · ambiguous branch · tool timeout · duplicate request · multi-intent detour · needs a person · noisy transcript. Seeds for caller phrasing: Taskmaster-1 and SGD (appointment dialogues) → persona and paraphrase banks; utterances used for test paraphrases never seed training personas.

## 9. Ticket routing taxonomy (12 categories)

| Category | Definition | Examples |
| --- | --- | --- |
| `card_lost_stolen` | card missing or taken; needs block/replacement | lost my debit card; wallet stolen; card never arrived; found my card, unblock it; replacement card request |
| `card_declined` | card refused at a merchant or ATM, card is in hand | declined at the grocery store; ATM won't give cash; card stopped working abroad; chip not reading; online purchase blocked |
| `unauthorized_transaction` | charges the customer did not make; fraud | charge I don't recognize; someone used my card; account drained; subscription I never signed up for; duplicate charge from unknown merchant |
| `fee_dispute` | customer disputes a fee the bank charged | overdraft fee unfair; monthly fee should be waived; ATM fee refund; late fee on credit card; wire fee too high |
| `account_access` | cannot log in or use online/mobile banking | locked out of app; forgot password; two-factor code not arriving; new phone, can't log in; username unknown |
| `statement_document` | wants a statement, tax form or letter | need last year's statements; 1099 form; proof of funds letter; mortgage interest statement; void check |
| `profile_change` | address, phone, name or email update | moved, new address; change phone number; name change after marriage; update email; add joint owner |
| `loan_mortgage` | questions on existing or new loans | mortgage payoff amount; refinance rates; car loan payment date; HELOC application; pre-approval letter |
| `appointment` | wants to book, move or cancel a branch visit | book a notary; reschedule Thursday; cancel my appointment; which branch is open Saturday; see a banker about retirement |
| `transfer_payment` | wires, transfers or bill pay not going through | wire not received; transfer pending for days; bill pay went to wrong payee; Zelle failed; recurring transfer stopped |
| `complaint` | dissatisfaction with service, no transactional ask | rude teller; waited an hour; promised a call back and nobody did; branch closed early; app keeps crashing |
| `other` | none of the above, or unreadable | foreign currency order; notary hours; check cashing policy; empty ticket; spam |

Costly confusions (weighted 3× in the confusion matrix): `unauthorized_transaction` ↔ `fee_dispute` (fraud has a regulatory clock); `card_lost_stolen` ↔ `card_declined` (block vs no block); `complaint` → `other` (lost escalation). Invalid labels become `needs_review`.

## 10. Repo layout and sessions

```
CONTRACT.md RUNBOOK.md .env.example
schemas/{tools/*.json, state.json, events.json}
services/{proxy, agent, tools}      ← S
eval/{simulator, scorers, runner, sandbox, scenarios/{test,gen}}   ← E
data/{gen, shape, format, splits}    ← E
train/{configs, launcher, fuse, eval_hook}   ← T
ui/ (external Astra branch; D wires it to Postgres)   db/migrations   scripts/{restart, gpu_flip}
prompts/{S,E,T,D}.md
```

Sessions: **S** serving · **E** eval + data · **T** training · **D** cost ledger, replay, scenario scripts, UI wiring. One worktree each, branches `s/*`, `e/*`, `t/*`, `d/*`; merges to main by the human at gates G1 T+4, G2 T+8, G3 T+13, G4 T+20.5. Shared state (Postgres, model server, `$R3_ARTIFACTS`, ports) lives outside worktrees.

## 11. Open items for the human

- [ ] Confirm the banking framing and the eight tools (or cut `block_card`)
- [ ] Confirm catalog prices and hours above
- [ ] Write the 27 test scenarios at T+5 (three per family in §8)
- [ ] Confirm routing taxonomy; name any category to drop
