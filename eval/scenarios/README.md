# Scenarios

One YAML (or JSON) file per scenario. `gen/` holds machine-drafted smoke/generation scenarios;
`test/` is the human-written test set (27, three per family) and is never generated into.

```yaml
id: correction-01              # unique; convention <family>-NN
family: correction             # ordinary | missing_information | correction | ambiguous_branch |
                               # tool_timeout | duplicate_request | multi_intent_detour |
                               # needs_person | noisy_transcript
title: one line
today: 2026-11-02              # the sandbox calendar; Mon–Fri 9–17, Sat 9–13, Sun closed
now: "09:00"                   # slots at or before now are not offered
seed: 42                       # sandbox seed; C-1042 Dana (5550142219, 1988-03-12) always exists
customer_id: C-1042            # the caller's record (seed or fixture); null for non-customers
known_facts: {phone: "5550142219", date_of_birth: "1988-03-12", name: Dana}   # what the caller knows
persona: {name: Dana, age: 38, style: terse, notes: ...}
goal: what the caller wants, in plain words (the caller model reads this)
script_hints: [free-text instructions for the caller model]
corrections:                   # the caller changes a detail; `to` must win at the final write
  - {when: after the agent reads back two o'clock, field: time, from: "14:00", to: "14:30", say: actually make it two thirty}
detours: [{when: once verified, say: what are your saturday hours, resolution: then continue}]
noise_level: 0                 # 0 clean … 3 heavy STT noise (fillers, homophones, drops)
injected_failures: [{tool: book_appointment, nth_call: 1}]   # that call returns timeout
fixture:                       # layered over the seed before the episode
  customers: [{customer_id: C-9001, first_name: Lee, phone: "5550000001", date_of_birth: "1970-01-01", branch_id: mill_rd, cards: ["1111"]}]
  cards: [{customer_id: C-1042, card_last4: "4219", status: blocked}]
  appointments: [{customer_id: C-1042, branch_id: marlow_st, service_type: notary, slot_start: "2026-11-05T14:00"}]
  free: [{branch_id: marlow_st, date: 2026-11-06, slots: ["14:00", "14:30"]}]   # guarantee open
  unavailable: [{branch_id: marlow_st, date: 2026-11-06}]                       # fill everything else
expected:
  max_turns: 12                # caller turns; the episode hard cap is 20
  handoff: false               # required? (reason optional)
  handoff_reason: suspected_fraud
  tool_steps:                  # ordered subsequence of real calls; args are a normalized subset;
    - {name: lookup_account}   # `date`/`time` match the slot's date/time part; ok: true|false filters
    - {name: book_appointment, args: {branch_id: marlow_st, slot_start: "2026-11-06T14:30"}}
  must_not_call: [cancel_appointment]
  corrected: {time: "14:30"}   # defaults to corrections[].to
  final_state:
    appointments:              # the customer's COMPLETE set of booked appointments at the end
      - {customer_id: C-1042, branch_id: marlow_st, service_type: notary, slot_start: "2026-11-06T14:30"}
      - {customer_id: C-1042, service_type: notary, slot_start: "2026-11-05T14:00", status: cancelled}
    cards: [{card_last4: "4219", status: blocked}]
    tickets: [{category: unauthorized_transaction}]   # exact count for the customer
```

Scorers: final_state, tool_steps, no_unsupported_success_claim, turn_budget, handoff,
corrected_details, verified_before_account_facts, no_jargon, brevity. See `eval/scorers`.

Open slots for seed 42 / today 2026-11-02 can be listed with:
`uv run python -c "from datetime import date; from eval.sandbox import Sandbox; s=Sandbox(); print(s.open_slots('marlow_st', date(2026,11,6)))"`
