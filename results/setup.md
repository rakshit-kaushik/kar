# How we measured and what we trained

## The task

A phone agent for Marlow Mutual (a bank) that verifies the caller, then books, reschedules or cancels branch appointments, opens tickets, and hands off to a person. It has seven tools (`lookup_account`, `check_availability`, `book_appointment`, `reschedule_appointment`, `cancel_appointment`, `create_ticket`, `request_handoff`). The full spec is in `CONTRACT.md`.

## The eval

**Test set.** 27 scenarios written by hand: 9 families × 3. They are never used to generate training data.

`ordinary` · `missing_info` · `correction` · `ambiguous_branch` · `tool_timeout` · `duplicate` · `multi_intent` · `needs_person` · `noisy_transcript`

Each scenario sets the caller's persona, goal, the details they know, any planned corrections or detours, and the final state the sandbox must end in. Each one runs with 3 seeds, so a full eval is **81 episodes**.

**Simulator.** A second model plays the caller, and the agent under test talks to it over the same OpenAI-compatible interface it uses live. Each episode gets a fresh sandbox, so every tool result is real. Episodes end on completion, handoff or a turn cap.

**Scorers.** These are deterministic: no LLM judge. An episode passes only if every one of these passes:

| Scorer | Checks |
|---|---|
| `final_state` | sandbox ends in the expected state |
| `tool_calls`, `valid_tool_calls` | right tools, right normalized arguments |
| `no_success_claim_before_result` | never says "booked" or "transferred" before the tool returned it |
| `corrections_hold` | a corrected detail is still correct at the final write |
| `handoff` | hands off when required and never when not |
| `verified_before_account_facts` | no account details before verification |
| `forbidden_tools`, `max_successful_calls`, `max_turns`, `must_mention`, `no_jargon`, `calendar_consistency` | scenario-specific rules |

Two more are reported but never fail an episode: `brevity` (at most two sentences or 35 words) and `reasked_given_slot` (asked again for something the caller already gave).

**Callers used.** The hosted model (Claude Sonnet 5.5) played the caller at first. After the hosted credits ran out, local Qwen3-4B played the caller for the final comparison. That caller gets its own phone number or date of birth wrong in about 20% of episodes. Those episodes are removed from every model's score together (`r3-eval compare --exclude-caller-faults`).

## The training data

**How it was made.** The simulator doubled as the generator. The hosted model played the agent, following the conversation policy, against generation-only scenarios (separate from the 27 test scenarios). Every tool result in the data came from the sandbox.

- **Filter, don't fix:** any episode that failed a scorer was dropped.
- **Personas:** seeded from Taskmaster and SGD utterances.
- **STT shaping:** caller turns were rewritten to look like Deepgram output (lowercase, filler words, numbers spelled out, occasional homophones).
- **Targets per conversation:** at most 3 (a tool call, a clarification and the closing line). Each target is a separate example: the conversation so far is the prompt, and loss falls only on that assistant turn, tool-call tokens included and tool results never.

**Versions.** v1 to v5. Run 1 trained on v2; run 2 trained on v5.

| | v2 (run 1) | v5 (run 2) |
|---|---|---|
| Episodes generated → kept | 1,346 → 1,182 (88%) | 1,436 → 1,250 (87%) |
| Targets generated → kept | 4,858 → 2,587 | 5,302 → 2,708 |
| Kept: tool call / clarification / closing | 1,252 / 1,101 / 234 | 1,331 / 1,141 / 236 |
| Dropped: too long for 2,048 / too wordy | 1,545 / 726 | 1,845 / 749 |
| Training examples used | 2,224 | 2,410 |

v5 rebalanced v2 toward the tools run 1 got wrong (`check_availability` ~570 targets, `book_appointment` 210, `request_handoff` 194) and added five scenario profiles drawn from run 1's failure labels. A 50-episode spot-check was read by hand on v1.

**Coverage gaps that matter:**

- The system prompt plus tool schemas take 1,471 tokens, which leaves little of the 2,048 window. **Only 24 of 829 closing lines after a write fit**, so what the agent says after a booking result is barely trained.
- `reschedule_appointment` had 6 targets and `cancel_appointment` 41.
- The data never contained a failed verification (`not_verified`), so the model never saw a second verification attempt.

## The training runs

Every run used the same recipe: Qwen3-4B 8-bit base, LoRA r32 / scale 2.0 / dropout 0.05 on q, k, v, o, gate, up and down, seq 2,048, lr 1e-4 cosine with 3% warmup, gradient accumulation 16, 1 epoch, no packing, thinking off. Each result was fused and re-quantized to 8-bit.

| Run | Data | Optimizer steps | Wall time | Tokens/s | Peak memory | Val loss | Model |
|---|---|---|---|---|---|---|---|
| Smoke (G2) | 88 examples | 4 | 5 min | 361 | 11.4 GB | 2.23 → 0.61 | `r3-voice-efc3a5e9f9` |
| Run 1 v1 | 2,044 | stopped at step 60 | — | — | — | — | missing timeout and handoff targets |
| Run 1 v2 | 2,224 | 139 | 115 min | 597 | 11.7 GB | 2.25 → 0.07 | `r3-voice-206506525c` |
| **Run 2 v5** | 2,410 | 151 | 117 min | 651 | 11.7 GB | 2.19 → 0.07 | **`r3-voice-730f8ba68e`** |

About 4.1 GPU hours went to training in total. Run 2 changed only the data. Both runs finished at the same validation loss, yet run 2 scored nearly four times as many full passes, which is why selection used the eval runner and never loss.
