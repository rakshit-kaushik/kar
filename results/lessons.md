# What went wrong, and what to keep in mind next time

## Training and data

1. **The prompt ate the context window.** The system prompt and seven tool schemas take 1,471 of the 2,048 training tokens. Long calls overflowed, so only 24 of 829 closing lines after a write survived as targets. The agent barely learned what to say after a booking or a ticket. Every closing line fits at 3,072 tokens; 545 fit at 2,304.
2. **The first data set was missing whole situations.** Run 1's first attempt was stopped at step 60 because it had no tool-timeout or non-caller handoff targets. Other gaps went unnoticed until the eval:
   - no failed verification (`not_verified`) anywhere in the data;
   - only 6 `reschedule_appointment` targets;
   - no transfer after a ticket.
3. **Run 1 learned bad habits.** It claimed actions it hadn't taken (7 episodes) and re-asked details the caller had already given (19 of 48). Its tool-call accuracy fell *below* base, from 53% to 20%. Rebalancing the data for run 2 fixed all three. Changing only the data was the right call.
4. **Loss said nothing.** Run 1 and run 2 both ended at a validation loss of 0.07, but scored 16% and 62% on the eval.
5. **The eval was noisier than its tables look.**
   - Once the hosted credits ran out, the local 4B caller played the caller, and it got its own details wrong in about 20% of episodes. That left `duplicate` and `noisy_transcript` with one clean episode each.
   - Base was scored on 17 episodes and the precision gate on 17 pairs.
   - The hosted baseline used a different caller from the local models.
   - The hosted endpoint ignored `temperature`, so seeds were the only source of variation.
   - Two analysis mistakes were caught and fixed: an unpaired precision gate, and tables built from Postgres that merged repeats.
6. **Some wins belong to agent code, not the model.** Booking-timeout recovery (0/5 → 5/5) and the fraud transfer (1/5 → 5/5) came from rules added to the agent, and the claim guard stepped in on 7 of 30 calls. The reports keep these separate; any summary should too.

## Operations

1. **The wrong model was served under the right name, twice.**
   - 07:10: run 1 was served as `r3-voice-base`, and 816 rows had to be relabelled.
   - 08:02–10:52: run 1 was served as run 2. **Every live number in that window is run 1's**, including the 30-call rehearsal. Run 2 was redeployed at 10:52 with a check of the loaded weights (`scripts/serving_proof.py`).

   The cause both times was the same: a stale `.env` value overrode the model folder chosen at deploy time. Both reports were written before the second case was found.
2. **The GPU schedule was enforced by a lock that leaked.**
   - A rehearsal cut off a baseline eval (01:32).
   - A test killed the live server (01:37).
   - Run 2 took the lock early (05:13).
   - A test suite released the real lock (07:14), and a leftover process grabbed the lock for nothing.
3. **The prefix cache failed silently for hours.** Every lookup raised an error that the server caught and logged at DEBUG. Once a one-file fix was found at 02:50, model time to first token dropped from about 0.7 s to 0.15 s.
4. **The server limited the design.** vllm-mlx 0.5.0 handles one request at a time and answers 503 to a second request unless told to wait, which killed one eval batch (04:22). Four concurrent calls was never reachable on this version.
5. **Vendor credits ran out mid-sprint.**
   - Anthropic: the hosted caller, the baseline, the fallback and any run 3 teacher data.
   - Cartesia: the agent's voice.

   This forced the weaker local caller and Deepgram TTS, and ruled out a run 3.
6. **The network was a phone hotspot.** It added about 0.3 s to every turn and dropped one live call.
7. **Results lived outside the repo.** Reports were written to `$R3_ARTIFACTS/runs`, the training branch is not merged to main, and nothing was pushed.

## Keep in mind next time

**Before training**
- Measure the prompt in tokens before writing the data. Keep the system prompt plus schemas well under half the training length, or train at 3,072.
- Run a coverage check before any GPU run: targets per family, per tool and per error code, with a minimum for each. It would have caught run 1 v1, `not_verified` and reschedule.
- Keep what worked: filter don't fix, change one thing per run, and select by the eval, never by loss.

**The eval**
- Use the same strong caller for every model, and don't let a credit balance decide which one.
- Check caller faults per family before trusting a score, and add seeds or scenarios to any family left with fewer than about 5 clean episodes.
- Compare paired episodes by default, and put the episode count beside every percentage.
- Report the model alone and the model with agent rules as separate columns.

**Operations**
- Make one deploy command the only way a model gets served. Have the server report the path of the weights it loaded, and log a hash of those weights on every row, not a name.
- Start every measurement batch by checking the served weights against the intended ones.
- Build the GPU lock fail-closed from the start, and keep tests away from the real lock file.
- Add to the evening-before preflight:
  - vendor balances, not only rate limits;
  - a wired network;
  - a check that the server actually handles concurrent requests;
  - a cache-hit check in the server logs.
- Write results into the repo and commit at each gate, so the record doesn't depend on reports scattered outside it.
