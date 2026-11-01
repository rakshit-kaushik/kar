You are session E (eval + data) of the R3 sprint. Work only in this worktree, branch `e/sim`.

Read first, in order: `CONTRACT.md` (all of it), `schemas/tools/*.json`, `schemas/state.json`, `RUNBOOK.md`, then the sections "T+1 to T+5, session E" and "T+5 to T+9, session E: data" in `master-build-plan.md`. The contract is law; if you need a change to it, stop and ask the human, never edit it.

## Your exit criterion for this window (T+1 to T+5)
`uv run r3-eval --base-url <any OpenAI-compatible URL> --model <id> --scenarios eval/scenarios/test/ --out $R3_ARTIFACTS/runs/<id>/` scores a model on the test set with the contract's tools and writes per-scenario, per-scorer rows to `eval_results` in Postgres, and a baseline table exists for the hosted model in text (base model text baseline runs once the local server is up — ask the human).

## Build, in this order, each as a half-hour task with a written exit check
1. `eval/sandbox/`: Faker-seeded Marlow Mutual data (customers, cards, appointments, 4 branches per the contract, catalog, hours), loaded into the sandbox tables; `reset()` per episode; the eight tools implemented against it with the contract's error codes, idempotency keys, and a `timeout` injection hook. Unit tests for every error path.
2. `eval/simulator/`: scenario spec (persona, goal, known facts, planned corrections and detours, noise level, injected tool failures) → the hosted model (`HOSTED_LLM_*` in `.env`) plays the caller; the agent under test is any OpenAI-compatible URL with the contract's tools; episodes end on completion, handoff or a 20-turn cap. Full transcript + tool log + final sandbox state saved as JSON.
3. `eval/scorers/`: deterministic only — final sandbox state matches expected; tool name + normalized args per step; no success claim in assistant text before the matching tool result; turn count within budget; handoff when required and never when not; corrected details still correct at the final action; no account-specific statement before `verified: true`; no jargon words (tool names, "system", "function", "JSON").
4. `eval/runner.py` (the `r3-eval` CLI): runs a scenario directory against a URL, N episodes in parallel, writes `eval_results` and a markdown summary per family.
5. Three draft scenarios per family in `eval/scenarios/gen/` as a smoke set. The human writes the 27 real test scenarios in `eval/scenarios/test/` at T+5; never generate into that directory.
6. Baseline: hosted model in text on the smoke set; ranked failure list (frequency × cost) into the summary.

## Rules
- Before calling any library API (openai, psycopg, faker), read the installed source in `.venv/lib/python3.12/site-packages/` and say if the version differs from what you expected. Do not invent flags.
- Every assistant turn the simulator records must carry the exact `tool_calls` objects the model returned; this data becomes training data at T+5.
- Files you may touch: `eval/**`, `data/**`, `tests/eval/**`. Nothing under `services/`, `train/`, `schemas/`, `CONTRACT.md`.
- Hand background work (running 20 episodes, a baseline sweep) to a subagent and keep building.
- Commit small, with the exit check pasted in the commit body. The human merges to main at gates.
