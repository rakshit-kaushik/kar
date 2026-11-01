You are session D (demo, economics, replay) of the R3 sprint. Work only in this worktree, branch `d/demo`. Start at T+6 or later.

Read first: `CONTRACT.md` §6, §7, `RUNBOOK.md`, then "T+6 to T+24, session D" in `master-build-plan.md`. The UI itself is being built by the human on a separate branch (Astra); your job is the data it reads and the pieces around it.

## Exit criterion
Cost per successfully completed task computed from measured usage for base, adapter and hosted paths; a replay tool that re-runs any recorded transcript through any model URL with the same tools; five scenario scripts; the Postgres views the UI needs.

## Build, in order
1. `db/migrations/00x_views.sql`: views over `turn_events` for per-turn stage latencies, per-call outcome, verified vs claimed facts from `sessions.state`, tool calls with args and results — what a UI polls.
2. Cost ledger: per call, Deepgram minutes, Cartesia characters, LiveKit minutes, LLM tokens; local path = Mac hourly cost ÷ measured calls per hour at the admission limit; hosted at list price; `measured` flag honest on every row; headline = cost per successful task.
3. `eval/replay.py` (coordinate with E's runner): replay a recorded caller transcript through any model URL with the sandbox tools, for side-by-side.
4. Five scenario scripts in `demo/scenarios/`: ordinary booking; correction mid-flow; tool timeout with recovery; interruption; handoff — each as caller lines, expected agent actions, required final state.

## Rules
- Everything shown is labeled live, recorded or scripted. Never a replay presented as live.
- Files you may touch: `db/migrations/00x_views.sql`, `demo/**`, `eval/replay.py`, `services/ledger/**`. Not `schemas/`, `CONTRACT.md`, `train/`.
