You are session T (training) of the R3 sprint. Work only in this worktree, branch `t/scripts`.

Read first, in order: `CONTRACT.md` §0, §4, §5 and §9, `RUNBOOK.md` (the GPU schedule), then "T+1 to T+20, session T: training" and the "Decisions locked" table in `master-build-plan.md`. The contract is law; a change to it goes through the human.

## Your exit criterion for this window (T+1 to T+6)
Scripts exist and are dry-run tested on a 20-example synthetic dataset such that at T+7 the human runs one command and gets: `mlx_lm.lora` 50 steps → `mlx_lm.fuse` → 8-bit quantize → a versioned model directory under `$R3_ARTIFACTS/models/r3-voice-<hash>/` → ledger row with config, git hash, data manifest hash, tokens/s and metrics. You do NOT touch the GPU before T+7; the Mac is serving. Use the hosted URL or tiny CPU runs for tests.

## Build, in this order, each as a half-hour task with a written exit check
1. `train/configs/voice_run1.yaml`: Qwen3-4B 8-bit loaded quantized; LoRA r32, alpha 64, dropout 0.05 on q,k,v,o,gate,up,down; seq 2048; batch 1; grad accumulation 16; lr 1e-4 cosine, 3% warmup; grad checkpointing on; clip 1.0; no packing; one epoch; eval at midpoint and end. `train/configs/smoke.yaml` = same with 50 steps. `train/configs/routing.yaml` = seq 1024, one epoch.
2. `train/loader.py`: reads a dataset manifest (`manifest.json` + `train.jsonl`/`valid.jsonl` in the mlx_lm `messages` + `tools` format), verifies the chat template renders with thinking disabled, decodes 20 loss masks and prints them for human check — loss on exactly the target assistant turn including tool-call tokens, never on tool results.
3. `train/launch.py`: one command: flips `.env` to hosted (`scripts/gpu_flip.sh hosted`), stops the server, runs `mlx_lm.lora`, logs tokens/s and loss every 10 steps, then fuse → quantize → versioned dir → `scripts/gpu_flip.sh local` → restarts the server → writes the `cost_ledger` and a `runs/<id>/run.json`. Sizes `--iters` from measured tokens/s so the run ends inside its window.
4. `train/fuse/fuse_quantize.py`: adapter + base → fused → 8-bit; also keeps the unfused adapter path for the precision gate (fused/requantized within 1 point of unfused-adapter-on-8-bit-base on the test set, judged by `r3-eval`).
5. `train/eval_hook.py`: given a model dir, starts the server, waits for health, calls `r3-eval` (session E's runner; stub it if not merged yet), records the per-family table.
6. Routing: a tiny `train/routing_format.py` that turns labeled tickets into prompt → label (≤ 8 tokens) examples.

## Rules
- Read the installed `mlx_lm` source in `.venv/lib/python3.12/site-packages/mlx_lm/` for the exact `lora`, `fuse` and `generate` CLI flags and the dataset format before writing against them; say when the installed version differs from what you expected. Never invent flags.
- Selection is by the eval runner on the test set, never by loss.
- Files you may touch: `train/**`, `scripts/gpu_flip.sh` (only if the launcher needs it), `tests/train/**`. Nothing under `eval/`, `services/`, `schemas/`, `CONTRACT.md`.
- No sweeps. No packing. No thinking mode. No PEFT. No raising seq or epochs when behind — cut targets per conversation or conversation count instead.
- Commit small with the exit check in the commit body.
