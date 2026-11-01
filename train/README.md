# train/ — session T

One command per GPU window (RUNBOOK.md schedule); everything else is scripts and checks that run off the GPU.

```
# T+7 smoke (G2): 50 steps on E's dataset -> fuse -> 8-bit -> $R3_ARTIFACTS/models/r3-voice-<hash>/ -> ledger
uv run python -m train.launch --config train/configs/smoke.yaml --data $R3_ARTIFACTS/datasets/<name>

# T+10 run 1: one epoch, sized to the 120-minute window from the smoke run's measured tokens/s
uv run python -m train.launch --config train/configs/voice_run1.yaml --data $R3_ARTIFACTS/datasets/<name> --window-minutes 120

# T+13 routing adapter (data from train/routing_format.py)
uv run python -m train.launch --config train/configs/routing.yaml --data $R3_ARTIFACTS/datasets/routing-v1

# score a versioned model: vllm-mlx on :8010 with S's production flags, S's proxy on :8011 in front, r3-eval through the proxy
uv run python -m train.eval_hook --model-dir $R3_ARTIFACTS/models/r3-voice-<hash> --scenarios eval/scenarios/test
uv run python -m train.eval_hook --model-dir $R3_ARTIFACTS/models/r3-voice-<hash> --gate     # + adapter-on-base via mlx_lm.server, 1 point
uv run python -m train.eval_hook --model-dir ... --base-url http://127.0.0.1:8001/v1          # score the live stack instead
```

Server cycle (from any worktree; `--services-root` = the checkout whose .env, restart.sh and gpu_lock.sh are used):
`uv run python -m train.launch --rehearse --services-root /Users/abgup/Documents/Code/kar/kar --config train/configs/smoke.yaml --data -`
takes the GPU lock (`scripts/gpu_lock.sh acquire T …`, waiting while another session holds it), flips .env to hosted, stops
:8000, flips back, runs restart.sh, retries the dispatch probe, releases the lock. Every real run does the same around training.

Before any run: `uv run python -m train.loader --data <dir> --model $R3_ARTIFACTS/models/Qwen3-4B-8bit --n 20`
prints 20 decoded loss masks and the token budget; the launcher runs the same checks and refuses over-length data.

| File | Role |
| --- | --- |
| `configs/voice_run1.yaml`, `smoke.yaml`, `routing.yaml` | mlx_lm.lora 0.32.2 recipes (locked decisions); run-sized keys added by the launcher |
| `loader.py` | manifest + sha256, structural checks, loss masks through mlx_lm's own ChatDataset, 20-mask printout |
| `launch.py` | preflight -> gpu_flip hosted + stop :8000 -> train -> fuse/quantize -> gpu_flip local + restart -> ledger + run.json |
| `lora_driver.py` | mlx_lm.lora `train_model` with JSONL metrics (every 10 steps), clip 1.0, `--device cpu` |
| `fuse/fuse_quantize.py` | `mlx_lm.fuse` on the 8-bit base (re-quantizes), base tokenizer files restored, adapter kept in `adapter/`, `r3.json` |
| `eval_hook.py`, `eval_stub.py` | eval-on-checkpoint; per-family table from `eval_results` or `eval_results.jsonl`; precision gate |
| `routing_format.py` | tickets -> prompt -> label examples, stratified split, manifest |
| `synth.py`, `prompt.py` | 20-example synthetic dry-run set; contract system prompt loader |
| `ledger.py`, `runinfo.py`, `env.py` | cost_ledger + run.json, git/config hashes and model ids, `.env` |

Dry run of the whole chain (0.6B on the GPU, ~20 s): `R3_DRY_RUN=1 uv run pytest tests/train/test_dry_run.py -q -s`.
Fast checks: `uv run pytest tests/train -q`.

Artifacts: `$R3_ARTIFACTS/runs/<id>/{config.resolved.yaml, data_report.json, metrics.jsonl, train.log, fuse.log, run.json, ledger.jsonl}`
and `$R3_ARTIFACTS/models/<prefix>-<hash>/` (served directly; `adapter/` for the gate; `r3.json` provenance).
