"""The exit criterion for T+1..T+6: one command on the 20-example synthetic set runs
train -> fuse -> 8-bit quantize -> versioned model dir -> run.json + ledger rows.

Default: Qwen3-0.6B-8bit on the GPU (the human cleared GPU dry runs; seconds beside the idle server).
    R3_DRY_RUN=1 uv run pytest tests/train/test_dry_run.py -q -s
Variants: R3_DRY_RUN_DEVICE=cpu (slow: MLX's CPU quantized matmul is scalar), R3_DRY_RUN_MODEL=tiny
(random-init 2-layer Qwen3 with the real tokenizer, the only CPU option that finishes in minutes) or
R3_DRY_RUN_MODEL=4b (the real base; 4 micro-steps at seq 2048).
"""
import json
import os
from pathlib import Path

import pytest
import yaml
from mlx_lm.lora import yaml_loader

from train.launch import main

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not os.environ.get("R3_DRY_RUN"), reason="set R3_DRY_RUN=1 to run the end-to-end dry run")


@pytest.fixture(scope="session")
def dry_model(request, base_4b, tmp_path_factory):
    which = os.environ.get("R3_DRY_RUN_MODEL", "0.6b").lower()
    if which == "tiny":
        from tiny_model import make_tiny_qwen3_8bit

        return make_tiny_qwen3_8bit(tmp_path_factory.mktemp("tiny") / "tiny-qwen3-8bit", base_4b)
    if which == "4b":
        return base_4b
    return request.getfixturevalue("base_06b")


def test_one_command_trains_fuses_quantizes_and_versions(synth_dir, dry_model, tmp_path):
    device = os.environ.get("R3_DRY_RUN_DEVICE", "gpu")
    recipe = yaml.load((REPO / "train" / "configs" / "smoke.yaml").read_text(), yaml_loader)
    recipe.update(model=str(dry_model), iters=4, grad_accumulation_steps=2, steps_per_eval=2, save_every=2, val_batches=1)
    cfg_path = tmp_path / "smoke_dry.yaml"
    cfg_path.write_text(yaml.safe_dump(recipe))
    art = tmp_path / "art"

    rc = main(["--config", str(cfg_path), "--data", str(synth_dir), "--artifacts", str(art), "--run-id", "dry",
               "--device", device, "--no-flip", "--no-server", "--no-db", "--no-lock"])
    assert rc == 0

    run = json.loads((art / "runs" / "dry" / "run.json").read_text())
    assert run["status"] == "done", run.get("error")
    m = run["metrics"]
    assert m["iters"] == 4 and m["total_tokens_per_second"] > 0 and m["final_val_loss"] is not None
    assert [v["iteration"] for v in m["val"]] == [0, 1, 3]  # mlx_lm evaluates at it=1, every steps_per_eval, and at the end
    assert run["git"]["hash"] and run["data"]["manifest_hash"] and run["config_hash"]
    assert run["sizing"]["fixed_iters"] is True and run["resolved_config"]["lr_schedule"]["name"] == "cosine_decay"

    model_dir = Path(run["model_dir"])
    assert model_dir.parent == art / "models" and model_dir.name.startswith("r3-voice-")
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["quantization"]["bits"] == 8 and cfg["quantization"]["group_size"] == 64
    assert list(model_dir.glob("model*.safetensors")) and (model_dir / "tokenizer_config.json").exists()
    assert (model_dir / "tokenizer.json").read_bytes() == (Path(dry_model) / "tokenizer.json").read_bytes()
    assert (model_dir / "adapter" / "adapters.safetensors").exists()
    assert (model_dir / "adapter" / "0000002_adapters.safetensors").exists()  # midpoint checkpoint
    adapter_cfg = json.loads((model_dir / "adapter" / "adapter_config.json").read_text())
    assert adapter_cfg["lora_parameters"]["scale"] == 2.0 and adapter_cfg["num_layers"] == -1
    meta = json.loads((model_dir / "r3.json").read_text())
    assert meta["model_id"] == model_dir.name and meta["fused"]["bits"] == 8 and meta["adapter_sha256"]
    assert meta["commands"][0].startswith("mlx_lm.fuse ") and len(meta["commands"]) == 1  # 8-bit base: fuse re-quantizes

    rows = [json.loads(l) for l in (art / "runs" / "dry" / "ledger.jsonl").read_text().splitlines()]
    items = {r["item"] for r in rows}
    assert {"train_run", "train_wall_time", "train_tokens_per_sec_total", "train_loss_final", "val_loss_final",
            "fuse_quantize_wall_time"} <= items
    assert all(r["model_id"] == model_dir.name and r["run_id"] == "dry" for r in rows)
    note = json.loads(next(r for r in rows if r["item"] == "train_run")["note"])
    assert note["git"] == run["git"]["hash"] and note["manifest_hash"] == run["data"]["manifest_hash"]

    # the fused 8-bit artifact loads with mlx_lm and runs a forward pass
    import mlx.core as mx
    from mlx_lm.utils import load

    if device == "cpu":
        mx.set_default_device(mx.cpu)
    model, tok = load(str(model_dir))
    ids = tok.apply_chat_template([{"role": "user", "content": "hello"}], add_generation_prompt=True, enable_thinking=False)
    logits = model(mx.array([ids]))
    mx.eval(logits)
    assert logits.shape[-1] == cfg["vocab_size"]
