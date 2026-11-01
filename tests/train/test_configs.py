"""Exit check for task 1: the three mlx_lm.lora recipes carry the locked decisions.

Every top-level key except `r3:` must be a real mlx_lm.lora config key (0.32.2), so a
misspelled key cannot silently fall back to a default.
"""
from pathlib import Path

import yaml
from mlx_lm.lora import CONFIG_DEFAULTS, yaml_loader

CONFIGS = Path(__file__).resolve().parents[2] / "train" / "configs"
LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def load(name: str) -> dict:
    return yaml.load((CONFIGS / name).read_text(), yaml_loader)


def test_voice_run1_carries_the_locked_recipe():
    c = load("voice_run1.yaml")
    assert "Qwen3-4B-8bit" in c["model"]
    assert c["train"] is True
    assert c["fine_tune_type"] == "lora"
    assert c["num_layers"] == -1  # every decoder block
    # mlx_lm uses `scale` (= alpha / rank), never `alpha`
    assert c["lora_parameters"] == {"rank": 32, "scale": 2.0, "dropout": 0.05, "keys": LORA_KEYS}
    assert c["max_seq_length"] == 2048
    assert c["batch_size"] == 1
    assert c["grad_accumulation_steps"] == 16
    assert isinstance(c["learning_rate"], float) and c["learning_rate"] == 1e-4
    assert c["grad_checkpoint"] is True
    assert c["mask_prompt"] is True
    assert c["steps_per_report"] == 10
    assert c["val_batches"] == -1
    assert "packing" not in c and "iters" not in c  # one epoch, sized by the launcher
    r3 = c["r3"]
    assert r3["task"] == "voice" and r3["model_prefix"] == "r3-voice"
    assert r3["epochs"] == 1
    assert r3["schedule"] == "cosine_decay" and r3["warmup_ratio"] == 0.03
    assert r3["clip_grad_norm"] == 1.0
    assert r3["eval_points"] == [0.5, 1.0]
    assert r3["enable_thinking"] is False
    assert r3["quantize"] == {"bits": 8, "group_size": 64}
    assert r3["window_minutes"] == 120


def test_every_top_level_key_is_a_real_mlx_lm_key():
    for name in ("voice_run1.yaml", "smoke.yaml", "routing.yaml"):
        c = load(name)
        unknown = {k for k in c if k != "r3"} - set(CONFIG_DEFAULTS)
        assert not unknown, f"{name}: not mlx_lm.lora keys: {unknown}"


def _diff(a: dict, b: dict) -> set:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_smoke_is_voice_run1_with_50_steps():
    voice, smoke = load("voice_run1.yaml"), load("smoke.yaml")
    assert _diff(voice, smoke) == {"iters", "steps_per_eval", "save_every", "r3"}
    assert smoke["iters"] == 50
    assert smoke["steps_per_eval"] == 25 and smoke["save_every"] == 25  # midpoint + end
    assert _diff(voice["r3"], smoke["r3"]) == {"window_minutes"}
    assert smoke["r3"]["window_minutes"] == 10


def test_routing_is_voice_run1_at_seq_1024():
    voice, routing = load("voice_run1.yaml"), load("routing.yaml")
    assert _diff(voice, routing) == {"max_seq_length", "r3"}
    assert routing["max_seq_length"] == 1024
    assert _diff(voice["r3"], routing["r3"]) == {"task", "model_prefix", "window_minutes"}
    assert routing["r3"]["task"] == "routing"
    assert routing["r3"]["model_prefix"] == "r3-routing"
    assert routing["r3"]["window_minutes"] == 20
