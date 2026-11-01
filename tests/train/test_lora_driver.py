"""The driver runs mlx_lm.lora's own train_model with a metrics callback, clip 1.0 and a device switch."""
import json

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm.lora import CONFIG_DEFAULTS

from train.lora_driver import JsonlCallback, clip_gradients, install_clipping, load_config


def test_load_config_fills_mlx_defaults_and_keeps_floats(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: x\nlearning_rate: 1.0e-4\nr3:\n  clip_grad_norm: 1.0\n")
    cfg = load_config(p)
    assert cfg["learning_rate"] == 1e-4 and isinstance(cfg["learning_rate"], float)
    assert cfg["iters"] == CONFIG_DEFAULTS["iters"] and cfg["mask_prompt"] is False
    assert cfg["r3"] == {"clip_grad_norm": 1.0}


def test_clip_gradients_scales_the_total_norm_down_to_max_norm():
    grads = {"a": mx.ones((4,)) * 3.0, "b": {"c": mx.ones((4,)) * 4.0}}  # total norm sqrt(4*9 + 4*16) = 10
    clipped, norm = clip_gradients(grads, 1.0)
    assert abs(norm.item() - 10.0) < 1e-5
    assert abs(clipped["a"][0].item() - 0.3) < 1e-6 and abs(clipped["b"]["c"][0].item() - 0.4) < 1e-6
    small, _ = clip_gradients({"a": mx.ones((4,)) * 0.1}, 1.0)
    assert abs(small["a"][0].item() - 0.1) < 1e-6  # under the limit: unchanged


def test_install_clipping_wraps_the_optimizer_classes_train_model_uses():
    import mlx_lm.lora as lora

    install_clipping(1.0)
    for name in ("Adam", "AdamW", "SGD", "Adafactor", "Muon"):
        cls = getattr(lora.optim, name)
        assert issubclass(cls, getattr(optim, name)) and cls.max_norm == 1.0
    # the clipped optimizer applies a clipped step: grad norm sqrt(4*144 + 4*256) = 40, lr 1
    opt = lora.optim.SGD(learning_rate=1.0)
    params = {"w": mx.zeros((4,)), "v": mx.zeros((4,))}
    grads = {"w": mx.ones((4,)) * 12.0, "v": mx.ones((4,)) * 16.0}
    new = opt.apply_gradients(grads, params)
    assert abs(new["w"][0].item() + 0.3) < 1e-5 and abs(new["v"][0].item() + 0.4) < 1e-5


def test_jsonl_callback_appends_train_and_val_records(tmp_path, capsys):
    p = tmp_path / "metrics.jsonl"
    cb = JsonlCallback(p)
    cb.on_val_loss_report({"iteration": 0, "val_loss": 2.5, "val_time": 1.0})
    cb.on_train_loss_report({"iteration": 10, "train_loss": 2.0, "learning_rate": 1e-4,
                             "iterations_per_second": 0.5, "tokens_per_second": 900.0,
                             "trained_tokens": 1000, "peak_memory": 12.3})
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["val", "train"]
    assert rows[1]["tokens_per_second"] == 900.0 and "ts" in rows[1]
    out = capsys.readouterr().out
    assert "iter 10" in out and "900" in out
