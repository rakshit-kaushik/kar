"""Run mlx_lm.lora (0.32.2) training with structured metrics, gradient clipping and a device switch.

Why not the `mlx_lm.lora` CLI directly:
  1. `mlx_lm.lora.run()` discards the training_callback it is given (it rebuilds one from
     --report-to), so tokens/s and loss would have to be scraped from rich console output;
  2. the recipe says clip 1.0 and 0.32.2 has no gradient-clipping option;
  3. dry runs must stay off the GPU (`--device cpu`).
The driver repeats the dozen lines of `run()` (seed, load model, load datasets) and then calls
mlx_lm's own `train_model`, so the LoRA conversion, adapter_config.json, trainer loop, prompt
masking, schedule and checkpoints are all mlx_lm's. Metrics go to a JSONL file, one record per
report (every `steps_per_report` steps) and per validation.

    uv run python -m train.lora_driver --config runs/<id>/config.resolved.yaml --metrics runs/<id>/metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx_lm.lora import CONFIG_DEFAULTS, yaml_loader
from mlx_lm.tuner.callbacks import TrainingCallback

_OPTIMIZERS = ("Adam", "AdamW", "Muon", "SGD", "Adafactor")


def load_config(path: str | Path) -> dict:
    """A resolved recipe with mlx_lm's CONFIG_DEFAULTS filled in, floats parsed like mlx_lm.lora does."""
    cfg = yaml.load(Path(path).read_text(), yaml_loader) or {}
    for k, v in CONFIG_DEFAULTS.items():
        if cfg.get(k) is None:
            cfg[k] = v
    return cfg


def clip_gradients(grads, max_norm: float):
    """Scale the whole gradient tree so its global L2 norm is at most max_norm; returns (grads, norm)."""
    return optim.clip_grad_norm(grads, max_norm)


def make_clipped(cls, max_norm: float):
    class Clipped(cls):
        def apply_gradients(self, gradients, parameters):
            gradients, _ = clip_gradients(gradients, self.max_norm)
            return super().apply_gradients(gradients, parameters)

    Clipped.max_norm = max_norm
    Clipped.__name__ = f"Clipped{cls.__name__}"
    Clipped.__qualname__ = Clipped.__name__
    return Clipped


def install_clipping(max_norm: float) -> None:
    """Make mlx_lm.lora.train_model build optimizers that clip the accumulated, averaged gradient to
    max_norm before every update (mlx_lm 0.32.2 has no flag for this). Process-local: it swaps the
    `optim` name that train_model resolves `optim.Adam` etc. through."""
    import mlx_lm.lora as lora

    lora.optim = types.SimpleNamespace(**{n: make_clipped(getattr(optim, n), max_norm) for n in _OPTIMIZERS})


class JsonlCallback(TrainingCallback):
    """Append every train/val report from mlx_lm's trainer to a JSONL file and echo one line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, kind: str, info: dict) -> dict:
        rec = {"kind": kind, "ts": time.time()}
        for k, v in info.items():
            rec[k] = float(v) if hasattr(v, "item") else v
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def on_train_loss_report(self, train_info: dict):
        r = self._write("train", train_info)
        print(
            f"[metrics] iter {r['iteration']} train_loss {r['train_loss']:.4f} lr {r['learning_rate']:.2e} "
            f"loss-tokens/s {r['tokens_per_second']:.0f} it/s {r['iterations_per_second']:.3f} "
            f"trained_tokens {r['trained_tokens']} peak_mem {r['peak_memory']:.1f}GB",
            flush=True,
        )

    def on_val_loss_report(self, val_info: dict):
        r = self._write("val", val_info)
        print(f"[metrics] iter {r['iteration']} val_loss {r['val_loss']:.4f} ({r['val_time']:.1f}s)", flush=True)


def run_training(cfg: dict, metrics_path: str | Path, device: str = "gpu") -> dict:
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    r3 = cfg.get("r3") or {}
    clip = r3.get("clip_grad_norm")
    if clip:
        install_clipping(float(clip))
    args = types.SimpleNamespace(**cfg)
    np.random.seed(args.seed)
    print(f"[driver] device {mx.default_device()} · clip_grad_norm {clip} · loading {args.model}", flush=True)
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    print(
        f"[driver] train {len(train_set)} · valid {len(valid_set)} · iters {args.iters} · batch {args.batch_size} "
        f"· grad_accum {args.grad_accumulation_steps} · seq {args.max_seq_length} · lr_schedule {args.lr_schedule}",
        flush=True,
    )
    t0 = time.perf_counter()
    train_model(args, model, train_set, valid_set, JsonlCallback(metrics_path))
    summary = {
        "wall_seconds": time.perf_counter() - t0,
        "device": str(mx.default_device()),
        "n_train": len(train_set),
        "n_valid": len(valid_set),
        "iters": args.iters,
        "clip_grad_norm": clip,
        "adapter_path": args.adapter_path,
    }
    Path(metrics_path).with_name("train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[driver] done in {summary['wall_seconds']:.0f}s; adapter at {args.adapter_path}", flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="resolved recipe (train/launch.py writes it)")
    ap.add_argument("--metrics", type=Path, default=None, help="JSONL metrics path (default: next to the config)")
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    run_training(cfg, a.metrics or a.config.with_name("metrics.jsonl"), a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
