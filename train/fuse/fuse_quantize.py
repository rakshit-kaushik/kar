"""adapter + 8-bit base -> fused, re-quantized 8-bit model directory, unfused adapter kept beside it.

mlx_lm.fuse (0.32.2) flags: --model, --adapter-path, --save-path, --dequantize, --export-gguf,
--gguf-path, --upload-repo, --trust-remote-code. With a quantized base and no --dequantize,
`LoRALinear.fuse()` dequantizes each layer, adds scale * (B @ A), and re-quantizes with the base's
group_size / bits / mode, so the saved model is already 8-bit and config.json keeps the
quantization block. If the base is not quantized at the requested bits, the fused model goes
through `mlx_lm.convert --hf-path <fused> --mlx-path <out> -q --q-bits 8 --q-group-size 64`.

Output layout (served directly: `vllm-mlx serve <out> --served-model-name <id>`):
    <out>/config.json, model*.safetensors, tokenizer*     fused 8-bit model
    <out>/adapter/                                        unfused adapter + checkpoints (precision gate:
                                                          mlx_lm.server --model <base> --adapter-path <out>/adapter)
    <out>/r3.json                                         provenance: base, adapter sha256, bits, timings
After the fuse the base's tokenizer files are copied over the re-saved ones (transformers 5.22 rewrites
them) and tokenization parity with the base is verified, so the served model tokenizes like the base.

    uv run python -m train.fuse.fuse_quantize --base $R3_ARTIFACTS/models/Qwen3-4B-8bit \
        --adapter $R3_ARTIFACTS/runs/<id>/adapter --out $R3_ARTIFACTS/models/r3-voice-<hash>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from train.env import REPO_ROOT


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "generation_config.json")
PARITY_PROBES = (
    "yeah it's five five five zero one four two two one nine",
    "Tuesday, November fourth at nine thirty in the morning. Shall I book it?",
    '<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "book_appointment", "arguments": '
    '{"slot_start":"2026-11-04T09:30","idempotency_key":"af19922a-d9b8-4714-a61a-441c12e0c8b2"}}\n</tool_call><|im_end|>\n',
    "  indented\n\n\n text 1234567890 ééé 日本語 <tool_response>\n{\"slots\":[]}\n</tool_response>",
)


def sync_tokenizer_files(base: str | Path, out: str | Path) -> dict:
    """Copy the base's tokenizer files over the ones mlx_lm's save() re-wrote through transformers
    (5.22 rewrites tokenizer.json's pre-tokenizer regex, drops vocab/merges/added_tokens and emits an
    'incorrect regex pattern' warning on every load). The served artifact must tokenize byte-for-byte
    like the base, so the base's files win; files the base does not have are removed."""
    base, out = Path(base), Path(out)
    copied, removed = [], []
    for name in TOKENIZER_FILES:
        src, dst = base / name, out / name
        if src.exists():
            shutil.copyfile(src, dst)
            copied.append(name)
        elif dst.exists():
            dst.unlink()
            removed.append(name)
    return {"copied": copied, "removed": removed}


def verify_tokenizer_parity(base: str | Path, out: str | Path, probes=PARITY_PROBES) -> bool:
    from mlx_lm.utils import load_tokenizer

    a, b = load_tokenizer(Path(base)), load_tokenizer(Path(out))
    for probe in probes:
        if a.encode(probe) != b.encode(probe):
            raise ValueError(f"{out}: tokenization differs from {base} on {probe!r}")
    msgs = [{"role": "user", "content": probes[0]}]
    if a.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False) != b.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False):
        raise ValueError(f"{out}: chat template renders differently from {base}")
    return True


def layout(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    return {"fused": out, "adapter": out / "adapter", "meta": out / "r3.json"}


def _quantization(config: dict) -> dict | None:
    return config.get("quantization") or config.get("quantization_config")


def verify_model_dir(path: str | Path, bits: int = 8) -> dict:
    """A loadable mlx model directory quantized at `bits`: config + weights + tokenizer."""
    p = Path(path)
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{p}: no config.json")
    cfg = json.loads(cfg_path.read_text())
    q = _quantization(cfg)
    if not q or int(q.get("bits", 0)) != bits:
        raise ValueError(f"{p}: expected a {bits}-bit model, config.json quantization = {q}")
    weights = sorted(f.name for f in p.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"{p}: no model*.safetensors")
    if not (p / "tokenizer_config.json").exists():
        raise FileNotFoundError(f"{p}: no tokenizer_config.json")
    return {"bits": int(q["bits"]), "group_size": q.get("group_size"), "mode": q.get("mode", "affine"),
            "weight_files": weights, "model_type": cfg.get("model_type")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(tool: str, args: list[str], device: str, log: Path | None) -> str:
    cmd = [sys.executable, "-m", "train.fuse._worker", "--device", device, tool, *args]
    shown = f"mlx_lm.{tool} " + " ".join(args)
    print(f"[fuse] {shown}" + (" (cpu)" if device == "cpu" else ""), flush=True)
    with (log.open("a") if log else open("/dev/null", "w")) as out:
        out.write(f"$ {shown}\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"{shown} failed with exit {proc.returncode}" + (f"; see {log}" if log else ""))
    return shown


def fuse_and_quantize(base: str | Path, adapter: str | Path, out_dir: str | Path, bits: int = 8,
                      group_size: int = 64, device: str = "gpu", log_path: str | Path | None = None,
                      extra_meta: dict | None = None) -> dict:
    base, adapter, out = Path(base), Path(adapter), Path(out_dir)
    log = Path(log_path) if log_path else None
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; a model id is never overwritten")
    for required in ("adapters.safetensors", "adapter_config.json"):
        if not (adapter / required).exists():
            raise FileNotFoundError(f"{adapter}: missing {required}")
    base_cfg = json.loads((base / "config.json").read_text())
    base_q = _quantization(base_cfg) or {}
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    commands = []
    if int(base_q.get("bits", 0)) == bits:
        # fuse re-quantizes per layer with the base's own bits/group_size -> already the served precision
        commands.append(_run("fuse", ["--model", str(base), "--adapter-path", str(adapter), "--save-path", str(out)], device, log))
    else:
        tmp = out.parent / (out.name + ".fused-tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        commands.append(_run("fuse", ["--model", str(base), "--adapter-path", str(adapter), "--save-path", str(tmp), "--dequantize"], device, log))
        commands.append(_run("convert", ["--hf-path", str(tmp), "--mlx-path", str(out), "-q", "--q-bits", str(bits), "--q-group-size", str(group_size)], device, log))
        shutil.rmtree(tmp)
    fuse_seconds = time.perf_counter() - t0
    tokenizer_sync = sync_tokenizer_files(base, out)
    verify_tokenizer_parity(base, out)
    info = verify_model_dir(out, bits=bits)
    if group_size and info["group_size"] not in (None, group_size):
        raise ValueError(f"{out}: group_size {info['group_size']} != {group_size}")
    lay = layout(out)
    shutil.copytree(adapter, lay["adapter"])
    import mlx_lm

    meta = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base_model": str(base),
        "base_quantization": base_q,
        "adapter_sha256": _sha256(adapter / "adapters.safetensors"),
        "adapter_config": json.loads((adapter / "adapter_config.json").read_text()),
        "fused": info,
        "fuse_seconds": round(fuse_seconds, 2),
        "commands": commands,
        "tokenizer_files": tokenizer_sync,
        "mlx_lm_version": mlx_lm.__version__,
        "serve": {"fused": str(out), "adapter_on_base": {"model": str(base), "adapter_path": str(lay["adapter"])}},
        **(extra_meta or {}),
    }
    lay["meta"].write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"[fuse] {out} verified: {info['bits']}-bit, group {info['group_size']}, {len(info['weight_files'])} weight file(s), "
          f"{fuse_seconds:.0f}s; unfused adapter kept at {lay['adapter']}", flush=True)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args(argv)
    fuse_and_quantize(a.base, a.adapter, a.out, a.bits, a.group_size, a.device, a.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
