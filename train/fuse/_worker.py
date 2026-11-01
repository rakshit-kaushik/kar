"""Subprocess entry that runs `mlx_lm.fuse` / `mlx_lm.convert` with an explicit device.

`uv run python -m train.fuse._worker --device cpu fuse --model B --adapter-path A --save-path F`
is `mlx_lm.fuse --model B --adapter-path A --save-path F` executed on the CPU. mlx has no
environment switch for the default device, so dry runs need this one-file wrapper.
"""
from __future__ import annotations

import importlib
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    device = "gpu"
    if argv[:1] == ["--device"]:
        device = argv[1]
        argv = argv[2:]
    tool, rest = argv[0], argv[1:]
    if tool not in ("fuse", "convert"):
        raise SystemExit(f"unknown tool {tool!r}; expected fuse or convert")
    if device == "cpu":
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
    sys.argv = [f"mlx_lm.{tool}", *rest]
    importlib.import_module(f"mlx_lm.{tool}").main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
