"""Dataset manifest reader + loss-mask verifier for mlx_lm.lora (0.32.2) chat data.

A dataset directory holds `manifest.json` plus `train.jsonl` / `valid.jsonl` in the mlx_lm
`messages` + `tools` format (LORA.md "tools"): each line is one training example whose LAST
message is the target assistant turn (text, tool calls, or both); everything before it is the
masked prefix. mlx_lm with `mask_prompt: true` puts the loss on `tokens[offset:]` where
`offset = len(apply_chat_template(messages[:-1], tools, add_generation_prompt=True))`.

With the Qwen3 template the target turn renders as
    <|im_start|>assistant\n<think>\n\n</think>\n\n{content}{<tool_call>...</tool_call>}<|im_end|>\n
so the loss region starts with the empty think block (thinking disabled) and the serving prompt
(`enable_thinking=False`) is an exact token prefix of the training sequence. Tool results render
inside `<tool_response>` in a user turn and always sit in the prefix. This module proves those
facts on real tokens and prints 20 decoded masks for the human check:

    uv run python -m train.loader --data $R3_ARTIFACTS/datasets/<name> --model $R3_ARTIFACTS/models/Qwen3-4B-8bit

manifest.json keys the loader uses: `files` ({split: filename}; default train/valid.jsonl),
`sha256` ({filename: hex}, verified when present). Everything else is provenance and is copied
into the run record by train/launch.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

EMPTY_THINK = "<think>\n\n</think>\n\n"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
ROLES = {"system", "user", "assistant", "tool"}
DEFAULT_FILES = {"train": "train.jsonl", "valid": "valid.jsonl"}


@dataclass
class Manifest:
    dir: Path
    path: Path
    data: dict
    files: dict[str, Path]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(dataset_dir: str | Path, verify: bool = True) -> Manifest:
    d = Path(dataset_dir)
    path = d / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json in {d}")
    data = json.loads(path.read_text())
    listed = data.get("files") or DEFAULT_FILES
    files: dict[str, Path] = {}
    for split, name in listed.items():
        p = d / name
        if not p.exists():
            if "files" in data:
                raise FileNotFoundError(f"manifest lists {name} for split {split!r} but it is missing")
            continue
        files[split] = p
    if "train" not in files:
        raise FileNotFoundError(f"no train split in {d}")
    if verify:
        for name, digest in (data.get("sha256") or {}).items():
            actual = sha256_file(d / name)
            if actual != digest:
                raise ValueError(f"sha256 mismatch for {name}: manifest {digest[:12]}…, file {actual[:12]}…")
    return Manifest(dir=d, path=path, data=data, files=files)


def manifest_hash(manifest: Manifest) -> str:
    """Identity of the dataset for the ledger: sha256 of manifest.json itself."""
    return sha256_file(manifest.path)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_example(ex: dict, require_tools: bool = True) -> list[str]:
    """Structural problems that would break the Qwen3 template or put loss in the wrong place."""
    errors: list[str] = []
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["messages must be a non-empty list"]
    tools = ex.get("tools")
    if require_tools and (not isinstance(tools, list) or not tools):
        errors.append("tools must be a non-empty list of OpenAI function schemas")
    if tools is not None and not isinstance(tools, list):
        errors.append("tools must be a list")
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role not in ROLES:
            errors.append(f"message {i}: unknown role {role!r}")
            continue
        if not isinstance(m.get("content"), str):
            errors.append(f"message {i} ({role}): content must be a string (the template cannot render null)")
        if role == "assistant":
            for j, call in enumerate(m.get("tool_calls") or []):
                fn = call.get("function") or {}
                if not isinstance(fn.get("name"), str):
                    errors.append(f"message {i}: tool_calls[{j}] has no function.name")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        errors.append(f"message {i}: tool_calls[{j}].function.arguments is not valid JSON")
                elif not isinstance(args, dict):
                    errors.append(f"message {i}: tool_calls[{j}].function.arguments must be a JSON string or object")
        elif m.get("tool_calls"):
            errors.append(f"message {i} ({role}): only assistant messages may carry tool_calls")
    last = msgs[-1]
    if last.get("role") != "assistant":
        errors.append(f"last message must be the target assistant turn, got role={last.get('role')!r}")
    elif not last.get("content") and not last.get("tool_calls"):
        errors.append("target assistant turn is empty (no content, no tool_calls)")
    return errors


@dataclass
class MaskedExample:
    tokens: list[int]
    offset: int
    prefix_text: str
    target_text: str
    boundary: list[tuple[int, int, str]]  # (position, token id, decoded text) around the first loss token
    checks: dict[str, bool]
    tool_names: list[str] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def n_target(self) -> int:
        return len(self.tokens) - self.offset

    @property
    def errors(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]


def _template(tokenizer, messages, tools, **kw) -> list[int]:
    return list(tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False, **kw))


def render_and_mask(tokenizer, ex: dict, max_seq_length: int) -> MaskedExample:
    """Tokens + loss offset exactly as mlx_lm.tuner.datasets.ChatDataset produces them, plus checks."""
    from mlx_lm.tuner.datasets import ChatDataset

    problems = validate_example(ex, require_tools=False)
    if problems:
        raise ValueError("; ".join(problems))
    msgs, tools = ex["messages"], ex.get("tools")
    tokens, offset = ChatDataset([ex], tokenizer, mask_prompt=True).process(ex)
    tokens = list(tokens)
    prefix_ids = _template(tokenizer, msgs[:-1], tools, add_generation_prompt=True)
    served_ids = _template(tokenizer, msgs[:-1], tools, add_generation_prompt=True, enable_thinking=False)
    empty_think_ids = _template(tokenizer, msgs[:-1], tools, add_generation_prompt=True, enable_thinking=False)[len(prefix_ids):]
    prefix_text = tokenizer.decode(tokens[:offset])
    target_text = tokenizer.decode(tokens[offset:])
    target = msgs[-1]
    tool_names = [c["function"]["name"] for c in (target.get("tool_calls") or [])]
    checks = {
        "prefix_aligned": offset == len(prefix_ids) and tokens[:offset] == prefix_ids,
        "assistant_header_before_loss": prefix_text.endswith(ASSISTANT_HEADER),
        "thinking_disabled": target_text.startswith(EMPTY_THINK),
        "serve_prompt_is_prefix": tokens[: len(served_ids)] == served_ids,
        "tool_calls_in_loss": all(f'<tool_call>\n{{"name": "{n}"' in target_text for n in tool_names),
        "no_tool_result_in_loss": "<tool_response>" not in target_text,
        "ends_with_im_end": target_text.endswith("<|im_end|>\n"),
        "target_nonempty": len(tokens) - offset > len(empty_think_ids),
        "fits_seq": len(tokens) <= max_seq_length,
    }
    lo, hi = max(0, offset - 3), min(len(tokens), offset + 4)
    boundary = [(i, tokens[i], tokenizer.decode([tokens[i]])) for i in range(lo, hi)]
    return MaskedExample(tokens, offset, prefix_text, target_text, boundary, checks, tool_names)


@dataclass
class DatasetReport:
    n: int
    masks: list[MaskedExample]
    failures: list[tuple[int, list[str]]]
    stats: dict

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def ok(self) -> bool:
        return not self.failures


def scaffold_tokens(tokenizer, ex: dict) -> int:
    """Tokens the system prompt + tools block + template scaffolding cost before any dialogue."""
    msgs = ex.get("messages") or []
    system = [m for m in msgs[:1] if m.get("role") == "system"]
    probe = system + [{"role": "user", "content": "x"}]
    return len(_template(tokenizer, probe, ex.get("tools"), add_generation_prompt=True)) - 1


def check_dataset(tokenizer, examples: list[dict], max_seq_length: int, require_tools: bool = True) -> DatasetReport:
    masks: list[MaskedExample] = []
    failures: list[tuple[int, list[str]]] = []
    for i, ex in enumerate(examples):
        errs = validate_example(ex, require_tools=require_tools)
        if errs:
            failures.append((i, errs))
            continue
        me = render_and_mask(tokenizer, ex, max_seq_length)
        masks.append(me)
        if me.errors:
            failures.append((i, me.errors))
    lens = [m.n_tokens for m in masks] or [0]
    tgt = [m.n_target for m in masks] or [0]
    scaffold = scaffold_tokens(tokenizer, examples[0]) if examples else 0
    stats = {
        "scaffold_tokens": scaffold,
        "conversation_budget": max_seq_length - scaffold,
        "n": len(examples),
        "n_rendered": len(masks),
        "mean_tokens": statistics.fmean(lens),
        "max_tokens": max(lens),
        "mean_target_tokens": statistics.fmean(tgt),
        "max_target_tokens": max(tgt),
        "total_tokens": sum(lens),
        "total_target_tokens": sum(tgt),
        "n_with_tool_calls": sum(1 for m in masks if m.tool_names),
        "n_over_seq": sum(1 for m in masks if not m.checks["fits_seq"]),
        "max_seq_length": max_seq_length,
    }
    return DatasetReport(len(examples), masks, failures, stats)


def format_mask(me: MaskedExample, index: int, prefix_chars: int = 320) -> str:
    calls = f" · tool calls: {', '.join(me.tool_names)}" if me.tool_names else ""
    marks = " ".join(("| " if pos == me.offset else "") + f"[{tid}:{text!r}]" for pos, tid, text in me.boundary)
    checks = "  ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in me.checks.items())
    head = me.prefix_text[-prefix_chars:]
    return "\n".join(
        [
            f"=== example {index}: {me.n_tokens} tokens, loss on {me.n_target} (offset {me.offset}){calls}",
            f"--- masked prefix (last {min(prefix_chars, len(me.prefix_text))} chars of {len(me.prefix_text)}) ---",
            ("…" if len(me.prefix_text) > prefix_chars else "") + head,
            "--- boundary tokens [id:text], | marks the first LOSS token ---",
            marks,
            "--- LOSS: the target assistant turn (every token below is trained on) ---",
            me.target_text,
            f"--- checks: {checks}",
            "",
        ]
    )


def load_tokenizer(model_path: str | Path):
    """The same tokenizer object mlx_lm.lora uses (TokenizerWrapper over the HF tokenizer)."""
    from mlx_lm.utils import load_tokenizer as _load

    return _load(Path(model_path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset directory with manifest.json")
    ap.add_argument("--model", required=True, help="base model directory (tokenizer + chat template)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--n", type=int, default=20, help="masks to print")
    ap.add_argument("--json", type=Path, default=None, help="write the report here")
    ap.add_argument("--no-require-tools", action="store_true", help="routing data has no tools")
    a = ap.parse_args(argv)

    manifest = load_manifest(a.data)
    if a.split not in manifest.files:
        print(f"split {a.split!r} not in manifest files {sorted(manifest.files)}", file=sys.stderr)
        return 2
    examples = read_jsonl(manifest.files[a.split])
    tokenizer = load_tokenizer(a.model)
    report = check_dataset(tokenizer, examples, a.max_seq_length, require_tools=not a.no_require_tools)

    for i, me in enumerate(report.masks[: a.n]):
        print(format_mask(me, i))
    s = report.stats
    print(f"dataset {manifest.dir} split={a.split} manifest sha256 {manifest_hash(manifest)}")
    print(
        f"prompt scaffold (system prompt + tools block + template) = {s['scaffold_tokens']} tokens -> "
        f"{s['conversation_budget']} tokens left for the dialogue at seq {s['max_seq_length']}"
    )
    print(
        f"{s['n']} examples, {s['n_rendered']} rendered; tokens mean {s['mean_tokens']:.0f} max {s['max_tokens']} "
        f"(limit {s['max_seq_length']}, {s['n_over_seq']} over); target tokens mean {s['mean_target_tokens']:.0f} "
        f"max {s['max_target_tokens']}; {s['n_with_tool_calls']} targets contain tool calls; "
        f"{s['total_tokens']} tokens total, {s['total_target_tokens']} in the loss"
    )
    for idx, errs in report.failures:
        print(f"FAIL example {idx}: {'; '.join(errs)}")
    if a.json:
        a.json.write_text(json.dumps({
            "data": str(manifest.dir), "split": a.split, "manifest_hash": manifest_hash(manifest),
            "manifest": manifest.data, "stats": s,
            "failures": [{"index": i, "errors": e} for i, e in report.failures],
            "checks": [m.checks for m in report.masks],
        }, indent=2))
    if report.ok:
        print(f"ALL CHECKS PASSED on {s['n']} examples")
        return 0
    print(f"CHECKS FAILED on {report.n_failed} of {s['n']} examples")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
