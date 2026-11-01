"""Scored run directory -> training dataset (mlx_lm `messages` + `tools` JSONL) with provenance.

Filter, do not correct: an episode failing any non-advisory scorer is dropped whole. Within a kept
episode up to three targets are taken (data.format.targets); a target over the brevity rule or
over 2,048 tokens is dropped alone. Splits are by scenario (hash of family + scenario id), never
by example, so a conversation's targets stay in one split.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from data.format.targets import KINDS, select_targets, to_example
from eval.sandbox import TOOL_SCHEMAS
from eval.scorers import ScorerResult, all_pass
from eval.simulator.episode import Episode

ROOT = Path(__file__).resolve().parents[2]
STT_RULE_VERSION = (
    "stt_shape-v1 (lowercase, numbers as words, fillers/homophones/drops by level 0-3)"
)


def split_of(scenario_id: str, family: str, valid_frac: float, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}:{family}:{scenario_id}".encode()).hexdigest()[:8], 16)
    return "valid" if (h % 10_000) / 10_000 < valid_frac else "train"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 — provenance only
        return "unknown"


def _load_run(run_dir: Path) -> list[tuple[Episode, list[ScorerResult]]]:
    out = []
    for p in sorted((run_dir / "episodes").glob("*.json")):
        d = json.loads(p.read_text())
        ep = Episode.from_dict(d)
        scores = [
            ScorerResult(
                s["scorer"],
                s["passed"],
                s.get("detail", {}),
                s.get("turn_idx"),
                s.get("advisory", False),
            )
            for s in d.get("scores", [])
        ]
        out.append((ep, scores))
    return out


def build_dataset(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    name: str,
    fits: Callable[[dict], bool] | None,
    valid_frac: float = 0.1,
    seed: int = 0,
    max_targets: int = 3,
) -> dict:
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = _load_run(run_dir)
    by_kind = {
        k: {"generated": 0, "kept": 0, "brevity": 0, "length": 0} for k in KINDS if k != "other"
    }
    verdicts: dict[str, dict] = {}
    rows: dict[str, list[dict]] = {"train": [], "valid": []}
    tpc: Counter[int] = Counter()
    n_failed = 0
    teacher = results[0][0].model_id if results else None
    system_text = None
    for ep, scores in results:
        key = f"{ep.scenario_id}-s{ep.repeat_seed}"
        failed = [s.scorer for s in scores if not s.passed and not s.advisory]
        verdicts[key] = {"all_pass": all_pass(scores), "failed": failed, "outcome": ep.outcome}
        if not all_pass(scores):
            n_failed += 1
            continue
        if system_text is None and ep.messages and ep.messages[0].get("role") == "system":
            system_text = ep.messages[0]["content"]
        chosen, dropped = select_targets(ep, fits=fits, max_targets=max_targets)
        for t in chosen:
            by_kind[t.kind]["generated"] += 1
            by_kind[t.kind]["kept"] += 1
            rows[split_of(ep.scenario_id, ep.family, valid_frac, seed)].append(to_example(ep, t))
        for t in dropped:
            by_kind[t.kind]["generated"] += 1
            by_kind[t.kind][t.reason] += 1
        tpc[len(chosen)] += 1
    for split, items in rows.items():
        with (out / f"{split}.jsonl").open("w") as fh:
            for ex in items:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    kept_eps = sum(tpc.values())
    generated_t = sum(v["generated"] for v in by_kind.values())
    kept_t = sum(v["kept"] for v in by_kind.values())
    dropped = {
        "brevity": sum(v["brevity"] for v in by_kind.values()),
        "length": sum(v["length"] for v in by_kind.values()),
    }
    tpc_stats = {
        "mean": round(sum(k * v for k, v in tpc.items()) / kept_eps, 2) if kept_eps else 0.0,
        "histogram": {str(k): v for k, v in sorted(tpc.items())},
    }
    stats = {
        "episodes": {"generated": len(results), "kept": kept_eps, "scorer_failed": n_failed},
        "targets": {
            "generated": generated_t,
            "kept": kept_t,
            "dropped": dropped,
            "by_kind": by_kind,
        },
        "targets_per_conversation": tpc_stats,
    }
    manifest = {
        "name": name,
        "files": {"train": "train.jsonl", "valid": "valid.jsonl"},
        "sha256": {
            f"{s}.jsonl": hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in rows
        },
        "n_examples": {s: len(items) for s, items in rows.items()},
        "max_seq_length": 2048,
        "targets_per_conversation": tpc_stats,
        "drops": {
            "scorer_failed_episodes": n_failed,
            "brevity_targets": dropped["brevity"],
            "over_2048_targets": dropped["length"],
        },
        "yield": stats["targets"],
        "provenance": {
            "generator_git_sha": _git_sha(),
            "system_prompt_sha256": hashlib.sha256((system_text or "").encode()).hexdigest(),
            "tools_sha256": hashlib.sha256(
                json.dumps(list(TOOL_SCHEMAS.values()), sort_keys=True).encode()
            ).hexdigest(),
            "tool_names": list(TOOL_SCHEMAS),
            "stt_rule_version": STT_RULE_VERSION,
            "scenario_ids": sorted({ep.scenario_id for ep, _ in results}),
            "scorer_verdicts": verdicts,
            "run_dir": str(run_dir),
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "teacher_model_id": teacher,
            "episodes": stats["episodes"],
            "split": {
                "by": "scenario (sha256 of seed:family:id)",
                "valid_frac": valid_frac,
                "seed": seed,
            },
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (out / "yield.md").write_text(yield_report(name, stats, manifest))
    return stats


def yield_report(name: str, stats: dict, manifest: dict) -> str:
    e, t = stats["episodes"], stats["targets"]
    lines = [
        f"# Yield — dataset `{name}`",
        "",
        (
            f"Episodes: {e['generated']} generated, {e['kept']} kept, {e['scorer_failed']} dropped by the scorers "
            f"({round(100 * e['kept'] / e['generated']) if e['generated'] else 0}% kept)."
        ),
        (
            f"Targets: {t['generated']} generated, {t['kept']} kept ({round(100 * t['kept'] / t['generated']) if t['generated'] else 0}%); "
            f"dropped for brevity {t['dropped']['brevity']}, for length {t['dropped']['length']}."
        ),
        (
            f"Targets per kept conversation: mean {stats['targets_per_conversation']['mean']}, "
            f"histogram {stats['targets_per_conversation']['histogram']}."
        ),
        "",
        "| kind | generated | kept | dropped: brevity | dropped: length | kept % |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in t["by_kind"].items():
        pct = round(100 * v["kept"] / v["generated"]) if v["generated"] else 0
        lines.append(
            f"| {k} | {v['generated']} | {v['kept']} | {v['brevity']} | {v['length']} | {pct}% |"
        )
    lines += [
        "",
        f"Examples: train {manifest['n_examples']['train']}, valid {manifest['n_examples']['valid']}.",
    ]
    return "\n".join(lines) + "\n"
