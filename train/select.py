"""Pick the candidate: the plan's rule, applied to family tables from train/eval_hook.py.

Selection is by the eval runner on the test set, never by loss: zero unsupported success claims first,
then task completion (final_state pass rate), then tool-argument accuracy (tool_calls). Base and every
run's fused model are candidates; the one that wins is the final model, the others stay on disk.

    uv run python -m train.select runs/<a>/eval/family_table.json runs/<b>/eval/family_table.json --out selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_candidates(tables: list[dict]) -> list[dict]:
    def key(t: dict):
        s = t.get("selection") or {}
        return (s.get("unsupported_success_claims") or 0, -(s.get("task_completion") or 0.0), -(s.get("tool_calls") or 0.0))

    seen = set()
    unique = []
    for t in sorted(tables, key=key):  # the same model's table can sit in two run dirs; keep its best-ranked copy
        if t.get("model") in seen:
            continue
        seen.add(t.get("model"))
        unique.append(t)
    ranked = []
    for rank, t in enumerate(unique, start=1):
        s = t.get("selection") or {}
        claims = s.get("unsupported_success_claims") or 0
        reason = (f"{claims} unsupported success claim{'s' if claims != 1 else ''}; " if claims else "no unsupported success claims; ") + \
                 f"task completion {s.get('task_completion')}; tool calls {s.get('tool_calls')}"
        ranked.append({"rank": rank, "model": t.get("model"), "model_dir": t.get("model_dir"), "run_id": t.get("run_id"),
                       "task_completion": s.get("task_completion"), "tool_calls": s.get("tool_calls"),
                       "unsupported_success_claims": claims, "episodes": s.get("episodes"), "reason": reason})
    return ranked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+", type=Path, help="family_table.json files (one per candidate)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    ranked = select_candidates([json.loads(p.read_text()) for p in a.tables])
    lines = ["| rank | model | task completion | tool calls | unsupported claims | episodes |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['rank']} | {r['model']} | {r['task_completion']} | {r['tool_calls']} | {r['unsupported_success_claims']} | {r['episodes']} |" for r in ranked]
    text = "\n".join(lines)
    print(text)
    print(f"selected: {ranked[0]['model']} ({ranked[0]['reason']})")
    if a.out:
        a.out.write_text(json.dumps({"selected": ranked[0], "ranked": ranked}, indent=2) + "\n")
        a.out.with_suffix(".md").write_text(text + f"\n\nselected: **{ranked[0]['model']}** ({ranked[0]['reason']})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
