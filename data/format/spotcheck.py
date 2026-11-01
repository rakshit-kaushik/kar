"""Readable transcripts of kept episodes for the human's spot-check (plan: 50 episodes read end to end).

uv run python -m data.format.spotcheck --runs <run-dir>... --out <dataset>/spotcheck.md [--n 50]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def render_episode(d: dict) -> str:
    sc = d.get("scenario", {})
    fails = [s["scorer"] for s in d.get("scores", []) if not s["passed"] and not s.get("advisory")]
    adv = [s["scorer"] for s in d.get("scores", []) if not s["passed"] and s.get("advisory")]
    verdict = "all scorers pass" if not fails else "FAILS " + ", ".join(fails)
    if adv:
        verdict += f" (advisory: {', '.join(adv)})"
    lines = [
        f"## {d['episode_id']}",
        f"family: {d['family']} · outcome: {d['outcome']} · verdict: {verdict}",
        f"persona: {sc.get('caller', {}).get('persona', '')}",
        f"goal: {sc.get('goal', '')}",
        "",
    ]
    for m in d["messages"][1:]:
        role = m.get("role")
        if role == "user":
            lines.append(f"C: {m.get('content', '')}")
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                lines.append(f"  -> {fn.get('name')}({fn.get('arguments')})")
            if m.get("content"):
                lines.append(f"A: {m['content']}")
        elif role == "tool":
            lines.append(f"  <- {str(m.get('content', ''))[:200]}")
    return "\n".join(lines) + "\n"


def write_spotcheck(
    run_dirs: list[str | Path], out: str | Path, n: int = 50, seed: int = 0, kept_only: bool = True
) -> Path:
    eps = []
    for d in run_dirs:
        for p in sorted((Path(d) / "episodes").glob("*.json")):
            data = json.loads(p.read_text())
            ok = all(s["passed"] or s.get("advisory") for s in data.get("scores", []))
            if kept_only and not ok:
                continue
            eps.append(data)
    random.Random(seed).shuffle(eps)
    chosen = sorted(eps[:n], key=lambda e: (e["family"], e["episode_id"]))
    text = f"# Spot-check: {len(chosen)} of {len(eps)} kept episodes (seed {seed})\n\n" + "\n".join(
        render_episode(e) for e in chosen
    )
    out = Path(out)
    out.write_text(text)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    path = write_spotcheck(a.runs, a.out, n=a.n, seed=a.seed)
    print(f"spot-check -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
