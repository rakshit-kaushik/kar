"""data.format.spotcheck: readable transcripts of kept episodes for the human's 50-episode check."""

import pytest
from helpers import VERIFY, VERIFY_LINES, ScriptedAgent, ScriptedCaller, say, scenario, tc, tools

from data.format.spotcheck import render_episode, write_spotcheck
from eval.runner import run_scenarios

AVAIL = {"branch_id": "market_st", "service_type": "mortgage_consultation", "date": "2026-11-03"}
BOOK = {
    "customer_id": "C-1001",
    "branch_id": "market_st",
    "service_type": "mortgage_consultation",
    "slot_start": "2026-11-03T09:30",
    "idempotency_key": "k",
}
GOOD = VERIFY + [
    tools(tc("check_availability", AVAIL)),
    say("Nine thirty is open. Book it?"),
    tools(tc("book_appointment", BOOK)),
    say("Booked. Anything else?"),
    say("Goodbye."),
]


@pytest.mark.asyncio
async def test_spotcheck_renders_transcripts_with_tools_and_verdicts(tmp_path):
    run = tmp_path / "run"
    scen = [
        scenario(id="gen_ordinary_aa_00"),
        scenario(id="gen_ordinary_aa_01", caller={"persona": "Terse"}),
    ]
    await run_scenarios(
        scen,
        lambda: ScriptedAgent(GOOD),
        lambda s, k: ScriptedCaller(VERIFY_LINES + ["book it", "yes", ("bye", True)]),
        out_dir=run,
        run_id="g",
        model_id="teacher",
        parallel=2,
        repeats=1,
    )
    path = write_spotcheck([run], tmp_path / "spotcheck.md", n=50, seed=0)
    text = path.read_text()
    assert text.count("## ") == 2 and "gen_ordinary_aa_00" in text and "gen_ordinary_aa_01" in text
    assert "C: " in text and "A: " in text and "-> book_appointment(" in text and "<- {" in text
    assert "verdict: all scorers pass" in text and "goal:" in text and "persona:" in text
    one = render_episode(
        __import__("json").loads((run / "episodes" / "gen_ordinary_aa_00-s0.json").read_text())
    )
    assert one.startswith("## gen_ordinary_aa_00-s0")
