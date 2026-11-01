"""The write-up's latency and targets tables must show the final model's rehearsal batch, not only the base model's G1 batch."""
import importlib.util
import pathlib
import re
import sys

spec = importlib.util.spec_from_file_location("writeup", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "writeup.py")
writeup = importlib.util.module_from_spec(spec)
sys.modules["writeup"] = writeup
spec.loader.exec_module(writeup)

LEDGER = [  # (run_id, item, qty) in id order
    ("g1-sim-base", "g1.eos_to_first_audio_heard.p50", 3060), ("g1-sim-base", "g1.llm_ttft_local.p50", 966), ("g1-sim-base", "g1.calls", 5),
    ("g1-sim-base", "g1.eos_to_first_audio_heard.n", 21),
    ("rehearsal-r3-voice-x-1101-0840-ordinary", "g1.eos_to_first_audio_heard.p50", 1500), ("rehearsal-r3-voice-x-1101-0840-ordinary", "g1.calls", 5),
    ("rehearsal-r3-voice-x-1101-0840-ordinary", "g1.eos_to_first_audio_heard.n", 30),
    ("rehearsal-r3-voice-x-1101-0840-all", "g1.eos_to_first_audio_heard.p50", 1572), ("rehearsal-r3-voice-x-1101-0840-all", "g1.llm_ttft_local.p50", 146),
    ("rehearsal-r3-voice-x-1101-0840-all", "g1.calls", 30), ("rehearsal-r3-voice-x-1101-0840-all", "g1.eos_to_first_audio_heard.n", 181),
]


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeCon:
    """Answers the two ledger queries the tables use: `run_id like %s` and the load-test selection; anything else is empty."""

    def execute(self, sql, params=()):
        if "run_id like %s" in sql and params:
            rx = re.compile("^" + re.escape(params[0]).replace("%", ".*").replace("_", ".") + "$")
            return FakeCursor([r for r in LEDGER if rx.match(r[0])])
        return FakeCursor([])


def test_latency_table_includes_the_combined_rehearsal_batch_but_not_its_per_scenario_rows():
    md = writeup.latency_table(FakeCon())
    assert "rehearsal-r3-voice-x-1101-0840-all" in md
    assert "rehearsal-r3-voice-x-1101-0840-ordinary" not in md
    assert "g1-sim-base" in md


def test_targets_table_has_a_column_for_the_latest_rehearsal_batch_beside_the_base_model():
    md = writeup.targets_table(FakeCon())
    header = md.splitlines()[0]
    assert "g1-sim-base" in header and "rehearsal-r3-voice-x-1101-0840-all" in header
    row = next(line for line in md.splitlines() if line.startswith("| End of speech → first audio heard (caller clock) p50"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2].startswith("3060") and cells[3].startswith("1572")


PASS_LEDGER = [
    ("cost-pass-rehearsal-x-0840", "r3-voice-x", "cost.total_per_call_usd", 0.0434, "30 calls"), ("cost-pass-rehearsal-x-0840", "r3-voice-x", "cost.per_pass_usd", 0.0814, "16 of 30 passed the scenario check (correct handoffs included)"),
    ("cost-pass-rehearsal-x-0840", "r3-voice-x", "cost.per_success_usd", 0.0930, "14 completed"), ("cost-pass-rehearsal-x-0840", "r3-voice-x", "cost.pass_calls", 16, "of 30"),
    ("cost-rehearsal-x-0840", "r3-voice-x", "cost.total_per_call_usd", 0.0434, "30 calls"),  # no pass row: not a headline row
]


class FakePassCon(FakeCon):
    def execute(self, sql, params=()):
        if "cost.per_pass_usd" in sql:
            return FakeCursor([r for r in PASS_LEDGER if r[0] in {x[0] for x in PASS_LEDGER if x[2] == "cost.per_pass_usd"}])
        return super().execute(sql, params)


def test_pass_cost_table_shows_cost_per_passing_call_with_both_denominators_stated():
    md = writeup.pass_cost_table(FakePassCon())
    header = md.splitlines()[0]
    assert "$/passing call" in header and "$/completed booking" in header
    row = next(line for line in md.splitlines() if "cost-pass-rehearsal-x-0840" in line)
    assert "0.0814" in row and "0.0930" in row and "16" in row
    assert "cost-rehearsal-x-0840 |" not in md
