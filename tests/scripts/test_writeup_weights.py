"""After the Nov 1 08:02 incident (run 1's weights served under run 2's name until 10:52) the write-up must say what was measured
on which weights, show the proof rows, and never present a mislabelled batch as the final model's."""
import importlib.util
import pathlib
import re
import sys

spec = importlib.util.spec_from_file_location("writeup_w", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "writeup.py")
writeup = importlib.util.module_from_spec(spec)
sys.modules["writeup_w"] = writeup
spec.loader.exec_module(writeup)

LEDGER = [  # (run_id, item, qty) in id order; the mislabelled batch is written last on purpose
    ("rehearsal-r3-voice-730f8ba68e-1101-1110-rules-all", "g1.eos_to_first_audio_heard.p50", 1500),
    ("rehearsal-r3-voice-206506525c-mislabelled-1101-0840-all", "g1.eos_to_first_audio_heard.p50", 1572),
]
PROOF = [  # (run_id, model_id, item, qty, note) in id order
    ("incident-1101-0802-served-run1", "r3-voice-730f8ba68e", "proof.serving_path", 0, "path=/m/r3-voice-206506525c; expected=/m/r3-voice-730f8ba68e; cmd=..."),
    ("incident-1101-0802-served-run1", "r3-voice-730f8ba68e", "proof.v1_models", 0, "ids=r3-voice-730f8ba68e; directory name=r3-voice-206506525c"),
    ("incident-1101-0802-served-run1", "r3-voice-730f8ba68e", "proof.fingerprint_reasks_weekday", 1, "re-asks the weekday (the run-1 signature): Which day would you like the consultation?"),
    ("deploy-r3-voice-730f8ba68e-1101-1052", "r3-voice-730f8ba68e", "proof.serving_path", 1, "path=/m/r3-voice-730f8ba68e; expected=/m/r3-voice-730f8ba68e; cmd=..."),
    ("deploy-r3-voice-730f8ba68e-1101-1052", "r3-voice-730f8ba68e", "proof.v1_models", 1, "ids=r3-voice-730f8ba68e; directory name=r3-voice-730f8ba68e"),
    ("deploy-r3-voice-730f8ba68e-1101-1052", "r3-voice-730f8ba68e", "proof.fingerprint_reasks_weekday", 0, "does not re-ask the weekday: {\"name\": \"check_availability\"}"),
]


class Cur:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeCon:
    def execute(self, sql, params=()):
        if "item like 'proof.%%'" in sql or "item like 'proof.%'" in sql:
            return Cur(PROOF)
        if "run_id like %s" in sql and params:
            rx = re.compile("^" + re.escape(params[0]).replace("%", ".*").replace("_", ".") + "$")
            return Cur([r for r in LEDGER if rx.match(r[0])])
        return Cur([])


def test_weights_section_shows_each_proof_and_the_window():
    text = writeup.weights_section(FakeCon())
    incident = next(l for l in text.splitlines() if l.startswith("| incident-1101-0802-served-run1"))
    deploy = next(l for l in text.splitlines() if l.startswith("| deploy-r3-voice-730f8ba68e-1101-1052"))
    assert "r3-voice-206506525c" in incident and incident.count("MISMATCH") == 2 and "re-asks" in incident
    assert "MISMATCH" not in deploy and "r3-voice-730f8ba68e" in deploy
    assert "206506525c-mislabelled" in text and "08:02" in text and "10:52" in text


def test_targets_table_never_shows_a_mislabelled_batch_as_the_final_model():
    header = writeup.targets_table(FakeCon()).splitlines()[0]
    assert "rehearsal-r3-voice-730f8ba68e-1101-1110-rules-all" in header and "mislabelled" not in header
