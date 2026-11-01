"""Every lane-S script must import cleanly: a broken import makes an unattended block (measure_block, rehearsal, deploy) fail in
seconds and lose its GPU window (Nov 1 05:11: sim_caller died on an import and the whole slot was spent). Importing only; main() is
never run (every script guards it with __name__)."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ["sim_caller", "g1_report", "demo_check", "cost_report", "writeup", "replay", "dry_run_agent", "probe_dispatch", "load_text", "deploy_model", "bench_base", "rooms", "join_token", "smoke_tools", "relabel_model"]


@pytest.mark.parametrize("name", [s for s in SCRIPTS if (ROOT / "scripts" / f"{s}.py").exists()])
def test_script_imports(name):
    spec = importlib.util.spec_from_file_location(f"r3_script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module is not None
