"""restart.sh override block (Nov 1 08:02 incident): a MODEL_DIR given on the command line must survive sourcing a .env that
also defines MODEL_DIR; agent switches are overridable only when named in R3_OVERRIDES."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def override_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# values given on the command line win over .env"))
    end = next(i for i, l in enumerate(lines) if l.startswith("ART=${R3_ARTIFACTS"))
    return "\n".join(lines[start:end])


def run_block(text: str, tmp_path, env: dict) -> list[str]:
    (tmp_path / ".env").write_text("MODEL_DIR=/models/run1\nLLM_MODEL=r3-voice-run1\nAGENT_TOOL_RETRY=1\nAGENT_POLICY_HANDOFF=1\n")
    script = "set -euo pipefail\n" + override_block(text) + '\necho "$MODEL_DIR $LLM_MODEL $AGENT_TOOL_RETRY $AGENT_POLICY_HANDOFF"\n'
    out = subprocess.run(["bash", "-c", script], cwd=tmp_path, env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_deploy_override_of_model_dir_survives_dotenv(tmp_path):
    got = run_block((ROOT / "scripts/restart.sh").read_text(), tmp_path, {"MODEL_DIR": "/models/run2", "LLM_MODEL": "r3-voice-run2"})
    assert got[:2] == ["/models/run2", "r3-voice-run2"]


def test_agent_switches_need_the_opt_in(tmp_path):
    text = (ROOT / "scripts/restart.sh").read_text()
    assert run_block(text, tmp_path, {"AGENT_TOOL_RETRY": "0"})[2] == "1"  # a stale shell value does not win
    got = run_block(text, tmp_path, {"R3_OVERRIDES": "AGENT_TOOL_RETRY AGENT_POLICY_HANDOFF", "AGENT_TOOL_RETRY": "0", "AGENT_POLICY_HANDOFF": "0"})
    assert got[2:] == ["0", "0"]
