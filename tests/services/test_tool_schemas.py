"""Contract artifacts the agent loads: model-facing tool schemas (schemas/tools/*.json) and the shared
system prompt (schemas/system_prompt.md). Idempotency keys are a tool-layer concern: the executor
generates them, the model never sees or produces them (approved Nov 1). block_card was cut Nov 1."""
import pathlib
import re

from services.agent.state import WRITE_TOOLS
from services.tools.executor import load_tool_schemas
from services.tools.sandbox import Sandbox

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEVEN = ["book_appointment", "cancel_appointment", "check_availability", "create_ticket", "lookup_account", "request_handoff", "reschedule_appointment"]


def test_model_facing_schemas_do_not_expose_idempotency_key():
    for name, schema in load_tool_schemas().items():
        params = schema["function"]["parameters"]
        assert "idempotency_key" not in params["properties"], name
        assert "idempotency_key" not in params.get("required", []), name


def test_write_tools_still_have_their_business_arguments_required():
    schemas = load_tool_schemas()
    assert schemas["book_appointment"]["function"]["parameters"]["required"] == ["customer_id", "branch_id", "service_type", "slot_start"]
    assert schemas["create_ticket"]["function"]["parameters"]["required"] == ["customer_id", "category", "summary"]
    assert set(WRITE_TOOLS) <= set(schemas)


def test_block_card_is_gone_from_schemas_sandbox_and_state():
    assert sorted(load_tool_schemas()) == SEVEN
    assert "block_card" not in WRITE_TOOLS
    assert "block_card" not in Sandbox.TOOLS
    assert "block_card" not in (ROOT / "eval/sandbox/seed.json").read_text()


def test_shared_system_prompt_is_the_only_prompt_file():
    assert not (ROOT / "services/agent/system_prompt.md").exists(), "the agent must load schemas/system_prompt.md so the live prompt cannot drift from the training data"
    text = (ROOT / "schemas/system_prompt.md").read_text()
    assert re.findall(r"\{[a-z_]+\}", text) == ["{today}"]
    for path in ("services/agent/main.py", "scripts/dry_run_agent.py", "scripts/bench_base.py", "scripts/smoke_tools.py"):
        src = (ROOT / path).read_text()
        assert "schemas/system_prompt.md" in src and "services/agent/system_prompt.md" not in src, path
