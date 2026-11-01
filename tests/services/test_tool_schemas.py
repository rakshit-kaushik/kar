"""Model-facing tool schemas (schemas/tools/*.json). Idempotency keys are a tool-layer concern:
the executor generates them, the model never sees or produces them (approved Nov 1)."""
from services.agent.state import WRITE_TOOLS
from services.tools.executor import load_tool_schemas


def test_model_facing_schemas_do_not_expose_idempotency_key():
    for name, schema in load_tool_schemas().items():
        params = schema["function"]["parameters"]
        assert "idempotency_key" not in params["properties"], name
        assert "idempotency_key" not in params.get("required", []), name


def test_write_tools_still_have_their_business_arguments_required():
    schemas = load_tool_schemas()
    assert schemas["book_appointment"]["function"]["parameters"]["required"] == ["customer_id", "branch_id", "service_type", "slot_start"]
    assert schemas["block_card"]["function"]["parameters"]["required"] == ["customer_id", "card_last4", "reason"]
    assert set(WRITE_TOOLS) <= set(schemas)
