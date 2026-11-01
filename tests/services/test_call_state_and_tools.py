"""Per-call state (CONTRACT.md §5) and the tool executor that sits between the LLM's tool calls and
the sandbox: fills idempotency keys, emits tool_start/tool_end events, keeps verified facts apart
from claimed ones, never raises into the model."""
import json
import pathlib
import uuid

from services.agent.state import CallState
from services.tools.executor import ToolExecutor, load_tool_schemas
from services.tools.sandbox import Sandbox

SEED = pathlib.Path(__file__).resolve().parents[2] / "eval/sandbox/seed.json"


def make():
    events = []

    async def sink(ev):
        events.append(ev)

    sb = Sandbox.from_seed(SEED)
    state = CallState(call_id="room-1", model_id="r3-voice-base")
    return ToolExecutor(sandbox=sb, state=state, event_sink=sink), state, events, sb


def test_schemas_load_eight_tools_in_openai_format():
    schemas = load_tool_schemas()
    assert sorted(schemas) == ["block_card", "book_appointment", "cancel_appointment", "check_availability", "create_ticket", "lookup_account", "request_handoff", "reschedule_appointment"]
    assert schemas["lookup_account"]["type"] == "function"
    assert schemas["lookup_account"]["function"]["parameters"]["required"] == ["phone", "date_of_birth"]


async def test_lookup_moves_verified_facts_into_state_and_returns_json_string():
    ex, state, _, _ = make()
    out = await ex.execute("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    assert isinstance(out, str) and json.loads(out)["verified"] is True
    assert state.verified["customer_id"] == "C-1001" and state.verified["first_name"] == "Dana"
    assert state.stage == "intent"
    assert state.claimed["phone"] == "5550142219" and state.claimed["date_of_birth"] == "1988-03-12"


async def test_failed_lookup_keeps_claimed_facts_but_nothing_verified():
    ex, state, _, _ = make()
    out = json.loads(await ex.execute("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-01-01"}))
    assert out["error"] == "not_verified"
    assert state.verified["customer_id"] is None and state.claimed["phone"] == "5550142219"
    assert state.verification_failures == 1


async def test_missing_idempotency_key_is_generated_and_write_is_recorded_as_completed():
    ex, state, _, sb = make()
    await ex.execute("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"})
    out = json.loads(await ex.execute("book_appointment", {"customer_id": "C-1001", "branch_id": "market_st", "service_type": "notary", "slot_start": "2026-11-03T10:00"}))
    assert out["appointment_id"].startswith("A-3")
    assert state.completed[0]["tool"] == "book_appointment" and state.completed[0]["result"]["appointment_id"] == out["appointment_id"]
    key = sb.log[-1]["args"]["idempotency_key"]
    assert uuid.UUID(key)  # a real UUID was injected
    assert state.stage == "closing"


async def test_bad_idempotency_key_from_model_is_replaced():
    ex, _, _, sb = make()
    await ex.execute("cancel_appointment", {"appointment_id": "A-2001", "idempotency_key": "abc"})
    assert uuid.UUID(sb.log[-1]["args"]["idempotency_key"])


async def test_tool_events_carry_timing_and_ok_flag():
    ex, _, events, _ = make()
    await ex.execute("check_availability", {"branch_id": "mill_rd", "service_type": "notary", "date": "2026-11-08"})
    assert [e["event"] for e in events] == ["tool_start", "tool_end"]
    assert events[0]["call_id"] == "room-1" and events[0]["payload"]["name"] == "check_availability"
    assert events[1]["payload"] == {"name": "check_availability", "ok": False, "error": "unavailable", "ms": events[1]["payload"]["ms"]}
    assert isinstance(events[1]["payload"]["ms"], int)


async def test_handoff_marks_state_and_errors_count_toward_tool_failures():
    ex, state, _, sb = make()
    sb.inject([{"tool": "cancel_appointment", "call_index": 1, "error": "timeout"}])
    await ex.execute("cancel_appointment", {"appointment_id": "A-2001"})
    assert state.tool_failures == 1 and state.completed == []
    await ex.execute("request_handoff", {"reason": "tool_failure", "summary": "cancel kept timing out"})
    assert state.handoff == {"requested": True, "reason": "tool_failure", "summary": "cancel kept timing out"}
    assert state.stage == "handoff"


async def test_unknown_tool_or_bad_json_returns_error_json_not_exception():
    ex, _, _, _ = make()
    assert json.loads(await ex.execute("teleport", {}))["error"] == "invalid_argument"
    assert json.loads(await ex.execute("lookup_account", "{not json"))["error"] == "invalid_argument"


def test_state_serializes_per_contract_schema():
    state = CallState(call_id="room-1", model_id="m")
    d = state.to_dict()
    assert set(d) == {"verified", "claimed", "stage", "pending", "completed", "handoff", "model_id", "call_id", "started_at"}
    assert d["stage"] == "greeting" and d["handoff"] is None and d["verified"]["appointments"] == []


def test_build_livekit_tools_wraps_every_schema_as_a_raw_function_tool():
    from livekit.agents.llm import tool_context

    from services.tools.livekit_tools import build_livekit_tools

    tools = build_livekit_tools(lambda ctx: None)
    assert len(tools) == 8
    assert all(tool_context.is_raw_function_tool(t) for t in tools)
    names = sorted(tool_context.get_raw_function_info(t).raw_schema["name"] for t in tools)
    assert names[0] == "block_card" and names[-1] == "reschedule_appointment"
