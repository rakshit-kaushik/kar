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


def test_schemas_load_the_seven_contract_tools_in_openai_format():
    schemas = load_tool_schemas()
    assert sorted(schemas) == ["book_appointment", "cancel_appointment", "check_availability", "create_ticket", "lookup_account", "request_handoff", "reschedule_appointment"]
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
    ex.retry_infra_errors = False  # the retry has its own tests below; here the single failed attempt is the point
    await ex.execute("check_availability", {"branch_id": "mill_rd", "service_type": "notary", "date": "2026-11-08"})
    assert [e["event"] for e in events] == ["tool_start", "tool_end"]
    assert events[0]["call_id"] == "room-1" and events[0]["payload"]["name"] == "check_availability"
    assert events[1]["payload"] == {"name": "check_availability", "ok": False, "error": "unavailable", "ms": events[1]["payload"]["ms"]}
    assert isinstance(events[1]["payload"]["ms"], int)


async def test_handoff_marks_state_and_errors_count_toward_tool_failures():
    ex, state, _, sb = make()
    ex.retry_infra_errors = False  # one failed attempt must count as a tool failure and complete nothing
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
    assert len(tools) == 7
    assert all(tool_context.is_raw_function_tool(t) for t in tools)
    names = sorted(tool_context.get_raw_function_info(t).raw_schema["name"] for t in tools)
    assert names[0] == "book_appointment" and names[-1] == "reschedule_appointment"


# --- one retry on infrastructure errors (Nov 1 09:00: the final model never retried a timed-out booking when the caller asked it to;
# a transient timeout on an idempotent write is the client's to absorb, identical for every model, logged as tool_retry)
BOOK = {"customer_id": "C-1001", "branch_id": "market_st", "service_type": "mortgage_consultation", "slot_start": "2026-11-03T09:30"}


async def test_a_tool_that_times_out_is_retried_once_with_the_same_arguments_and_both_attempts_are_logged():
    ex, state, events, sb = make()
    sb.inject([{"tool": "book_appointment", "call_index": 1, "error": "timeout"}])
    res = json.loads(await ex.execute("book_appointment", dict(BOOK)))
    assert res.get("status") == "booked" and "error" not in res
    assert [e["event"] for e in events] == ["tool_start", "tool_end", "tool_retry", "tool_start", "tool_end"]
    assert events[1]["payload"]["ok"] is False and events[1]["payload"]["error"] == "timeout" and events[1]["payload"]["attempt"] == 1
    assert events[2]["payload"] == {"name": "book_appointment", "error": "timeout"}
    assert events[4]["payload"]["ok"] is True and events[4]["payload"]["attempt"] == 2
    keys = [c["args"].get("idempotency_key") for c in sb.log if c["tool"] == "book_appointment"]
    assert len(keys) == 2 and keys[0] == keys[1]  # the retry reuses the key, so a late success of the first attempt cannot double-book
    assert [c["tool"] for c in state.completed] == ["book_appointment"] and state.tool_failures == 1


async def test_a_second_infrastructure_error_goes_to_the_model_and_business_errors_are_not_retried():
    ex, state, events, sb = make()
    sb.inject([{"tool": "book_appointment", "call_index": 1, "error": "timeout"}, {"tool": "book_appointment", "call_index": 2, "error": "unavailable"}])
    res = json.loads(await ex.execute("book_appointment", dict(BOOK)))
    assert res["error"] == "unavailable" and state.tool_failures == 2 and state.completed == []
    assert [e["event"] for e in events] == ["tool_start", "tool_end", "tool_retry", "tool_start", "tool_end"]
    ex2, _, events2, sb2 = make()
    sb2.inject([{"tool": "book_appointment", "call_index": 1, "error": "conflict"}])
    assert json.loads(await ex2.execute("book_appointment", dict(BOOK)))["error"] == "conflict"
    assert [e["event"] for e in events2] == ["tool_start", "tool_end"]


async def test_the_retry_can_be_switched_off():
    ex, _, events, sb = make()
    ex.retry_infra_errors = False
    sb.inject([{"tool": "lookup_account", "call_index": 1, "error": "timeout"}])
    assert json.loads(await ex.execute("lookup_account", {"phone": "5550142219", "date_of_birth": "1988-03-12"}))["error"] == "timeout"
    assert [e["event"] for e in events] == ["tool_start", "tool_end"]
