"""Live-demo fault injection: a room name can carry a sandbox fault (CONTRACT exceptions: tool timeout), e.g.
`demo-timeout--fault-check_availability-timeout` or `...--fault-book_appointment-unavailable-2` (2nd call fails)."""
from services.agent.faults import faults_from_room_name


def test_room_name_without_marker_injects_nothing():
    assert faults_from_room_name("r3-g1-0116-012024-3") == []


def test_room_name_with_tool_and_error_injects_first_call_by_default():
    assert faults_from_room_name("demo-timeout--fault-check_availability-timeout") == [{"tool": "check_availability", "call_index": 1, "error": "timeout"}]


def test_room_name_with_call_index_and_two_faults():
    name = "x--fault-book_appointment-unavailable-2--fault-lookup_account-timeout"
    assert faults_from_room_name(name) == [
        {"tool": "book_appointment", "call_index": 2, "error": "unavailable"},
        {"tool": "lookup_account", "call_index": 1, "error": "timeout"},
    ]


def test_unknown_tool_or_error_is_ignored():
    assert faults_from_room_name("x--fault-launch_rockets-timeout") == []
    assert faults_from_room_name("x--fault-lookup_account-explode") == []
