"""The LiveKit tool wrappers call the optional on_tool_start hook (the agent's filler + filler_start event) before the executor."""
from services.tools.livekit_tools import build_livekit_tools


class FakeExecutor:
    def __init__(self, log):
        self.log = log

    async def execute(self, name, args):
        self.log.append(("execute", name, args))
        return '{"ok": true}'


async def test_on_tool_start_runs_before_the_executor():
    log, ctx = [], object()
    tools = build_livekit_tools(lambda c: FakeExecutor(log), on_tool_start=lambda c, name: log.append(("start", name, c is ctx)))
    lookup = next(t for t in tools if t.__name__ == "lookup_account")
    assert await lookup({"phone": "5550142219"}, ctx) == '{"ok": true}'
    assert log == [("start", "lookup_account", True), ("execute", "lookup_account", {"phone": "5550142219"})]


async def test_without_a_hook_the_wrapper_only_executes():
    log = []
    tools = build_livekit_tools(lambda c: FakeExecutor(log))
    book = next(t for t in tools if t.__name__ == "book_appointment")
    await book({}, object())
    assert log == [("execute", "book_appointment", {})]
