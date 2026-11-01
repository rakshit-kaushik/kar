"""The eight contract tools as LiveKit raw function tools. Schemas come from schemas/tools/*.json
(the same files the simulator and eval runner load); execution goes through the per-call
ToolExecutor that `get_executor(context)` returns."""
from __future__ import annotations

from typing import Any, Callable

from livekit.agents.llm import RawFunctionTool, function_tool
from livekit.agents.voice import RunContext

from services.tools.executor import ToolExecutor, load_tool_schemas


def build_livekit_tools(
    get_executor: Callable[[RunContext], ToolExecutor],
    on_tool_start: Callable[[RunContext, str], None] | None = None,
) -> list[RawFunctionTool]:
    """`on_tool_start(context, tool_name)` runs right before the executor: the agent speaks its filler line and logs
    `filler_start` there (services/agent/filler.py). It must not raise; the tool call goes ahead regardless."""
    tools: list[RawFunctionTool] = []
    for name, schema in load_tool_schemas().items():

        def make(tool_name: str):
            async def handler(raw_arguments: dict[str, Any], context: RunContext) -> str:
                if on_tool_start is not None:
                    on_tool_start(context, tool_name)
                return await get_executor(context).execute(tool_name, raw_arguments)

            handler.__name__ = tool_name
            handler.__qualname__ = tool_name
            return handler

        tools.append(function_tool(make(name), raw_schema=schema["function"]))
    return tools
