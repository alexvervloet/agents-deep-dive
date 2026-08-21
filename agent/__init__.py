"""
agent: a small, from-scratch agent framework.

Built to be *read*. The pieces:

  tools.py      what a tool is (name + description + schema + function), and a
                  safe default toolbox (calculator, search_notes, save_note)
  contracts.py  validates, authorizes, bounds, executes, and audits tool calls,
                  plus seal_schema for adopting a schema you did not write
  providers.py  the ONLY provider-specific file: normalizes a tool-calling turn
                  across the openai/claude stacks
  loop.py       run_agent: the while-loop that IS the agent, plus a Tracer

Typical use:

    from agent import run_agent, default_tools
    result = run_agent("You are a helpful assistant.", "What is 12*9?", default_tools())
    print(result.answer)
"""

from .contracts import (
    ApprovalState,
    ExecutionContext,
    ToolAuditRecord,
    ToolExecutor,
    ToolOutcome,
    seal_schema,
)
from .loop import AgentResult, Step, Tracer, run_agent
from .providers import (
    HostedResult,
    ToolCall,
    Turn,
    describe,
    ensure_ready,
    hosted_web_search,
    provider_name,
)
from .tools import (
    CALCULATOR,
    SAVE_NOTE,
    SEARCH_NOTES,
    Tool,
    default_tools,
)

__all__ = [
    "Tool",
    "default_tools",
    "CALCULATOR",
    "SEARCH_NOTES",
    "SAVE_NOTE",
    "ApprovalState",
    "ExecutionContext",
    "ToolExecutor",
    "ToolOutcome",
    "ToolAuditRecord",
    "seal_schema",
    "run_agent",
    "AgentResult",
    "Step",
    "Tracer",
    "ToolCall",
    "Turn",
    "HostedResult",
    "hosted_web_search",
    "provider_name",
    "describe",
    "ensure_ready",
]
