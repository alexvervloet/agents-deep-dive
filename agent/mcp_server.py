"""
agent/mcp_server.py: a tiny tool server that speaks MCP, from scratch.

MCP (the Model Context Protocol) answers one question: *how does an agent use a
tool it didn't ship with?* In this repo a tool is a Python object the agent
imports directly (`agent/tools.py`). That's fine until the tool lives somewhere
else: another team's service, a vendor's product, a process in another language.
MCP standardizes the wire format so any agent can talk to any tool server: the
server advertises its tools, and the client calls them, over one agreed protocol.

This file is a real (if minimal) MCP-style server. It speaks the actual MCP
shapes (JSON-RPC 2.0 over stdio, the `tools/list` and `tools/call` methods, the
`inputSchema` field, and `content` blocks back) but built from scratch with the
standard library so you can *see* the protocol instead of importing an SDK. The
tools it serves are the very same functions from `agent/tools.py`; the only thing
that changed is they're now reachable over a protocol instead of an import.

Run it directly to poke at it by hand:

    python agent/mcp_server.py
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "calculator", "arguments": {"expression": "6 * 7"}}}

But normally you don't type at it; `examples/10_mcp.py` launches it as a
subprocess and drives it as a client. Protocol traffic goes on stdout; anything
human is logged to stderr so it never corrupts the channel.
"""

import json
import os
import sys

# Make `import agent` work whether this file is launched directly
# (`python agent/mcp_server.py`) or spawned as a subprocess by the client.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The tools we expose are the SAME functions the rest of the repo uses: we're
# only changing how they're *reached* (a protocol), not what they *are*.
from agent.contracts import ExecutionContext, ToolExecutor
from agent.providers import ToolCall
from agent.tools import CALCULATOR, SEARCH_NOTES


# Map a tool name to its MCP descriptor. MCP uses `inputSchema` (camelCase) where
# this repo's Tool uses `parameters`; same JSON Schema, different key, so we
# translate once, here. The functions themselves are reached only through
# _EXECUTOR below: a client is no more trusted than a model, so nothing on this
# server calls a tool without a contract decision first.
def _descriptor(tool):
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.parameters,
    }


_TOOLS = {t.name: _descriptor(t) for t in (CALCULATOR, SEARCH_NOTES)}
_EXECUTOR = ToolExecutor([CALCULATOR, SEARCH_NOTES])
_CONTEXT = ExecutionContext(
    request_id="mcp-server-process",
    subject_id="local-mcp-client",
    tenant_id="local-workspace",
    roles=frozenset({"user"}),
)


def _result(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(request: dict) -> dict | None:
    """Turn one JSON-RPC request into one response (or None for a notification)."""
    request_id = request.get("id")
    method = request.get("method")

    if method == "initialize":
        # A real handshake negotiates protocol version + capabilities; we keep a
        # token version of it so the shape is recognizable.
        return _result(request_id, {"serverInfo": {"name": "nimbus-tools", "version": "0.1"}})

    if method == "tools/list":
        return _result(request_id, {"tools": list(_TOOLS.values())})

    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in _TOOLS:
            return _error(request_id, -32602, f"unknown tool {name!r}")
        outcome = _EXECUTOR.execute(
            ToolCall(str(request_id), str(name), arguments),
            _CONTEXT,
        )
        if not outcome.ok:
            return _result(request_id, {
                "content": [{"type": "text", "text": outcome.for_model()}],
                "isError": True,
            })
        return _result(
            request_id,
            {"content": [{"type": "text", "text": outcome.for_model()}]},
        )

    if request_id is None:
        return None  # a notification (no id), nothing to reply
    return _error(request_id, -32601, f"unknown method {method!r}")


def main() -> None:
    print("mcp_server: ready (tools/list, tools/call)", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
