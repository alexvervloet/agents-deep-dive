#!/usr/bin/env python3
"""
10_mcp.py: use a tool you didn't ship with, over a protocol (offline, no key).

    python examples/10_mcp.py

Every earlier example imported its tools straight from `agent/tools.py`. Real
agents often can't: the tool lives in another team's service, a vendor's product,
or a process in another language. MCP (Model Context Protocol) is the standard
that makes that work: the tool *server* advertises what it offers, and the agent
*client* discovers and calls those tools over one agreed wire format, with no
bespoke glue per tool.

This script is the client. It launches `agent/mcp_server.py` as a subprocess and
speaks MCP to it: `tools/list` to discover what's there, then `tools/call` to run
one. The payoff is the conversion step: each remote tool descriptor becomes an
ordinary `Tool` object (the same dataclass from `agent/tools.py`), so an
MCP-served tool drops into the loop from example 03 unchanged. To the agent loop,
"a local function" and "a tool on a server across the world" look identical.

It's fully offline: the protocol is just JSON lines over a pipe, and the tools
(calculator, search_notes) run locally in the server process. No model, no key.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import Tool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO_ROOT, "agent", "mcp_server.py")


class MCPClient:
    """A minimal MCP client: spawn a tool server and call it over stdio.

    A production client would use the official `mcp` SDK and support more
    transports (HTTP/SSE), auth, and streaming. The protocol *shape* (list tools,
    call a tool by name with JSON arguments, read content blocks back) is exactly
    this.
    """

    def __init__(self, command: list[str]):
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._next_id = 0

    def _call(self, method: str, params: dict | None = None):
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            request["params"] = params
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        response = json.loads(self._proc.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"]["message"])
        return response["result"]

    def list_tools(self) -> list[dict]:
        return self._call("tools/list")["tools"]

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        # MCP returns a list of content blocks; we read the text ones (same idea
        # as reading Claude's content blocks in the API repo).
        return "".join(b["text"] for b in result["content"] if b["type"] == "text")

    def as_tools(self) -> list[Tool]:
        """Turn every remote tool into a local `Tool` object. Its `func` is a
        closure that makes a `tools/call` over the protocol, so the rest of the
        repo (the loop, the tracer, approval gates) treats it like any other tool."""
        tools = []
        for desc in self.list_tools():
            name = desc["name"]
            tools.append(
                Tool(
                    name=name,
                    description=desc["description"],
                    parameters=desc["inputSchema"],  # MCP's key for the JSON Schema
                    func=lambda _name=name, **kwargs: self.call_tool(_name, kwargs),
                )
            )
        return tools

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait(timeout=5)


def main() -> None:
    client = MCPClient([sys.executable, SERVER])
    try:
        print("Connecting to the MCP tool server and asking what it offers...\n")
        tools = client.as_tools()
        for tool in tools:
            print(f"  • {tool.name}: {tool.description}")

        print("\nCalling tools over the protocol (the agent never imported them):\n")
        print(
            "  calculator(expression='6 * 7') ->",
            client.call_tool("calculator", {"expression": "6 * 7"}),
        )
        print("  search_notes(query='what plans are offered') ->")
        for line in client.call_tool(
            "search_notes", {"query": "what plans are offered"}
        ).splitlines():
            print("     ", line)

        print("\nThese came back as ordinary Tool objects:")
        print(f"  {[t.name for t in tools]}")
        print("...so they drop straight into `run_agent(...)` from example 03. The loop")
        print("can't tell a local function from a tool served across a protocol.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
