"""Prove protocol-served tools cross the same local contract boundary."""

import runpy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent
from agent import providers
from agent.mcp_server import handle

# What a schema from someone else's MCP server usually looks like: a real JSON
# Schema that simply never says whether extra fields are allowed.
DISCOVERED_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
}


class MCPContractTests(unittest.TestCase):
    def test_client_refuses_a_transport_without_stdio_streams(self) -> None:
        namespace = runpy.run_path("examples/10_mcp.py")
        client = object.__new__(namespace["MCPClient"])
        client._proc = SimpleNamespace(stdin=None, stdout=None)
        client._next_id = 0

        with self.assertRaisesRegex(RuntimeError, "missing its stdio streams"):
            client._call("tools/list")

    def test_server_rejects_undeclared_arguments_instead_of_dispatching(self) -> None:
        response = handle(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "calculator",
                    "arguments": {"expression": "6 * 7", "debug": True},
                },
            }
        )

        self.assertIsNotNone(response)
        result = response["result"]
        self.assertIs(result["isError"], True)
        self.assertIn("schema_validation", result["content"][0]["text"])

    def test_server_returns_a_valid_contract_result(self) -> None:
        response = handle(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "calculator",
                    "arguments": {"expression": "6 * 7"},
                },
            }
        )

        self.assertIsNotNone(response)
        self.assertEqual(response["result"]["content"][0]["text"], "42")


class DiscoveredSchemaTests(unittest.TestCase):
    def test_an_unsealed_schema_is_refused_and_sealing_admits_it(self) -> None:
        def remote_search(query: str, limit: int = 10) -> str:
            return f"{query}:{limit}"

        unsealed = agent.Tool(
            "remote_search", "From another team's server.", DISCOVERED_SCHEMA, remote_search
        )
        with self.assertRaisesRegex(ValueError, "additionalProperties"):
            agent.ToolExecutor([unsealed])

        sealed = agent.Tool(
            "remote_search",
            "From another team's server.",
            agent.seal_schema(DISCOVERED_SCHEMA),
            remote_search,
        )
        executor = agent.ToolExecutor([sealed])
        allowed = executor.execute(
            agent.ToolCall("c1", "remote_search", {"query": "plans"}),
            agent.ExecutionContext.local("discovery"),
        )
        invented = executor.execute(
            agent.ToolCall("c2", "remote_search", {"query": "plans", "admin": True}),
            agent.ExecutionContext.local("discovery"),
        )

        self.assertEqual(allowed.output, "plans:10")
        self.assertEqual(invented.code, "schema_validation")
        # Sealing must not edit the descriptor the server sent us.
        self.assertNotIn("additionalProperties", DISCOVERED_SCHEMA)

    def test_a_sealed_discovered_tool_runs_in_the_agent_loop(self) -> None:
        sealed = agent.Tool(
            "remote_search",
            "From another team's server.",
            agent.seal_schema(DISCOVERED_SCHEMA),
            lambda query, limit=10: f"{query}:{limit}",
        )
        turns = iter(
            [
                providers.Turn(
                    None,
                    [providers.ToolCall("c1", "remote_search", {"query": "plans"})],
                    {"role": "assistant", "content": None},
                ),
                providers.Turn("Found them.", [], {"role": "assistant", "content": "Found them."}),
            ]
        )
        with patch.object(providers, "run_turn", side_effect=lambda *_args: next(turns)):
            result = agent.run_agent("system", "find the plans", [sealed])

        self.assertEqual(result.steps[0].status, "ok")
        self.assertEqual(result.steps[0].result, "plans:10")


if __name__ == "__main__":
    unittest.main()
