"""Prove protocol-served tools cross the same local contract boundary."""

import unittest

from agent.mcp_server import handle


class MCPContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
