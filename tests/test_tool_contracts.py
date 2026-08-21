"""Adversarial and counterfactual tests for the client tool boundary."""

from __future__ import annotations

import hashlib
import json
import os
import time
import unittest
from unittest.mock import patch

import agent
from agent import providers


def _refund_tool(effect_log: list[dict]) -> agent.Tool:
    """Build the realistic contract used across independent policy probes."""

    def refund_order(
        order_id: str,
        amount_cents: int,
        reason: str,
        tenant_id: str,
        requested_by: str,
    ) -> str:
        effect_log.append(
            {
                "order_id": order_id,
                "amount_cents": amount_cents,
                "reason": reason,
                "tenant_id": tenant_id,
                "requested_by": requested_by,
            }
        )
        return f"refund:{order_id}:{amount_cents}"

    return agent.Tool(
        name="refund_order",
        description="Refund part of one order.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "pattern": "^ORD-[0-9]{4}$",
                },
                "amount_cents": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5_000,
                },
                "reason": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 40,
                    "enum": ["duplicate", "damaged", "late"],
                },
            },
            "required": ["order_id", "amount_cents", "reason"],
            "additionalProperties": False,
        },
        func=refund_order,
        dangerous=True,
        mutating=True,
        allowed_roles=frozenset({"billing"}),
        trusted_context_args={
            "tenant_id": "tenant_id",
            "requested_by": "subject_id",
        },
        timeout_seconds=0.2,
        maximum_output_bytes=128,
    )


def _billing_context(request_id: str = "request-1") -> agent.ExecutionContext:
    return agent.ExecutionContext(
        request_id=request_id,
        subject_id="user:bea",
        tenant_id="acme",
        roles=frozenset({"billing"}),
    )


def _valid_call(call_id: str = "call-1", **changes: object) -> agent.ToolCall:
    arguments: dict[str, object] = {
        "order_id": "ORD-0042",
        "amount_cents": 700,
        "reason": "damaged",
    }
    arguments.update(changes)
    return agent.ToolCall(call_id, "refund_order", arguments)


class ToolContractTests(unittest.TestCase):
    def test_malformed_provider_json_cannot_become_a_valid_empty_call(self) -> None:
        dispatches: list[str] = []
        tool = agent.Tool(
            "empty",
            "Accept no model arguments.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            lambda: dispatches.append("ran"),
        )
        call = agent.ToolCall(
            "malformed-1",
            "empty",
            {},
            parse_error="arguments were not valid JSON",
        )

        outcome = agent.ToolExecutor([tool]).execute(call, _billing_context())

        self.assertEqual(outcome.code, "invalid_arguments")
        self.assertEqual(dispatches, [])

    def test_every_schema_constraint_blocks_dispatch(self) -> None:
        invalid_arguments = {
            "extra": {
                "order_id": "ORD-0042",
                "amount_cents": 700,
                "reason": "damaged",
                "currency": "USD",
            },
            "missing": {"order_id": "ORD-0042", "reason": "damaged"},
            "wrong_type": {
                "order_id": "ORD-0042",
                "amount_cents": "700",
                "reason": "damaged",
            },
            "enum": {
                "order_id": "ORD-0042",
                "amount_cents": 700,
                "reason": "changed_mind",
            },
            "range": {
                "order_id": "ORD-0042",
                "amount_cents": 5_001,
                "reason": "damaged",
            },
            "pattern": {
                "order_id": "order-42",
                "amount_cents": 700,
                "reason": "damaged",
            },
            "length": {
                "order_id": "ORD-0042",
                "amount_cents": 700,
                "reason": "x",
            },
        }
        for index, (mutation, arguments) in enumerate(invalid_arguments.items()):
            with self.subTest(mutation=mutation):
                effects: list[dict] = []
                executor = agent.ToolExecutor([_refund_tool(effects)])
                outcome = executor.execute(
                    agent.ToolCall(f"invalid-{index}", "refund_order", arguments),
                    _billing_context(),
                    approve=lambda _call: True,
                )
                self.assertEqual(outcome.code, "schema_validation")
                self.assertEqual(effects, [])

    def test_trusted_fields_are_rejected_then_injected_from_context(self) -> None:
        effects: list[dict] = []
        executor = agent.ToolExecutor([_refund_tool(effects)])

        forged = executor.execute(
            _valid_call(tenant_id="attacker"),
            _billing_context(),
            approve=lambda _call: True,
        )
        self.assertEqual(forged.code, "trusted_context_forgery")
        self.assertEqual(effects, [])

        allowed = executor.execute(
            _valid_call(),
            _billing_context(),
            approve=lambda _call: True,
        )
        self.assertTrue(allowed.ok)
        self.assertEqual(effects[0]["tenant_id"], "acme")
        self.assertEqual(effects[0]["requested_by"], "user:bea")

    def test_role_and_approval_controls_change_independently(self) -> None:
        effects: list[dict] = []
        executor = agent.ToolExecutor([_refund_tool(effects)])
        support = agent.ExecutionContext(
            "request-1", "user:sam", "acme", frozenset({"support"})
        )

        wrong_role = executor.execute(
            _valid_call("wrong-role"), support, approve=lambda _call: True
        )
        no_approval = executor.execute(
            _valid_call("no-approval"), _billing_context()
        )
        denied = executor.execute(
            _valid_call("denied"),
            _billing_context(),
            approve=lambda _call: False,
        )
        allowed = executor.execute(
            _valid_call("allowed"),
            _billing_context(),
            approve=lambda _call: True,
        )

        self.assertEqual(wrong_role.code, "not_authorized")
        self.assertIs(wrong_role.approval, agent.ApprovalState.NOT_REACHED)
        self.assertEqual(no_approval.code, "approval_required")
        self.assertIs(no_approval.approval, agent.ApprovalState.REQUIRED)
        self.assertEqual(denied.code, "approval_denied")
        self.assertIs(denied.approval, agent.ApprovalState.DENIED)
        self.assertEqual(allowed.code, "ok")
        self.assertIs(allowed.approval, agent.ApprovalState.APPROVED)
        self.assertEqual(len(effects), 1)

    def test_safe_tool_and_failed_approval_have_distinct_states(self) -> None:
        safe = agent.ToolExecutor([agent.CALCULATOR]).execute(
            agent.ToolCall("calculate", "calculator", {"expression": "6 * 7"}),
            _billing_context(),
        )
        failed = agent.ToolExecutor([_refund_tool([])]).execute(
            _valid_call("approval-error"),
            _billing_context(),
            approve=lambda _call: (_ for _ in ()).throw(RuntimeError("offline")),
        )

        self.assertEqual(safe.code, "ok")
        self.assertIs(safe.approval, agent.ApprovalState.NOT_REQUIRED)
        self.assertEqual(failed.code, "approval_error")
        self.assertIs(failed.approval, agent.ApprovalState.ERROR)

    def test_mutating_replay_scope_uses_request_call_and_tool(self) -> None:
        effects: list[dict] = []
        approvals: list[str] = []
        executor = agent.ToolExecutor([_refund_tool(effects)])

        def approve(call: agent.ToolCall) -> bool:
            approvals.append(call.id)
            return True

        first = executor.execute(_valid_call(), _billing_context(), approve=approve)
        replay = executor.execute(_valid_call(), _billing_context(), approve=approve)
        new_call = executor.execute(
            _valid_call("call-2"), _billing_context(), approve=approve
        )
        new_request = executor.execute(
            _valid_call(), _billing_context("request-2"), approve=approve
        )

        self.assertFalse(first.replayed)
        self.assertTrue(replay.replayed)
        self.assertFalse(new_call.replayed)
        self.assertFalse(new_request.replayed)
        self.assertEqual(len(effects), 3)
        self.assertEqual(approvals, ["call-1", "call-2", "call-1"])

    def test_reused_call_id_with_new_arguments_is_denied_not_replayed(self) -> None:
        effects: list[dict] = []
        executor = agent.ToolExecutor([_refund_tool(effects)])
        context = _billing_context()
        approve = lambda _call: True

        first = executor.execute(_valid_call("call-1"), context, approve=approve)
        reused = executor.execute(
            _valid_call("call-1", order_id="ORD-9999", amount_cents=4_900),
            context,
            approve=approve,
        )
        honest_retry = executor.execute(_valid_call("call-1"), context, approve=approve)

        self.assertTrue(first.ok)
        self.assertEqual(reused.code, "idempotency_key_reuse")
        self.assertFalse(reused.replayed)
        self.assertTrue(honest_retry.replayed)
        self.assertEqual(len(effects), 1)
        # The denial must be visible as its own proposal, not hidden behind the
        # settled one, so the audit trail can show what was actually attempted.
        self.assertNotEqual(reused.arguments_sha256, first.arguments_sha256)
        self.assertEqual(
            [record.code for record in executor.audit_records],
            ["ok", "idempotency_key_reuse", "ok"],
        )

    def test_replay_storage_is_bounded_and_eviction_is_explicit(self) -> None:
        effects: list[dict] = []
        executor = agent.ToolExecutor(
            [_refund_tool(effects)], maximum_replay_entries=1
        )
        context = _billing_context()
        executor.execute(_valid_call("call-1"), context, approve=lambda _call: True)
        executor.execute(_valid_call("call-2"), context, approve=lambda _call: True)
        evicted = executor.execute(
            _valid_call("call-1"), context, approve=lambda _call: True
        )

        self.assertFalse(evicted.replayed)
        self.assertEqual(len(effects), 3)

    def test_after_dispatch_failure_is_replayed_without_a_second_effect(self) -> None:
        effects: list[str] = []

        def fails_after_commit(item: str) -> str:
            effects.append(item)
            raise RuntimeError("connection lost after commit")

        tool = agent.Tool(
            "commit_then_fail",
            "Simulate an uncertain write result.",
            {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
                "additionalProperties": False,
            },
            fails_after_commit,
            mutating=True,
        )
        executor = agent.ToolExecutor([tool])
        call = agent.ToolCall("write-1", tool.name, {"item": "invoice"})

        first = executor.execute(call, _billing_context())
        replay = executor.execute(call, _billing_context())

        self.assertEqual(first.code, "tool_error")
        self.assertEqual(replay.code, "tool_error")
        self.assertTrue(replay.replayed)
        self.assertEqual(effects, ["invoice"])

    def test_timeout_and_utf8_output_bytes_have_distinct_reasons(self) -> None:
        def slow() -> str:
            time.sleep(0.03)
            return "done"

        timeout_tool = agent.Tool(
            "slow",
            "Sleep past the teaching deadline.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            slow,
            timeout_seconds=0.005,
        )
        unicode_tool = agent.Tool(
            "unicode",
            "Return two multibyte characters.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            lambda: "éé",
            maximum_output_bytes=3,
        )
        context = _billing_context()

        timeout = agent.ToolExecutor([timeout_tool]).execute(
            agent.ToolCall("slow-1", "slow", {}), context
        )
        overflow = agent.ToolExecutor([unicode_tool]).execute(
            agent.ToolCall("unicode-1", "unicode", {}), context
        )

        self.assertEqual(timeout.code, "timeout")
        self.assertEqual(overflow.code, "output_limit")
        self.assertIn("4 bytes", overflow.message)

    def test_audit_is_bounded_and_hashes_actual_effective_data(self) -> None:
        tool = agent.Tool(
            "echo",
            "Echo one value.",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            lambda value, tenant_id: f"{tenant_id}:{value}",
            trusted_context_args={"tenant_id": "tenant_id"},
        )
        executor = agent.ToolExecutor([tool], maximum_audit_entries=1)
        context = _billing_context()
        executor.execute(agent.ToolCall("echo-1", "echo", {"value": "first"}), context)
        second = executor.execute(
            agent.ToolCall("echo-2", "echo", {"value": "second"}), context
        )
        records = executor.audit_records
        expected_arguments = json.dumps(
            {"tenant_id": "acme", "value": "second"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].tool_call_id, "echo-2")
        self.assertEqual(
            second.arguments_sha256,
            hashlib.sha256(expected_arguments).hexdigest(),
        )
        self.assertEqual(
            second.output_sha256,
            hashlib.sha256(b"acme:second").hexdigest(),
        )

    def test_contract_definition_rejects_ambiguous_boundaries(self) -> None:
        open_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        }
        with self.assertRaisesRegex(ValueError, "additionalProperties"):
            agent.ToolExecutor([agent.Tool("open", "Open schema.", open_schema, lambda: None)])

        overlapping = agent.Tool(
            "overlap",
            "Expose a trusted field by mistake.",
            {
                "type": "object",
                "properties": {"tenant_id": {"type": "string"}},
                "required": ["tenant_id"],
                "additionalProperties": False,
            },
            lambda tenant_id: tenant_id,
            trusted_context_args={"tenant_id": "tenant_id"},
        )
        with self.assertRaisesRegex(ValueError, "model-visible"):
            agent.ToolExecutor([overlapping])

    def test_agent_loop_never_dispatches_a_schema_invalid_call(self) -> None:
        dispatches: list[int] = []
        tool = agent.Tool(
            "increment",
            "Increment one integer.",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            lambda value: dispatches.append(value) or value + 1,
        )
        turns = iter(
            [
                providers.Turn(
                    None,
                    [providers.ToolCall("call-1", "increment", {"value": 2, "extra": 9})],
                    {"role": "assistant", "content": None},
                ),
                providers.Turn(
                    "I could not run it.",
                    [],
                    {"role": "assistant", "content": "I could not run it."},
                ),
            ]
        )
        with patch.object(providers, "run_turn", side_effect=lambda *_args: next(turns)):
            result = agent.run_agent("system", "increment two", [tool])

        self.assertEqual(dispatches, [])
        self.assertEqual(result.steps[0].status, "schema_validation")
        self.assertIn("schema_validation", result.steps[0].result)

    def test_reusable_executor_without_context_fails_before_provider_call(self) -> None:
        executor = agent.ToolExecutor([agent.CALCULATOR])
        with patch.object(providers, "run_turn") as run_turn:
            with self.assertRaisesRegex(ValueError, "context is required"):
                agent.run_agent(
                    "system",
                    "calculate two plus two",
                    [agent.CALCULATOR],
                    executor=executor,
                )
        run_turn.assert_not_called()

    def test_context_without_executor_remains_a_valid_one_shot(self) -> None:
        final = providers.Turn(
            "No tool needed.",
            [],
            {"role": "assistant", "content": "No tool needed."},
        )
        with patch.object(providers, "run_turn", return_value=final):
            result = agent.run_agent(
                "system",
                "hello",
                [agent.CALCULATOR],
                context=_billing_context(),
            )
        self.assertEqual(result.answer, "No tool needed.")

    def test_executor_and_context_together_replay_across_loop_invocations(self) -> None:
        effects: list[str] = []
        tool = agent.Tool(
            "write",
            "Record one item.",
            {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
                "additionalProperties": False,
            },
            lambda item: effects.append(item) or "written",
            mutating=True,
        )
        executor = agent.ToolExecutor([tool])
        context = _billing_context()
        requested = providers.Turn(
            None,
            [providers.ToolCall("write-1", "write", {"item": "invoice"})],
            {"role": "assistant", "content": None},
        )
        final = providers.Turn(
            "Done.",
            [],
            {"role": "assistant", "content": "Done."},
        )
        turns = iter([requested, final, requested, final])
        with patch.object(providers, "run_turn", side_effect=lambda *_args: next(turns)):
            first = agent.run_agent(
                "system", "write", [tool], context=context, executor=executor
            )
            second = agent.run_agent(
                "system", "write again", [tool], context=context, executor=executor
            )

        self.assertFalse(first.steps[0].replayed)
        self.assertTrue(second.steps[0].replayed)
        self.assertEqual(effects, ["invoice"])

    def test_openai_strictness_is_claimed_only_for_compatible_shapes(self) -> None:
        exact = agent.CALCULATOR
        optional = agent.Tool(
            "optional",
            "Has one optional field.",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            lambda value=None: value,
        )
        with patch.dict(os.environ, {"PROVIDER": "openai"}):
            schemas = providers.to_tool_schema([exact, optional])

        self.assertIs(schemas[0]["function"]["strict"], True)
        self.assertIs(schemas[1]["function"]["strict"], False)


if __name__ == "__main__":
    unittest.main()
