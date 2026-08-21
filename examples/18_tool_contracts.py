#!/usr/bin/env python3
"""18_tool_contracts.py: treat every tool call as untrusted input (offline).

    python examples/18_tool_contracts.py

Prediction before running: only ``allowed`` should create a refund. ``forged``
must fail even though its tenant looks plausible, ``too_large`` must fail local
schema validation, ``wrong_role`` must fail authorization, and ``replay`` must
return the first result without creating a second refund.

The expected shape is five distinct decision codes and ``effects committed: 1``.
The expectations above are fixed course requirements; the executor receives only
calls and trusted context, never an expected answer to copy.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent


REFUND_EFFECTS: list[dict[str, object]] = []


def issue_refund(
    order_id: str,
    amount_cents: int,
    reason: str,
    tenant_id: str,
    requested_by: str,
) -> str:
    """Record one stand-in refund using identity injected by trusted context."""

    REFUND_EFFECTS.append(
        {
            "order_id": order_id,
            "amount_cents": amount_cents,
            "reason": reason,
            "tenant_id": tenant_id,
            "requested_by": requested_by,
        }
    )
    return f"refund accepted for {order_id}: {amount_cents} cents"


REFUND = agent.Tool(
    name="issue_refund",
    description="Refund up to 5000 cents on one order.",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "pattern": "^ORD-[0-9]{4}$"},
            "amount_cents": {"type": "integer", "minimum": 1, "maximum": 5_000},
            "reason": {
                "type": "string",
                "enum": ["duplicate", "damaged", "late"],
            },
        },
        "required": ["order_id", "amount_cents", "reason"],
        "additionalProperties": False,
    },
    func=issue_refund,
    allowed_roles=frozenset({"billing"}),
    trusted_context_args={
        "tenant_id": "tenant_id",
        "requested_by": "subject_id",
    },
    dangerous=True,
    mutating=True,
    timeout_seconds=1,
    maximum_output_bytes=256,
)


def context(roles: frozenset[str]) -> agent.ExecutionContext:
    """Return authenticated application context outside the model's arguments."""

    return agent.ExecutionContext(
        request_id="http-request-7",
        subject_id="user:bea",
        tenant_id="acme",
        roles=roles,
    )


def call(call_id: str, **changes: object) -> agent.ToolCall:
    """Create one model proposal; changes are probes, not expected decisions."""

    arguments: dict[str, object] = {
        "order_id": "ORD-0042",
        "amount_cents": 700,
        "reason": "damaged",
    }
    arguments.update(changes)
    return agent.ToolCall(call_id, "issue_refund", arguments)


executor = agent.ToolExecutor([REFUND], maximum_replay_entries=8)
billing = context(frozenset({"billing"}))
support = context(frozenset({"support"}))
approve = lambda _call: True

probes = [
    ("allowed", call("refund-1"), billing),
    ("forged", call("refund-2", tenant_id="other-company"), billing),
    ("too_large", call("refund-3", amount_cents=5_001), billing),
    ("wrong_role", call("refund-4"), support),
    ("replay", call("refund-1"), billing),
]

print("Prediction: ok, trusted_context_forgery, schema_validation, not_authorized, ok(replay)\n")
for label, proposal, trusted_context in probes:
    outcome = executor.execute(proposal, trusted_context, approve=approve)
    replay_marker = " replayed" if outcome.replayed else ""
    print(f"{label:12} -> {outcome.code}{replay_marker}")

print(f"\neffects committed: {len(REFUND_EFFECTS)}")
print(f"effective tenant:  {REFUND_EFFECTS[0]['tenant_id']}")
print(f"effective subject: {REFUND_EFFECTS[0]['requested_by']}")
print(f"audit records:     {len(executor.audit_records)} (digests, not raw payloads)")

print(
    "\nTakeaway: the schema is request syntax, not authority. The application "
    "validates again locally, derives identity from authenticated context, checks "
    "policy and approval, and only then crosses the side-effect boundary."
)
