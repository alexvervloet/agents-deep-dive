"""
Example 07: observability: see what the agent did (offline, no key).

An agent makes its own decisions, so when it misbehaves you need to see *why*.
The cure is a trace: a record of every step, including the tool, arguments,
result category, approval state, replay decision, and privacy-preserving digests.

This example scripts only the model's proposals so the contract evidence is
deterministic. The real loop and executor handle both attempts. The second attempt
reuses the trusted request context and call ID, so predict before running: will the
effect happen twice, or will the step say it was replayed?

Run it:

    python examples/07_observability.py
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from agent import providers


events: list[str] = []


def record_audit_event(event: str) -> str:
    """Cross the example's effect boundary exactly once per accepted call."""

    events.append(event)
    return f"recorded event: {event}"


AUDIT_TOOL = agent.Tool(
    name="record_audit_event",
    description="Record one audit event in the application event sink.",
    parameters={
        "type": "object",
        "properties": {
            "event": {"type": "string", "minLength": 1, "maxLength": 120}
        },
        "required": ["event"],
        "additionalProperties": False,
    },
    func=record_audit_event,
    dangerous=True,
    mutating=True,
)
CALL = providers.ToolCall(
    "audit-call-1",
    "record_audit_event",
    {"event": "customer requested an account export"},
)


def scripted_turns(final_answer: str):
    """Yield one model proposal and one final answer; never supply the outcome."""

    return iter(
        [
            providers.Turn(
                text=None,
                tool_calls=[CALL],
                raw_assistant={"role": "assistant", "content": None},
            ),
            providers.Turn(
                text=final_answer,
                tool_calls=[],
                raw_assistant={"role": "assistant", "content": final_answer},
            ),
        ]
    )


def run_scripted_attempt(
    final_answer: str,
    *,
    executor: agent.ToolExecutor,
    context: agent.ExecutionContext,
    tracer: agent.Tracer | None = None,
) -> agent.AgentResult:
    """Run the real loop against deterministic model proposals."""

    turns = scripted_turns(final_answer)
    with patch.object(providers, "run_turn", side_effect=lambda *_args: next(turns)):
        return agent.run_agent(
            "Record the requested event, then confirm it.",
            "Record the account-export request.",
            [AUDIT_TOOL],
            approve=lambda _call: True,
            tracer=tracer,
            context=context,
            executor=executor,
        )


context = agent.ExecutionContext.local("observability-example")
executor = agent.ToolExecutor([AUDIT_TOOL])

print("First attempt, with the live tracer:")
first = run_scripted_attempt(
    "The event was recorded.", executor=executor, context=context, tracer=agent.Tracer()
)
print(f"Final answer: {first.answer}\n")

print("Retry with the same trusted request context, call ID, and payload:")
retry = run_scripted_attempt("The prior result was reused.", executor=executor, context=context)
print(f"Final answer: {retry.answer}\n")

# Post-hoc: these are the structured records a production tracer or eval consumes.
print("Structured trace (result.steps):")
for i, step in enumerate([*first.steps, *retry.steps], start=1):
    preview = " ".join(step.result.split())[:70]
    print(
        f"  {i}. {step.tool}({step.arguments}) "
        f"status={step.status} approval={step.approval.value} "
        f"replayed={step.replayed}"
    )
    print(f"     -> {preview}")
    print(f"     args sha256={step.arguments_sha256[:12]}...")
    output_digest = (
        f"{step.output_sha256[:12]}..."
        if step.output_sha256 is not None
        else "none"
    )
    print(f"     output sha256={output_digest}")

print(f"\nEffects committed: {len(events)} (expected 1: the retry used the cache)")
print(
    "Read `status` for why the call ended and `approval` for how far it reached "
    "through that separate gate. `replayed` distinguishes dispatch from cache use. "
    "The digests compare attempts without copying raw customer data into the trace; "
    "an output digest of `none` would mean no output crossed the boundary."
)
