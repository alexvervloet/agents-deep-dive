"""
agent/loop.py: the agent loop. This is the whole idea.

Strip away the vocabulary and an agent is a `while` loop:

    give the model the tools and the conversation so far
    ask it for the next step
    if it asked to call tools:
        run them, append the results, loop again
    else:
        it gave a final answer, stop

That's `run_agent` below, in ~20 lines. Everything else in this repo (multiple
tools, error recovery, approval gates, tracing, memory, sub-agents) is a small
addition to this loop, not a new concept.

Three pieces of control logic worth seeing here, because they're what make a loop
safe instead of a runaway:

  - max_steps: a hard ceiling so a confused model can't loop forever.
  - contract execution: every model-proposed call is locally validated,
    authorized, bounded, and recorded before its function can run.
  - error handling: a structured tool failure returns *as the result*, so the
    model can adapt instead of crashing the loop.
  - approval: a `dangerous` tool fails closed unless an `approve` callback allows
    it. A denial comes back as a normal result, and the agent adapts.
"""

from dataclasses import dataclass, field

from . import providers
from .contracts import ExecutionContext, ToolExecutor


@dataclass
class Step:
    """A record of one tool execution, for tracing and inspection."""

    tool: str
    arguments: dict
    result: str
    approved: bool = True
    status: str = "ok"
    replayed: bool = False
    arguments_sha256: str = ""
    output_sha256: str | None = None


@dataclass
class AgentResult:
    """What run_agent returns: the final answer plus the steps it took to get there."""

    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_early: bool = False


class Tracer:
    """A minimal step-by-step printer (stdlib only). Pass one to run_agent to
    watch the agent think out loud; example 07 and the capstone use it."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._n = 0

    def on_tool_call(self, call: providers.ToolCall) -> None:
        if not self.enabled:
            return
        self._n += 1
        args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
        print(f"  step {self._n}: {call.name}({args})")

    def on_tool_result(self, call: providers.ToolCall, result: str) -> None:
        if not self.enabled:
            return
        shown = result if len(result) <= 120 else result[:117] + "..."
        print(f"          -> {shown}")

    def on_final(self, text: str) -> None:
        if self.enabled:
            print("  (final answer reached)")


def run_agent(
    system: str,
    user_input: str,
    tools: list,
    max_steps: int = 6,
    approve=None,
    tracer: Tracer | None = None,
    history: list | None = None,
    context: ExecutionContext | None = None,
    executor: ToolExecutor | None = None,
) -> AgentResult:
    """Run the agentic loop until the model gives a final answer or hits max_steps.

    - `tools`: a list of Tool objects (see agent/tools.py).
    - `approve`: callback `(ToolCall) -> bool`; dangerous tools fail closed when
      it is absent and run only when it returns True.
    - `tracer`: optional Tracer to print each step.
    - `history`: optional message list. Pass the SAME list across calls to give the
      agent memory of earlier turns. The loop appends this turn's messages to it
      in place. Omit it for a one-shot run. (The API itself is stateless; "memory"
      is just you re-sending the growing list, exactly as in the sibling repos.)
    - `context`: trusted request identity, tenant, and roles. The local teaching
      context is used when omitted; server applications should always pass their
      authenticated context.
    - `executor`: optional reusable ToolExecutor. Its replay cache is keyed partly
      on `context.request_id`, so reusing an executor only recognizes a retry if
      you pass the SAME `context` too. Reuse one without the other and every call
      looks new, silently: a fresh `ExecutionContext.local()` mints a fresh ID.
    """
    tool_schema = providers.to_tool_schema(tools)
    context = context or ExecutionContext.local()
    executor = executor or ToolExecutor(tools)
    if history is None:
        history = []
    history.append(providers.user_message(user_input))
    steps: list[Step] = []

    for _ in range(max_steps):
        turn = providers.run_turn(system, history, tool_schema)
        history.append(turn.raw_assistant)

        # No tool calls -> the model is done.
        if not turn.tool_calls:
            if tracer:
                tracer.on_final(turn.text or "")
            return AgentResult(answer=turn.text or "", steps=steps)

        # Otherwise: run each requested tool and collect results to feed back.
        results = []
        for call in turn.tool_calls:
            if tracer:
                tracer.on_tool_call(call)
            outcome = executor.execute(call, context, approve=approve)
            result = outcome.for_model()

            if tracer:
                tracer.on_tool_result(call, result)
            steps.append(
                Step(
                    tool=call.name,
                    arguments=call.arguments,
                    result=result,
                    approved=outcome.approved,
                    status=outcome.code,
                    replayed=outcome.replayed,
                    arguments_sha256=outcome.arguments_sha256,
                    output_sha256=outcome.output_sha256,
                )
            )
            results.append((call.id, result))

        history += providers.format_tool_results(results)

    # Fell out of the loop without a final answer.
    return AgentResult(
        answer="(stopped: reached the step limit without finishing)",
        steps=steps,
        stopped_early=True,
    )
