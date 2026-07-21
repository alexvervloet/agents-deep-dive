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
  - error handling: a tool that raises returns its error *as the result*, so the
    model can see what went wrong and try something else instead of crashing.
  - approval: a `dangerous` tool can be gated behind an `approve` callback, so the
    human-in-the-loop. A denied call comes back as a normal result, and the agent
    adapts.
"""

from dataclasses import dataclass, field

from . import providers


@dataclass
class Step:
    """A record of one tool execution, for tracing and inspection."""

    tool: str
    arguments: dict
    result: str
    approved: bool = True


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
) -> AgentResult:
    """Run the agentic loop until the model gives a final answer or hits max_steps.

    - `tools`: a list of Tool objects (see agent/tools.py).
    - `approve`: optional callback `(ToolCall) -> bool`; consulted before running a
      tool whose `dangerous` flag is set. Return False to deny.
    - `tracer`: optional Tracer to print each step.
    - `history`: optional message list. Pass the SAME list across calls to give the
      agent memory of earlier turns. The loop appends this turn's messages to it
      in place. Omit it for a one-shot run. (The API itself is stateless; "memory"
      is just you re-sending the growing list, exactly as in the sibling repos.)
    """
    by_name = {t.name: t for t in tools}
    tool_schema = providers.to_tool_schema(tools)
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
            tool = by_name.get(call.name)
            approved = True

            if tool is None:
                result = f"Error: no tool named {call.name!r}."
            elif tool.dangerous and approve is not None and not approve(call):
                approved = False
                result = "Error: the user denied permission to run this tool."
            else:
                try:
                    result = str(tool.func(**call.arguments))
                except Exception as e:  # noqa: BLE001 - feed any failure back to the model
                    result = f"Error running {call.name}: {e}"

            if tracer:
                tracer.on_tool_result(call, result)
            steps.append(Step(tool=call.name, arguments=call.arguments, result=result, approved=approved))
            results.append((call.id, result))

        history += providers.format_tool_results(results)

    # Fell out of the loop without a final answer.
    return AgentResult(
        answer="(stopped: reached the step limit without finishing)",
        steps=steps,
        stopped_early=True,
    )
