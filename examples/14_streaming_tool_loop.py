"""
Example 14: streaming WITHIN the tool loop.

Example 13 streamed the *final* answer, after every tool had run. But the answer
isn't the only thing worth streaming. In a real assistant the model often narrates
as it works ("Let me look up the price... now I'll compute the total..."), and that
text appears on the *same* turns that request tools. Streaming only the final answer
makes the user stare at a spinner through every tool step; streaming inside the loop
makes the whole thing feel alive, the pattern most production assistants use.

The change is small: swap the loop's non-streaming `run_turn` for `stream_turn`,
which prints the assistant's text token by token via a callback AND still returns
the normalized tool calls. Everything else (run the tools, feed results back, loop)
is the agent loop you already know from example 03.

This drives a two-step task (look up a price, then do arithmetic on it) and streams
every turn, so you can watch the agent think between tool calls.

Run it:

    secrun python examples/14_streaming_tool_loop.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from agent import providers
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")

# Reuse the safe default tools: a calculator and a read-only notes search.
TOOLS = [agent.CALCULATOR, agent.SEARCH_NOTES]

SYSTEM = (
    "You are a helpful assistant for Nimbus Notes. Use search_notes for product "
    "facts and the calculator for arithmetic. Briefly narrate what you're doing "
    "before each tool call, then give a final answer."
)
QUESTION = "The Plus plan: what does a full year cost, and what's 15% tax on that yearly price?"


def streaming_agent(system: str, question: str, tools: list, max_steps: int = 6) -> str:
    """The example-03 loop, but each turn STREAMS its text as it arrives."""
    schema = providers.to_tool_schema(tools)
    executor = agent.ToolExecutor(tools)
    context = agent.ExecutionContext.local("streaming-example")
    history = [providers.user_message(question)]

    for step in range(max_steps):
        # Stream this turn's text live; tool calls come back normalized as usual.
        print(f"\n[turn {step + 1}] ", end="", flush=True)
        turn = providers.stream_turn(
            system,
            history,
            schema,
            on_text=lambda piece: print(piece, end="", flush=True),
        )
        print()  # newline after the streamed text
        history.append(turn.raw_assistant)  # type: ignore[arg-type]

        if not turn.tool_calls:
            return turn.text or ""

        # Enforce each request and feed the result back (same as the basic loop).
        results = []
        for call in turn.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
            outcome = executor.execute(call, context)
            result = outcome.for_model()
            print(f"  ↳ {call.name}({args}) -> {result} [{outcome.code}]")
            results.append((call.id, result))
        history += providers.format_tool_results(results)

    return "(stopped: reached the step limit)"


if __name__ == "__main__":
    print(f"Question: {QUESTION}")
    answer = streaming_agent(SYSTEM, QUESTION, TOOLS)
    print(f"\nFinal answer: {answer}")
    print(
        "\nTakeaway: streaming isn't just for the final answer. Swap `run_turn` for\n"
        "`stream_turn` and the agent narrates live through every tool step, the same\n"
        "loop, but it now feels responsive instead of frozen between calls. (Accumulating\n"
        "streamed tool-call fragments is the one fiddly bit; it lives in agent/providers.py.)"
    )
