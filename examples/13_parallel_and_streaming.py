"""
Example 13: parallel tool calls & streaming the answer.

Two upgrades that make an agent faster and feel faster, both layered on the same
loop you already know.

  PARALLEL TOOLS. In one turn the model often asks for several *independent* tool
  calls at once (weather in 3 cities, 3 database lookups). The basic loop runs them
  one after another; but independent calls have no reason to wait on each other, so
  you run them concurrently and the turn finishes in the time of the slowest call,
  not the sum.

  STREAMING. The final answer is just a chat completion, so you can stream it token
  by token (exactly like the API deep dives), so the user reads immediately instead of
  staring at a spinner while a long answer generates.

This script drives one turn that fans out into several (deliberately slow) tool
calls, times sequential vs. parallel execution, then STREAMS the final answer.

Run it:

    secrun python examples/13_parallel_and_streaming.py
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from agent import providers
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")

MODELS = {
    "openai": "gpt-5.4-nano",
    "claude": "claude-haiku-4-5",
}  # mirrors agent/providers.py


# --- A deliberately SLOW tool, so the parallel speedup is visible. ------------
def slow_weather(city: str) -> str:
    time.sleep(0.8)  # pretend this is a network call
    fake = {"Paris": "18C rain", "Tokyo": "27C sunny", "Cairo": "33C clear"}
    return fake.get(city, "unknown")


WEATHER = agent.Tool(
    name="get_weather",
    description="Get the current weather for a city. Call once per city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    func=slow_weather,
)

SYSTEM = (
    "You are a travel assistant. The user names several cities; call get_weather "
    "for EACH city (all of them) and then summarize."
)
QUESTION = "What's the weather in Paris, Tokyo, and Cairo right now? Summarize in one line each."


def run_one_turn():
    """Run a single assistant turn and return (calls, history). The model should
    request one weather call per city in this one turn."""
    history = [providers.user_message(QUESTION)]
    schema = providers.to_tool_schema([WEATHER])
    turn = providers.run_turn(SYSTEM, history, schema)
    history.append(turn.raw_assistant)  # type: ignore[arg-type]
    return turn.tool_calls, history


def execute(calls, parallel: bool):
    """Run the requested tool calls, sequentially or concurrently; return results."""

    def run(call):
        return (call.id, slow_weather(**call.arguments))

    if parallel:
        with ThreadPoolExecutor(max_workers=len(calls) or 1) as pool:
            return list(pool.map(run, calls))
    return [run(c) for c in calls]


def stream_final(system: str, history: list):
    """Stream the final answer token by token (no tools, just the answer)."""
    name = providers.provider_name()
    if name == "openai":
        from openai import OpenAI

        stream = OpenAI().chat.completions.create(
            model=MODELS["openai"],
            messages=[{"role": "system", "content": system}, *history],
            stream=True,
        )
        for chunk in stream:
            piece = chunk.choices[0].delta.content
            if piece:
                print(piece, end="", flush=True)
    else:  # claude
        import anthropic

        with anthropic.Anthropic().messages.stream(
            model=MODELS["claude"], max_tokens=400, system=system, messages=history
        ) as s:
            for text in s.text_stream:
                print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    print(f"Question: {QUESTION}\n")
    calls, history = run_one_turn()
    print(
        f"The model requested {len(calls)} tool call(s) in one turn: "
        f"{', '.join(c.arguments.get('city', '?') for c in calls)}\n"
    )

    if len(calls) < 2:
        print(
            "(The model didn't fan out this run; parallel speedup needs ≥2 independent\n"
            " calls in one turn. Re-run, or try the other provider.)"
        )

    # 1) Time sequential vs. parallel execution of the SAME calls.
    t0 = time.time()
    execute(calls, parallel=False)
    seq = time.time() - t0
    t0 = time.time()
    results = execute(calls, parallel=True)
    par = time.time() - t0
    print(f"sequential tool execution: {seq:.1f}s")
    print(
        f"parallel tool execution:   {par:.1f}s   ->  ~{(seq / par) if par else 1:.1f}x faster\n"
    )

    # 2) Feed the results back, then STREAM the final answer.
    history += providers.format_tool_results(results)
    print("Final answer (streaming):")
    stream_final(SYSTEM, history)

    print(
        "\nTakeaway: independent tool calls in a turn should run concurrently, so the turn\n"
        "then costs the slowest call, not the sum. And the final answer is an ordinary\n"
        "completion, so stream it (like the API dives) for instant, responsive output."
    )
