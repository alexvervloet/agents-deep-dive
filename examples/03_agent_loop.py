"""
Example 03: the agent loop. This is the whole idea.

Example 02 did one turn by hand. An agent just repeats it: run the tool, feed the
result back, ask again, until the model stops asking and gives a final answer.
That loop is `run_agent` (see agent/loop.py). Its control flow is small enough
to hold in your head; the checks that make a tool call safe to run live behind
one call to the contract executor from section 7A, deliberately kept out of it.

We give it a question that needs several steps of arithmetic, turn on the Tracer
so you can watch each tool call, and print the final answer. Notice the model
chains calls: it uses one result to decide the next.

Run it:

    secrun python examples/03_agent_loop.py
    secrun python examples/03_agent_loop.py "What is 15% of 240, plus 7 squared?"

Then run the SAME question against a different model and watch the trace change:

    PROVIDER=openai secrun python examples/03_agent_loop.py
    PROVIDER=claude secrun python examples/03_agent_loop.py

The loop, the tools, and the question are identical, but the number of tool
calls is not. One model may break `(23 * 47) + (88 / 4)` into four separate
calls (one per operation), while another folds several operations into one call
or reasons about part of it directly. Same architecture, different "planning."
That variation is the point: how a task gets decomposed is a property of the
model, not of your loop, worth remembering when you're picking a model or
debugging why an agent takes more steps than you expected.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")

SYSTEM = "You are a careful assistant. Use the calculator for any arithmetic; don't compute in your head."

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What is (23 * 47) + (88 / 4), and then that total multiplied by 3?"
)
print(f"Question: {question}\n")
print("Trace:")

result = agent.run_agent(SYSTEM, question, [agent.CALCULATOR], tracer=agent.Tracer())

print(f"\nFinal answer: {result.answer}")
print(f"({len(result.steps)} tool call(s) along the way)")
print("Re-run with the other PROVIDER on the same question; the count often changes.")

print(
    "\nThat's an agent: a loop around tool-calling. Everything from here (more "
    "tools, error recovery, approval, memory, sub-agents) is a small addition to "
    "this loop, not a new concept."
)
