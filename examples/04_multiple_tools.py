"""
Example 04: multiple tools: the model chooses.

Give the agent more than one tool and a new skill appears: it picks the *right*
tool for each step, and chains them. Here it has `search_notes` (product facts)
and `calculator` (math). A question like "what does the Plus plan cost per year?"
needs both (look up the monthly price, then multiply by 12) and the model
sequences them on its own.

The tool *descriptions* are what let it choose well, which is why they're written
to say when each applies. Watch the trace: search first, then calculator.

Run it:

    secrun python examples/04_multiple_tools.py
    secrun python examples/04_multiple_tools.py "Is offline editing on Free, and what's Team's yearly price per user?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")

SYSTEM = (
    "You are a Nimbus Notes assistant. Use search_notes for any product facts "
    "(plans, prices, features) instead of guessing, and the calculator for any "
    "arithmetic."
)

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "How much does the Plus plan cost for a full year?"
)
print(f"Question: {question}\n")
print("Trace:")

result = agent.run_agent(
    SYSTEM, question, [agent.CALCULATOR, agent.SEARCH_NOTES], tracer=agent.Tracer()
)

print(f"\nFinal answer: {result.answer}")
tools_used = ", ".join(sorted({s.tool for s in result.steps})) or "none"
print(f"(tools used: {tools_used})")

print(
    "\nThe agent routed each sub-task to the right tool with no hard-coded plan. "
    "that routing is the model's job, and good tool descriptions are how you make "
    "it reliable."
)
