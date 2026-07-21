"""
Example 07: observability: see what the agent did.

An agent makes its own decisions, so when it misbehaves you need to see *why*. The
cure is a trace: a record of every step, which tool, what arguments, what result.
You've seen the live Tracer print steps as they happen; this example also shows
the same information *after the fact*, from `result.steps`, which is what you'd log
and inspect in a real system.

Those step records are also what you'd feed an eval (see the evals-deep-dive repo)
to score an agent's behavior: did it call the right tools, in a sensible order,
without wasted steps?

Run it:

    secrun python examples/07_observability.py
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
    "You are a Nimbus Notes assistant. Use search_notes for product facts and the "
    "calculator for arithmetic."
)
question = (
    "What's the price gap per year between the Plus and Team plans, for one user?"
)
print(f"Question: {question}\n")

print("Live trace:")
result = agent.run_agent(
    SYSTEM, question, [agent.CALCULATOR, agent.SEARCH_NOTES], tracer=agent.Tracer()
)

print(f"\nFinal answer: {result.answer}")

# Post-hoc: the same steps as structured data you could log, render, or evaluate.
print("\nStructured trace (result.steps): what you'd log in production:")
for i, s in enumerate(result.steps, start=1):
    preview = " ".join(s.result.split())[:70]
    print(f"  {i}. {s.tool}({s.arguments})  approved={s.approved}")
    print(f"     -> {preview}")

print(
    "\nNo trace, no debugging: an agent's value and its failures both live in the "
    "sequence of tool calls. Capture them, for humans now and for evals later."
)
