"""
Example 05 — control: step limits and error recovery.
=====================================================

An unsupervised loop around a model needs guardrails. Two essential ones, both
built into run_agent:

  1. A STEP LIMIT (max_steps). A confused agent can loop forever — calling tools,
     never deciding it's done. A hard ceiling turns "runs up a huge bill" into
     "stops and tells you it didn't finish."

  2. ERROR RECOVERY. When a tool raises, we don't crash — we hand the error text
     back to the model *as the tool result*. The model reads it and adapts (tries
     different inputs, or explains the problem) instead of the whole program dying.

This example shows both. (Division by zero makes the calculator raise; a tiny
max_steps cuts a multi-step task short.)

Run it:

    secrun python examples/05_limits_and_errors.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")

SYSTEM = "You are a careful assistant. Use the calculator for arithmetic."

# 1. Error recovery: the calculator raises on 10/0; the error goes back to the
#    model, which then explains rather than crashing.
print("=== error recovery (10 / 0) ===")
r1 = agent.run_agent(
    SYSTEM, "What is 10 divided by 0?", [agent.CALCULATOR], tracer=agent.Tracer()
)
print(f"Final answer: {r1.answer}")
errored = any(s.result.startswith("Error") for s in r1.steps)
print(
    f"(a tool returned an error mid-run: {errored} — the agent handled it, didn't crash)"
)

# 2. Step limit: a task that needs several calls, capped at 1 step.
print("\n=== step limit (max_steps=1 on a multi-step task) ===")
r2 = agent.run_agent(
    SYSTEM,
    "Compute (5*5) + (6*6) + (7*7), then multiply the total by 10.",
    [agent.CALCULATOR],
    max_steps=1,
    tracer=agent.Tracer(),
)
print(f"Final answer: {r2.answer}")
print(
    f"(stopped early: {r2.stopped_early} — the ceiling protected you from an unbounded loop)"
)

print(
    "\nThese two knobs — a hard step cap and feeding errors back as results — are "
    "the difference between a toy loop and one you'd let run unattended."
)
