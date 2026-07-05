"""
Example 02 — one tool call (no loop yet).
=========================================

Here's the single most important mechanic, in isolation: you give the model some
tools and a question; instead of answering, it replies "please run calculator
with expression='23 * 47'." That's a *request* — the model can't run anything
itself. You run it.

This example does exactly one turn so you can see the shape of that request
clearly. It does NOT feed the result back yet — that's the loop, and it's the
next example. (We call the provider shim directly here to expose the raw turn;
normally you'd just use run_agent.)

Run it:

    secrun python examples/02_one_tool_call.py
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

tools = [agent.CALCULATOR]
schema = providers.to_tool_schema(tools)
history = [providers.user_message("What is 23 * 47?")]

# One assistant turn. The shim normalizes both providers to the same Turn shape.
turn = providers.run_turn(
    "You are a helpful assistant. Use tools for arithmetic.", history, schema
)

if not turn.tool_calls:
    print(f"The model answered directly (no tool call): {turn.text}")
else:
    call = turn.tool_calls[0]
    print("The model did NOT answer. It requested a tool call:")
    print(f"  tool: {call.name}")
    print(f"  args: {call.arguments}")
    print(f"  id:   {call.id}")

    # WE run it — the model only asked.
    result = agent.CALCULATOR.func(**call.arguments)
    print(f"\nWe run calculator{tuple(call.arguments.values())} -> {result}")

print(
    "\nThat's one turn: question -> tool *request* -> you execute it. To turn the "
    "result into a final answer, you append it and ask again — and if you keep "
    "doing that until the model stops asking, you have an agent. That's example 03."
)
