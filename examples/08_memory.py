"""
Example 08 — memory: an agent that remembers the conversation.
=============================================================

So far each run started fresh. A useful assistant remembers earlier turns — but
the API is stateless (the same lesson as the sibling repos): "memory" is just YOU
re-sending the growing message list each turn. `run_agent` takes an optional
`history` list and appends to it in place, so passing the same list across calls
gives the agent memory — including the tool calls and results from earlier turns.

This is a small REPL. Try a follow-up that only makes sense with memory:

    you> search the Nimbus Notes plans
    you> which of those is the cheapest paid one?      <- needs the earlier turn

Run it (type 'quit' to exit):

    secrun python examples/08_memory.py
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
    "You are a helpful Nimbus Notes assistant. Use search_notes for product facts "
    "and the calculator for math. Remember earlier turns in the conversation."
)

history: list = []  # the conversation; reused across turns -> memory
print("Chat with the agent. It remembers the conversation. Type 'quit' to exit.\n")

while True:
    try:
        user_input = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    # Same `history` list every turn — that's the memory.
    result = agent.run_agent(
        SYSTEM, user_input, [agent.CALCULATOR, agent.SEARCH_NOTES], history=history
    )
    print(f"agent> {result.answer}\n")

print(
    "The agent 'remembered' only because you passed the same history list back in. "
    "Drop the list (start fresh each turn) and it forgets — memory is a choice you "
    "make by holding onto the conversation, not a server-side feature."
)
