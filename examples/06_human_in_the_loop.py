"""
Example 06: human-in-the-loop approval.

Some tools have consequences: sending email, deleting data, spending money,
writing files. You don't want the model to trigger those unsupervised. The fix:
mark a tool `dangerous=True` and pass run_agent an `approve` callback. Before
running a dangerous tool, the loop asks your callback for permission; a denial
comes back to the model as an ordinary result, and it adapts.

Here, `save_note` writes a file, so it's dangerous. `calculator` and
`search_notes` aren't, so they run freely. You're only prompted for the action
that actually matters.

This example is interactive: it asks YOU to approve at the terminal.

Run it:

    secrun python examples/06_human_in_the_loop.py
    secrun python examples/06_human_in_the_loop.py "Save a note titled 'Ideas' with body 'try the new editor'"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")


def approve(call: agent.ToolCall) -> bool:
    """Ask the human before a dangerous tool runs. Return True to allow."""
    print("\n  [approval needed] the agent wants to run:")
    print(f"      {call.name}({call.arguments})")
    answer = input("  Allow this? [y/N] ").strip().lower()
    return answer in ("y", "yes")


SYSTEM = (
    "You are an assistant that can save notes for the user. When the user asks you "
    "to remember or save something, use the save_note tool."
)

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Please save a note titled 'Shopping list' with body 'milk, eggs, bread'."
)
print(f"Question: {question}")

result = agent.run_agent(
    SYSTEM, question, agent.default_tools(), approve=approve, tracer=agent.Tracer()
)

print(f"\nFinal answer: {result.answer}")
denied = any(not s.approved for s in result.steps)
if denied:
    print(
        "(you denied a tool call; notice the agent acknowledged it instead of forcing it)"
    )

print(
    "\nApproval gating is how you keep a human in control of the actions that "
    "matter while letting the agent run freely on the safe ones. Which tools are "
    "'dangerous' is your policy decision, declared on the tool."
)
