"""
Example 11: workflows vs. agents: use the simplest thing that works.

"Agent" is fashionable, but a loop where the model drives is not always what you
want. There's a spectrum:

  WORKFLOW: YOU orchestrate fixed steps in code (classify -> route -> handle). The
             path is known, so you hard-code it. Predictable, cheap, easy to test,
             easy to debug. Most "AI features" are really workflows.

  AGENT:    the MODEL drives an open-ended loop, choosing tools and steps itself
             (examples 03-10). Flexible and powerful, but less predictable, pricier,
             and harder to test. Reach for it when the path *can't* be known up front.

This example does the SAME customer-support task both ways:
  - a routing WORKFLOW: one call classifies the message, then code dispatches to a
    category-specific prompt. The control flow lives in Python.
  - the AGENT loop: hand the model tools and let it decide.

Both work here, which is exactly the point. The workflow is the right default;
the agent earns its keep only when the task is genuinely open-ended.

Run it:

    secrun python examples/11_workflows_vs_agents.py
    secrun python examples/11_workflows_vs_agents.py "my invoice looks wrong and the export button is broken"
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


def llm(system: str, user: str) -> str:
    """A plain, tool-free LLM call (one turn): the building block of a workflow."""
    turn = providers.run_turn(system, [providers.user_message(user)], [])
    return (turn.text or "").strip()


# --- THE WORKFLOW: classify -> route -> handle, orchestrated in code. ---------
HANDLERS = {
    "billing": "You are a billing specialist. Be precise about charges and refunds. One short paragraph.",
    "technical": "You are a technical support engineer. Give clear troubleshooting steps. One short paragraph.",
    "general": "You are a friendly support generalist. Answer helpfully in one short paragraph.",
}


def workflow(message: str) -> str:
    # Step 1: classify (a constrained call). YOU decide what happens next.
    category = llm(
        "Classify the support message into exactly one word: billing, technical, or general.",
        message,
    ).lower()
    category = next((c for c in HANDLERS if c in category), "general")
    print(f"  [workflow] routed to: {category}")
    # Step 2: dispatch to the matching handler prompt.
    return llm(HANDLERS[category], message)


# --- THE AGENT: hand it tools, let it choose. --------------------------------
def run_as_agent(message: str) -> str:
    system = (
        "You are a support assistant. Use search_notes for product facts and "
        "the calculator for any arithmetic; otherwise just answer."
    )
    result = agent.run_agent(
        system, message, [agent.SEARCH_NOTES, agent.CALCULATOR], tracer=agent.Tracer()
    )
    return result.answer


if __name__ == "__main__":
    message = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I was charged twice this month, can I get a refund?"
    )
    print(f"Message: {message}\n")

    print("=== WORKFLOW (code controls the steps) ===")
    print(workflow(message))

    print("\n=== AGENT (the model controls the steps) ===")
    print(run_as_agent(message))

    print(
        "\nTakeaway: don't reach for an agent by default. If you can draw the flowchart,\n"
        "build the WORKFLOW; it's cheaper, more predictable, and easier to test. Use an\n"
        "agent when the steps genuinely can't be known in advance. 'Simplest thing that\n"
        "works' applies to architecture, not just prompts."
    )
