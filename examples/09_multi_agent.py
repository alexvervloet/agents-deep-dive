"""
Example 09: multi-agent: an agent that delegates to another agent.

As tasks grow, one agent with twenty tools gets unfocused. The fix is the same one
humans use: delegate. A "sub-agent" is not a new mechanism. It's just a tool
whose function happens to run its own agent loop, with its own system prompt and
its own (smaller) toolset.

Here an *orchestrator* coordinates two specialists:

  - a research sub-agent (system: "find facts", tools: [search_notes])
  - the calculator, used directly

The orchestrator calls `research` as if it were any other tool; behind that tool,
a whole second loop runs. Each agent stays focused on its own job.

Run it:

    secrun python examples/09_multi_agent.py
    secrun python examples/09_multi_agent.py "What does a year of Team cost for 3 users?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")


def research(question: str) -> str:
    """A sub-agent: its own loop, its own prompt, its own tools. Returns an answer."""
    sub = agent.run_agent(
        "You are a research specialist. Use search_notes to find product facts and "
        "answer in one concise sentence.",
        question,
        [agent.SEARCH_NOTES],
        max_steps=4,
    )
    return sub.answer


# Wrap the sub-agent as a tool the orchestrator can call.
RESEARCH_TOOL = agent.Tool(
    name="research",
    description="Delegate a factual question about Nimbus Notes to a research specialist. Returns a concise answer.",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The factual question to research",
            }
        },
        "required": ["question"],
    },
    func=research,
)

SYSTEM = (
    "You are an orchestrator. For factual questions about the product, delegate to "
    "the 'research' tool. For arithmetic, use the 'calculator'. Combine their "
    "results into a final answer."
)

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "How much does the Plus plan cost for a full year? Research the price, then compute it."
)
print(f"Question: {question}\n")
print("Orchestrator trace (the research tool runs its own loop inside):")

result = agent.run_agent(
    SYSTEM, question, [agent.CALCULATOR, RESEARCH_TOOL], tracer=agent.Tracer()
)

print(f"\nFinal answer: {result.answer}")

print(
    "\nNotice 'research' is just a tool to the orchestrator, but it's an agent "
    "underneath. That's how big agent systems are built: not one giant agent, but "
    "focused agents that call each other through the same tool interface."
)
