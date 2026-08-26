"""
Example 12: planning & reflection: think before, check after.

The bare loop (example 03) reacts step by step. Two cheap additions make it far
more reliable on hard, multi-part tasks, and both are just extra LLM passes around
the loop:

  PLAN (before): ask the model to break the task into a short, explicit plan first.
    A written plan keeps a long task on track and stops the agent from forgetting a
    sub-goal halfway through. You then run the loop *against* that plan.

  REFLECT (after): once the agent produces an answer, run a critic pass: "does this
    fully and correctly answer the question? if not, what's missing?" Then let it
    revise. Catches the half-answers and arithmetic slips a single pass misses.

This script plans a multi-part task, executes it with the tool-using loop, then
reflects and (if needed) revises. (The critic here is the model judging itself,
useful but fallible; the strongest version checks against a real verifier or tests,
see the prompt-engineering 'reflexion' lesson and the evals dive.)

Run it:

    secrun python examples/12_planning_reflection.py
    secrun python examples/12_planning_reflection.py "Compare the Free and Plus plans, and compute the yearly cost of Plus."
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

TOOLS = [agent.SEARCH_NOTES, agent.CALCULATOR]


def llm(system: str, user: str) -> str:
    turn = providers.run_turn(system, [providers.user_message(user)], [])
    return (turn.text or "").strip()


def plan(task: str) -> str:
    return llm(
        "You break a task into a short numbered plan (2-4 steps). Output only the plan.",
        f"Task: {task}",
    )


def reflect(task: str, answer: str) -> tuple[bool, str]:
    """Critic pass. Returns (looks_good, critique)."""
    verdict = llm(
        "You are a strict reviewer. Given a QUESTION and an ANSWER, reply with 'OK' if the "
        "answer is complete and correct, or 'REVISE: <what is wrong or missing>' otherwise. "
        "Be terse.",
        f"QUESTION:\n{task}\n\nANSWER:\n{answer}",
    )
    return verdict.upper().startswith("OK"), verdict


if __name__ == "__main__":
    task = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What does the Plus plan cost per year, and how many notes does the Free plan allow?"
    )
    print(f"Task: {task}\n")

    # 1) PLAN
    the_plan = plan(task)
    print("Plan:\n" + the_plan + "\n")

    # 2) EXECUTE against the plan, using the tool loop.
    system = (
        "You are a careful assistant. Follow the given plan. Use search_notes for "
        "product facts and the calculator for arithmetic. Never guess at numbers.\n\n"
        f"PLAN:\n{the_plan}"
    )
    print("Execution trace:")
    result = agent.run_agent(system, task, TOOLS, tracer=agent.Tracer())
    answer = result.answer
    print(f"\nDraft answer: {answer}\n")

    # 3) REFLECT, and revise once if the critic objects.
    good, critique = reflect(task, answer)
    print(f"Reflection: {critique}")
    if not good:
        print("\nRevising based on the critique...\n")
        revise_system = (
            system
            + f"\n\nA reviewer noted: {critique}\nProduce a corrected, complete answer."
        )
        answer = agent.run_agent(
            revise_system, task, TOOLS, tracer=agent.Tracer()
        ).answer

    print(f"\nFinal answer: {answer}")
    print(
        "\nTakeaway: a quick PLAN keeps multi-step tasks on the rails, and a REFLECT pass\n"
        "catches half-answers before the user sees them. Both are small wrappers around\n"
        "the same loop. For trustworthy reflection, ground the critic in a real check\n"
        "(tests, a schema, a verifier) rather than the model's own opinion."
    )
