#!/usr/bin/env python3
"""
agent_cli.py — the capstone: a real CLI agent.
==============================================

Everything in the repo, assembled into one tool you can actually use. It has the
full toolbox (calculator, knowledge-base search, save-note), runs the agentic
loop with a step cap, gates the dangerous tool behind your approval, and can show
its work with a step trace. Give it a one-off task, or run it with no task for an
interactive chat that remembers the conversation.

Examples
--------
  # One-off task
  secrun python hands_on/agent_cli.py "What's a year of the Plus plan, and is offline editing included?"

  # Watch every step it takes
  secrun python hands_on/agent_cli.py "What is 19% of 240?" --trace

  # Interactive chat with memory (type 'quit' to exit)
  secrun python hands_on/agent_cli.py

  # Let it save notes without prompting you each time
  secrun python hands_on/agent_cli.py "Save a note titled 'todo' with body 'ship the repo'" --yes

The agent only asks permission for *dangerous* tools (save_note writes a file);
the calculator and search run freely.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

import agent

SYSTEM = (
    "You are a helpful assistant for Nimbus Notes. You have three tools: a "
    "calculator for arithmetic, search_notes for product facts (plans, billing, "
    "security, features), and save_note to store something for the user. Use "
    "search_notes instead of guessing product details, and the calculator instead "
    "of doing math yourself. Be concise."
)


def make_approver(auto_yes: bool, console: Console):
    def approve(call: agent.ToolCall) -> bool:  # type: ignore[attr-defined]
        if auto_yes:
            console.print(f"[yellow][auto-approved][/yellow] {call.name}({call.arguments})")
            return True
        console.print(f"\n[yellow]Approval needed[/yellow]: {call.name}({call.arguments})")
        return input("  Allow this? [y/N] ").strip().lower() in ("y", "yes")

    return approve


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="A CLI agent: tools + loop + approval + trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", nargs="?", help="A task to run. Omit for interactive chat mode.")
    p.add_argument("--trace", action="store_true", help="Show each tool call the agent makes.")
    p.add_argument("--max-steps", type=int, default=6, help="Max tool-calling steps (default 6).")
    p.add_argument("--yes", action="store_true", help="Auto-approve dangerous tools (no prompt).")
    return p.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
    load_dotenv()
    agent.ensure_ready()  # type: ignore[attr-defined]

    console = Console()
    console.print(f"[dim]Provider: {agent.describe()}  |  max_steps={args.max_steps}[/dim]")  # type: ignore[attr-defined]

    tools = agent.default_tools()  # type: ignore[attr-defined]
    approve = make_approver(args.yes, console)
    tracer = agent.Tracer(enabled=args.trace)  # type: ignore[attr-defined]

    # One-off mode.
    if args.task:
        if args.trace:
            console.print("\n[dim]Trace:[/dim]")
        result = agent.run_agent(  # type: ignore[attr-defined]
            SYSTEM, args.task, tools, max_steps=args.max_steps, approve=approve, tracer=tracer
        )
        console.print()
        console.print(Markdown(result.answer))
        if result.stopped_early:
            console.print("[red](stopped early — hit the step limit)[/red]")
        return 0

    # Interactive mode with memory.
    console.print("Interactive agent — it remembers the conversation. Type 'quit' to exit.\n")
    history: list = []
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
        if args.trace:
            console.print("[dim]Trace:[/dim]")
        result = agent.run_agent(  # type: ignore[attr-defined]
            SYSTEM, user_input, tools,
            max_steps=args.max_steps, approve=approve, tracer=tracer, history=history,
        )
        console.print()
        console.print(Markdown(result.answer))
        console.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
