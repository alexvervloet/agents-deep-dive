"""
Example 15: a provider-hosted tool: the loop never sees it.

Every tool in examples 01–14 is *client-executed*: the model asks to call it,
YOUR loop runs the function, and you feed the result back. That round-trip of
tool_use out and tool_result in is the whole mechanic this repo is built on.

A **hosted** (server-side) tool is different in kind. You *declare* it, and the
provider runs it on its own infrastructure, *inside the turn*. You send one
request; you get back one final answer, already grounded in the tool's results.
There is no tool_use/tool_result round-trip for your loop to manage, because
your loop isn't in the middle of it. The clearest example in 2026 is hosted web
search: `web_search` is week-one product work, and it's server-side on both
providers.

This example makes exactly one request with the hosted web-search tool declared,
and prints how many times the *provider* ran search during that turn. The number
is > 0 (it really searched), but the count of client-side tool rounds YOUR code
handled is 0; you never saw a tool call.

Why this matters for the mental model: the agent loop is still the core idea, but
some tools now live on the provider's side of the wire. When you reach for a
hosted tool you trade control (you can't gate, log, or sandbox the call; see
Section 7) for zero plumbing. Client tools and hosted tools coexist in one real
agent; knowing which is which is the point.

This one makes a small, real, billed call (it needs live web access), so it needs
a key. If the hosted tool isn't enabled for your key/model, it says so and exits
cleanly rather than crashing.

Run it:

    secrun python examples/15_hosted_tools.py
    secrun python examples/15_hosted_tools.py "Who won the most recent Formula 1 race?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from dotenv import load_dotenv

load_dotenv()
agent.ensure_ready()
print(f"Provider: {agent.describe()}\n")

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What is a recent, notable news headline from this week? One sentence."
)
print(f"Question: {question}\n")

try:
    result = agent.hosted_web_search(question)
except Exception as e:  # noqa: BLE001, degrade gracefully like the multimodal dive
    print("The hosted web-search tool isn't available on this key/model right now.")
    print(f"  ({type(e).__name__}: {e})\n")
    print(
        "That's fine; the lesson is the shape, not this one call. A hosted tool is\n"
        "declared, not executed by you: the provider runs it inside the turn, so your\n"
        "agent loop never handles a tool_use/tool_result round-trip for it. Client\n"
        "tools (examples 01–14) and hosted tools coexist; you choose per tool whether\n"
        "you need the control of running it yourself, or the zero-plumbing of letting\n"
        "the provider run it."
    )
    sys.exit(0)

print("Answer (grounded in live search the provider ran for you):")
print(f"  {result.text.strip()}\n")

if agent.provider_name() == "openai":
    print(
        "Note: on OpenAI this is the one lesson in the series that uses the Responses\n"
        "API instead of Chat Completions; hosted tools like web_search live there, not\n"
        "in the Chat Completions interface the OpenAI dive teaches. (Claude's hosted\n"
        "tools ride on the same Messages API you already know.)\n"
    )

print(f"Times the PROVIDER ran search this turn: {result.server_tool_calls}")
print("Client-side tool rounds YOUR loop handled:  0")

print(
    "\nThat gap is the whole lesson. Search really happened, but not as a step in\n"
    "your loop. You sent one request and got one final answer; the tool lived on the\n"
    "provider's side of the wire. That's a hosted tool: less control (no gate, no\n"
    "custom logging, no sandbox, so Section 7's approval can't reach it), zero plumbing.\n"
    "Real agents mix both: client tools for actions you must govern, hosted tools\n"
    "(web search, code execution) for capability you're happy to rent."
)
