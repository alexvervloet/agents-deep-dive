"""
Example 16: the tool surface at scale (tool search + programmatic tool calling).

Everything so far assumed a handful of tools and a handful of calls. Both
assumptions break in real agents, and they break in the same place: the context
window.

  MANY TOOLS       Every tool's schema is sent on every request. Thirty tools is
                   thousands of tokens of JSON the model re-reads each turn,
                   most of it irrelevant to the question actually asked.

  MANY CALLS       Every tool RESULT lands in the context too. Ask for the total
                   price of forty SKUs and you get forty round trips and forty
                   result blobs, when what you wanted was one number.

Anthropic ships a feature for each.

TOOL SEARCH: don't send schemas you won't use
    Mark tools `defer_loading: True` and add a search tool. The model searches
    the catalogue and loads only the schemas it needs. Crucially the loaded
    schemas are *appended*, not swapped in, so the cached prefix survives (the
    Context Engineering dive's §10 lesson, applied to tools).

    Never defer everything: the search tool itself must stay loaded, and at
    least one tool must be non-deferred, or the API returns 400.

PROGRAMMATIC TOOL CALLING: don't route results you won't read
    Give a custom tool `allowed_callers: ["code_execution_20260120"]` and Claude
    can call it from *inside* a script running in the code-execution container.
    The tool result returns to the running code, not to the context window. Loop
    over forty SKUs, sum them, and only the total comes back.

    The shape is worth pausing on. Standard tool use is: model asks, you answer,
    model reads. PTC is: model writes a program, the program asks, the program
    reads. Token cost stops scaling with the number of calls and starts scaling
    with the size of the answer.

MODEL GATES
    Tool search runs on Haiku 4.5. PTC does not: it needs Sonnet 4.5 / Opus 4.5
    or newer, so this example uses Sonnet 4.6 for that half and says so.

Anthropic-only, so this one does not honour PROVIDER. Run:

    secrun python examples/16_tool_search_and_ptc.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit(
        "This example is Anthropic-specific (tool search and PTC are Claude features).\n"
        "Set ANTHROPIC_API_KEY via secrun (see ../SECRETS.md) and try again."
    )

import anthropic  # noqa: E402

client = anthropic.Anthropic()

SEARCH_MODEL = "claude-haiku-4-5"
PTC_MODEL = "claude-sonnet-4-6"  # PTC needs 4.5+; Haiku 4.5 returns a 400

# A catalogue big enough to be annoying if you sent all of it every turn. In a
# real agent this is where your thirty CRM/billing/calendar tools would live.
CATALOGUE = [
    {
        "name": name,
        "description": desc,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "defer_loading": True,  # not sent until the model searches for it
    }
    for name, desc in [
        ("search_invoices", "Find invoices by customer, date or amount."),
        ("search_contacts", "Find people in the CRM by name or company."),
        ("search_calendar", "Find meetings and availability."),
        ("search_tickets", "Find support tickets by status or reporter."),
    ]
]

# --- 1. Tool search ----------------------------------------------------------
print("--- 1. tool search: four deferred tools, one loaded ---")
searched = client.messages.create(
    model=SEARCH_MODEL,
    max_tokens=1024,
    tools=[
        # The search tool itself is NOT deferred. It cannot be.
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        *CATALOGUE,
    ],
    messages=[{"role": "user", "content": "Find me the support tickets that are still open."}],
)
print(f"stop_reason: {searched.stop_reason}")
for block in searched.content:
    if block.type == "server_tool_use":
        print(f"  searched the catalogue: {block.name}")
    elif block.type == "tool_search_tool_result":
        print("  catalogue returned matching schemas")
    elif block.type == "tool_use":
        print(f"  then called: {block.name}({block.input})")
    elif block.type == "text" and block.text.strip():
        print(f"  said: {block.text.strip()[:120]}")
print(
    "\nThe four schemas were available but only the relevant one was loaded.\n"
    "With four tools that saves little; with forty it is the difference between\n"
    "a lean prompt and one that is mostly JSON the model has to skim past."
)

# --- 2. Programmatic tool calling -------------------------------------------
# `get_price` is an ordinary custom tool with one extra field: allowed_callers.
# That field is what lets Claude invoke it from generated code.
print(f"\n--- 2. programmatic tool calling on {PTC_MODEL} ---")
ptc = client.messages.create(
    model=PTC_MODEL,
    max_tokens=4096,
    tools=[
        {"type": "code_execution_20260120", "name": "code_execution"},
        {
            "name": "get_price",
            "description": "Return the unit price in USD for one SKU.",
            "input_schema": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
            "allowed_callers": ["code_execution_20260120"],  # the PTC opt-in
        },
    ],
    messages=[
        {
            "role": "user",
            "content": (
                "The SKUs are A-1, A-2, B-7, C-3 and C-9. Get the price of each "
                "and tell me the total. Write a loop rather than asking five times."
            ),
        }
    ],
)
print(f"stop_reason: {ptc.stop_reason}")
print("blocks:", [b.type for b in ptc.content])
for block in ptc.content:
    if block.type == "tool_use":
        print(f"  the SCRIPT is asking us for: {block.name}({block.input})")
print(
    "\nA `tool_use` here is a request from the running program, not from the\n"
    "model's turn. You answer it the same way, but note what did NOT happen:\n"
    "five separate round trips, each dropping a price into the context window."
)
