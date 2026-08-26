"""
Example 01: what a tool is (offline, no API call).

Before any agent, understand the tool. A tool has two faces:

  - to YOUR code, it's a plain Python function (here: calculator, search_notes).
  - to the MODEL, it's just a name, a description, and a JSON Schema of inputs;
    the model never sees your function body, only this "menu entry," and uses it
    to decide when and how to call.

This example calls the tools directly (no model involved) and prints the schema
the model *would* see. It's completely offline and free, the foundation the rest
of the repo builds the loop on top of.

Run it:

    python examples/01_tools.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from agent import providers
from agent.tools import calculator, search_notes

# 1. Tools are just functions. Call them yourself: no model, no key.
print("Calling the tools directly (this is what the model can ask you to run):")
print(f"  calculator('12 * (3 + 4)')              -> {calculator('12 * (3 + 4)')}")
print(
    f"  search_notes('how long are deleted notes kept') ->\n      {search_notes('how long are deleted notes kept')!r}"
)

# 2. This is all the model sees about a tool: the "menu." It picks from this.
print("\nWhat the model receives (the tool schema it chooses from):")
schema = providers.to_tool_schema([agent.CALCULATOR])
schema2 = providers.to_tool_schema([agent.SEARCH_NOTES])
print(json.dumps(schema[0], indent=2))
print(json.dumps(schema2[0], indent=2))

print(
    "\nThe description and parameter names are the model's only clues for *when* "
    "and *how* to call a tool, so they're prompt engineering, not afterthoughts. "
    "Next (example 02), we hand these to the model and watch it ask to call one."
)
