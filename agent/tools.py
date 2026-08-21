"""
agent/tools.py: the toolbox.

A *tool* is the bridge between the model and the world. To the model it's just a
name, a description, and a JSON Schema of inputs. That's all it sees, and all it
needs to decide *when* and *how* to call the tool. To your program it's an
ordinary Python function. The model never runs anything itself; it only *asks*,
and your code chooses whether and how to execute. That gap is where all of an
agent's safety lives.

We ship three tools, chosen to exercise the important ideas:

  - calculator   pure, safe, deterministic. Good for multi-step reasoning.
  - search_notes read-only lookup over a tiny built-in knowledge base (the same
                   "Nimbus Notes" facts from the RAG repo). A real agent would
                   call your RAG pipeline here.
  - save_note    WRITES a file. Marked `dangerous=True`, so the loop can require
                   human approval before it runs (see example 06).

These are deliberately offline and side-effect-free (except save_note, which only
writes inside ./workspace/), so the repo is safe and reproducible.
"""

import ast
import operator
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.join(REPO_ROOT, "workspace")


@dataclass
class Tool:
    """Declare a callable's model-facing schema and application-owned policy.

    ``parameters`` describes only values the model may propose. Identity and
    tenancy belong in ``trusted_context_args``, whose keys are callable argument
    names and whose values name fields on ``ExecutionContext``. An empty role
    set permits every authenticated context; otherwise at least one role must
    match. ``dangerous`` requires explicit approval, while ``mutating`` enables
    replay protection. Time and output limits are enforced by ``ToolExecutor``.
    """

    name: str
    description: str
    parameters: dict
    func: Callable
    dangerous: bool = False
    allowed_roles: frozenset[str] = frozenset()
    trusted_context_args: dict[str, str] = field(default_factory=dict)
    mutating: bool = False
    timeout_seconds: float = 5.0
    maximum_output_bytes: int = 16_384

    def __post_init__(self) -> None:
        """Reject policy values that cannot produce a meaningful contract."""

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.maximum_output_bytes <= 0:
            raise ValueError("maximum_output_bytes must be positive")


# --- calculator: a safe arithmetic evaluator (NOT Python's eval()) ---------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str):
    """Evaluate a basic arithmetic expression. Parses to an AST and walks only
    arithmetic nodes, so (unlike eval) it can't run arbitrary code."""
    return _safe_eval(ast.parse(expression, mode="eval").body)


# --- search_notes: read-only lookup over a tiny knowledge base -------------

_KNOWLEDGE_BASE = {
    "plans": "Nimbus Notes has three plans: Free, Plus ($4/month), and Team ($10/user/month).",
    "trash": "Deleted notes are kept in Trash for 30 days before being permanently removed.",
    "data": "All Nimbus Notes customer data is stored in data centers in Frankfurt, Germany.",
    "refunds": "Annual subscriptions are refundable in full within 14 days of purchase.",
    "twofactor": "Enable two-factor authentication under Settings -> Security.",
    "export": "Any notebook can be exported to Markdown, PDF, or HTML.",
    "offline": "Offline editing is available on the Plus and Team plans, not Free.",
}


def search_notes(query: str) -> str:
    """Return the most relevant knowledge-base entries for a query (keyword
    overlap, a stand-in for a real RAG retrieval call)."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for key, text in _KNOWLEDGE_BASE.items():
        overlap = len(q & set(re.findall(r"[a-z0-9]+", text.lower())))
        if overlap:
            scored.append((overlap, text))
    scored.sort(key=lambda p: p[0], reverse=True)
    if not scored:
        return "No matching notes found."
    return "\n".join(f"- {text}" for _, text in scored[:2])


# --- save_note: the side-effecting (dangerous) tool ------------------------


def save_note(title: str, body: str) -> str:
    """Write a note into ./workspace/. This has a side effect on disk, which is
    why the tool is marked dangerous and the loop can gate it behind approval."""
    os.makedirs(WORKSPACE, exist_ok=True)
    safe = re.sub(r"[^a-z0-9_-]+", "_", title.lower()).strip("_") or "note"
    path = os.path.join(WORKSPACE, safe + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{body}\n")
    return f"Saved note to workspace/{safe}.md"


CALCULATOR = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression like '12 * (3 + 4)'. Use this for any math instead of computing it yourself.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "description": "e.g. '2 + 2 * 10'",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    func=calculator,
)

SEARCH_NOTES = Tool(
    name="search_notes",
    description="Search the Nimbus Notes help knowledge base. Use this to answer questions about the product (plans, billing, security, etc.).",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "description": "What to look up",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    func=search_notes,
)

SAVE_NOTE = Tool(
    name="save_note",
    description="Save a note to the user's workspace. Writes a file to disk.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 120},
            "body": {"type": "string", "maxLength": 10_000},
        },
        "required": ["title", "body"],
        "additionalProperties": False,
    },
    func=save_note,
    dangerous=True,
    mutating=True,
)


def default_tools() -> list[Tool]:
    """The standard toolbox used by most examples and the capstone."""
    return [CALCULATOR, SEARCH_NOTES, SAVE_NOTE]
