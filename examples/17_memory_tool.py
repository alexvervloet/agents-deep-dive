"""
Example 17: the memory tool: remembering across sessions, not just turns.

Section 9 gave the agent memory by re-sending the message list. That is real
memory and it is the right default, but it has a hard edge: it dies with the
process. Close the REPL and the agent has never met you.

The **memory tool** is the other kind. Claude gets a `/memories` directory it can
read and write through tool calls, and because YOU implement the storage, what
it writes outlives the conversation, the process, and the machine if you want it
to.

  §9 conversation memory   in the message list; dies with the run; free
  memory tool              in your storage; survives everything; you own it

IT IS A CLIENT-SIDE TOOL, WHICH IS THE WHOLE POINT
    Declaring `{"type": "memory_20250818", "name": "memory"}` does not give you
    storage. It tells Claude the tool exists and what commands it takes; the
    backend is yours to write. Claude sends `view` / `create` / `str_replace` /
    `insert` / `delete` / `rename` commands and you decide what they mean: a
    directory, a database, a per-user bucket, whatever fits.

    That is a feature, not a chore. Memory that persists is memory somebody has
    to govern, and the API declining to guess is what lets you scope it per user
    and delete it on request.

THE SECURITY BIT, WHICH IS NOT OPTIONAL
    Every path in a memory command is model-generated text. Resolve it and check
    it is still inside your memory root before you touch the filesystem. The
    guard below is six lines and it is the difference between a memory directory
    and an arbitrary file write. Do not skip it because the model seems friendly:
    the agent reads untrusted content (see the Prompt Injection dive), and
    anything it reads can propose a path.

    And do not put secrets in memory. It is replayed verbatim into future
    contexts, so a key written once leaks into every later session.

Anthropic-only. Run it twice to see the point: the second run starts knowing
what the first one wrote.

    secrun python examples/17_memory_tool.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit(
        "This example is Anthropic-specific (the memory tool is a Claude feature).\n"
        "Set ANTHROPIC_API_KEY via secrun (see ../SECRETS.md) and try again."
    )

import anthropic  # noqa: E402

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5"

# The memory root. Real systems scope this per user; a shared directory is a
# shared brain, which is rarely what you want.
ROOT = Path(__file__).resolve().parent.parent / "workspace" / "memories"
ROOT.mkdir(parents=True, exist_ok=True)


def safe_path(raw: str) -> Path:
    """Resolve a model-supplied path and refuse anything outside ROOT.

    This is the guard the docstring insists on. `/memories/notes.md` maps to
    ROOT/notes.md; `/memories/../../.ssh/id_rsa` raises instead of escaping.
    """
    rel = raw.removeprefix("/memories").lstrip("/")
    candidate = (ROOT / rel).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"path escapes the memory root: {raw!r}")
    return candidate


def handle(command: dict) -> str:
    """Execute one memory command. Returns the text Claude sees as the result."""
    cmd = command.get("command")

    if cmd == "view":
        target = safe_path(command["path"])
        if target.is_dir():
            names = sorted(p.name for p in target.iterdir())
            return "\n".join(names) if names else "(empty)"
        return target.read_text() if target.exists() else "(no such file)"

    if cmd == "create":
        target = safe_path(command["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(command.get("file_text", ""))
        return f"wrote {target.name}"

    if cmd == "str_replace":
        target = safe_path(command["path"])
        body = target.read_text()
        target.write_text(body.replace(command["old_str"], command["new_str"], 1))
        return "replaced"

    if cmd == "insert":
        target = safe_path(command["path"])
        lines = target.read_text().splitlines()
        lines.insert(int(command["insert_line"]), command["insert_text"])
        target.write_text("\n".join(lines))
        return "inserted"

    if cmd == "delete":
        target = safe_path(command["path"])
        if target.is_file():
            target.unlink()
        return "deleted"

    if cmd == "rename":
        safe_path(command["path"]).rename(safe_path(command["new_path"]))
        return "renamed"

    return f"unsupported command: {cmd}"


def run(user_text: str) -> None:
    """One turn, with a loop so Claude can use the tool more than once."""
    messages: list[dict] = [{"role": "user", "content": user_text}]
    tools = [{"type": "memory_20250818", "name": "memory"}]

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1024, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            print(f"  claude: {text.strip()}")
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                out = handle(dict(block.input))
                print(f"  [memory {block.input.get('command')}: {block.input.get('path','')}] -> {out[:60]}")
            except Exception as e:  # noqa: BLE001
                out, ok = f"error: {e}", False
                print(f"  [memory REFUSED] {e}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})


existing = sorted(p.name for p in ROOT.iterdir() if p.is_file())
print(f"memory root: {ROOT}")
print(f"files already there: {existing or '(none, this is a first run)'}\n")

if not existing:
    print("--- first run: teaching it something ---")
    run("Remember that I prefer terse answers and that I work in Python. Save it to memory.")
    print("\nRun this example again. The next run starts with that on disk.")
else:
    print("--- second run: it reads what the first run wrote ---")
    run("Check your memory. What do you know about my preferences?")
    print(
        "\nNothing was re-sent in the prompt. The agent went and looked, which is\n"
        "the difference between a conversation that remembers and an agent that does."
    )
