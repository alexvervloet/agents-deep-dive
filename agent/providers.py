"""
agent/providers.py: the ONLY provider-specific file.

Agents are an architecture, not a provider feature: the loop, the tools, the
control logic are all provider-agnostic. The one thing that genuinely differs
between providers is the *shape* of a tool-calling turn: how you describe tools,
how the model hands back a tool request, and how you send a result. This file
normalizes all of that to a tiny neutral interface the rest of the repo uses:

  to_tool_schema(tools)            -> the provider's tool format
  run_turn(system, history, tools) -> a normalized Turn (text and/or tool calls)
  format_tool_results(results)     -> provider-native messages to append
  user_message(text)               -> a provider-native user message

The two providers differ in concrete ways the sibling repos already taught:
OpenAI uses `tools=[{type:function,...}]`, returns `message.tool_calls` (arguments
as a JSON *string*), and takes each result back as a `role:"tool"` message. Claude
uses `tools=[{name,input_schema,...}]`, returns `tool_use` content blocks (input
already a dict), and takes results back as `tool_result` blocks inside ONE user
message. The normalization below hides those differences so the loop reads the
same regardless of stack.
"""

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache

_OPENAI_CHAT = "gpt-5.4-nano"
_CLAUDE_CHAT = "claude-haiku-4-5"
_KEYS = {"openai": ["OPENAI_API_KEY"], "claude": ["ANTHROPIC_API_KEY"]}


@dataclass
class ToolCall:
    """A normalized request from the model to run one tool."""

    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    """One assistant turn, normalized. `tool_calls` empty means the agent is done
    and `text` is the final answer. `raw_assistant` is the provider-native message
    to append to history (the loop treats it as opaque)."""

    text: str | None
    tool_calls: list[ToolCall]
    raw_assistant: object


def provider_name() -> str:
    return os.getenv("PROVIDER", "openai").strip().lower()


def required_keys() -> list[str]:
    return _KEYS.get(provider_name(), [])


def describe() -> str:
    p = provider_name()
    if p == "openai":
        return f"openai  (chat={_OPENAI_CHAT})"
    if p == "claude":
        return f"claude  (chat={_CLAUDE_CHAT})"
    return f"unknown provider {p!r}"


def ensure_ready() -> None:
    p = provider_name()
    if p not in _KEYS:
        sys.exit(f"PROVIDER={p!r} is not recognized. Set PROVIDER=openai or claude in .env.")
    missing = [k for k in required_keys() if not os.getenv(k)]
    if missing:
        sys.exit(
            f"PROVIDER={p} needs {', '.join(missing)} in the environment. "
            f"Provide them via secrun (see SECRETS.md), or run `secrun python check_setup.py`."
        )


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def user_message(text: str) -> dict:
    """A plain user message, the same shape on both providers."""
    return {"role": "user", "content": text}


def to_tool_schema(tools: list) -> list:
    """Convert neutral Tool objects to the active provider's tool format."""
    p = provider_name()
    if p == "openai":
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]
    if p == "claude":
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
    raise ValueError(f"Unknown PROVIDER={p!r}.")


def run_turn(system: str, history: list, tool_schema: list) -> Turn:
    """Run one assistant turn and normalize the result to a Turn."""
    p = provider_name()
    if p == "openai":
        messages = [{"role": "system", "content": system}, *history]
        resp = _openai_client().chat.completions.create(
            model=_OPENAI_CHAT, messages=messages, tools=tool_schema or None  # type: ignore[arg-type]
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            if tc.type != "function":
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        # The SDK message object can be appended back to messages as-is.
        return Turn(text=msg.content, tool_calls=calls, raw_assistant=msg)

    if p == "claude":
        resp = _anthropic_client().messages.create(
            model=_CLAUDE_CHAT, max_tokens=1024, system=system, messages=history, tools=tool_schema
        )
        calls, text_parts = [], []
        for block in resp.content:
            if block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
            elif block.type == "text":
                text_parts.append(block.text)
        return Turn(
            text="".join(text_parts) or None,
            tool_calls=calls,
            raw_assistant={"role": "assistant", "content": resp.content},
        )

    raise ValueError(f"Unknown PROVIDER={p!r}.")


def stream_turn(system: str, history: list, tool_schema: list, on_text=None) -> Turn:
    """Like `run_turn`, but STREAM the assistant's text as it's generated.

    Calls `on_text(piece)` for each text delta so the caller can print tokens live,
    then returns the same normalized `Turn` (text + any tool calls + the message to
    append). This is what lets you stream *inside* the loop, so the user watches the
    agent's words appear even on a turn that ends in a tool call, not just on the
    final answer.

    The provider-specific fiddliness lives here, as always: OpenAI streams tool
    calls as `arguments` fragments you reassemble by index; Claude streams text
    deltas and hands you the finished tool-use blocks in the final message.
    """
    p = provider_name()
    on_text = on_text or (lambda _piece: None)

    if p == "openai":
        messages = [{"role": "system", "content": system}, *history]
        stream = _openai_client().chat.completions.create(  # type: ignore[call-overload]
            model=_OPENAI_CHAT, messages=messages, tools=tool_schema or None, stream=True  # type: ignore[arg-type]
        )
        text_parts: list[str] = []
        acc: dict[int, dict] = {}  # tool calls arrive in fragments, keyed by index
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                on_text(delta.content)
            for tcd in delta.tool_calls or []:
                slot = acc.setdefault(tcd.index, {"id": "", "name": "", "args": ""})
                if tcd.id:
                    slot["id"] = tcd.id
                if tcd.function and tcd.function.name:
                    slot["name"] += tcd.function.name
                if tcd.function and tcd.function.arguments:
                    slot["args"] += tcd.function.arguments
        calls, raw_tool_calls = [], []
        for idx in sorted(acc):
            slot = acc[idx]
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))
            raw_tool_calls.append({
                "id": slot["id"], "type": "function",
                "function": {"name": slot["name"], "arguments": slot["args"] or "{}"},
            })
        text = "".join(text_parts)
        raw_assistant: dict = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            raw_assistant["tool_calls"] = raw_tool_calls
        return Turn(text=text or None, tool_calls=calls, raw_assistant=raw_assistant)

    if p == "claude":
        with _anthropic_client().messages.stream(
            model=_CLAUDE_CHAT, max_tokens=1024, system=system, messages=history, tools=tool_schema
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    on_text(event.delta.text)
            final = stream.get_final_message()
        calls, text_parts = [], []
        for block in final.content:
            if block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
            elif block.type == "text":
                text_parts.append(block.text)
        return Turn(
            text="".join(text_parts) or None,
            tool_calls=calls,
            raw_assistant={"role": "assistant", "content": final.content},
        )

    raise ValueError(f"Unknown PROVIDER={p!r}.")


@dataclass
class HostedResult:
    """The outcome of a provider-hosted tool call (see hosted_web_search).

    `text` is the final answer. `server_tool_calls` is how many times the
    *provider* ran the tool inside the turn: proof that the work happened, even
    though your code never saw a tool request or sent a result back."""

    text: str
    server_tool_calls: int


def hosted_web_search(query: str) -> HostedResult:
    """Ask the model a question with the provider's SERVER-SIDE web search tool.

    Every tool in examples 01–14 is *client-executed*: the model asks, your loop
    runs the function, you feed the result back. A hosted (server-side) tool is
    different in kind: you declare it and the provider runs it *inside the turn*,
    on its own infrastructure. One request goes out; one final answer comes back,
    already grounded in live search results. There is no tool_use/tool_result
    round-trip for your loop to manage, because the loop isn't in the middle.

    That changes the mental model this repo taught: the agent loop is still the
    core, but some tools now live on the provider's side of the wire. We return
    how many times the provider ran search so you can see it happened.

    Raises on any failure so the example can degrade gracefully (the hosted tool
    may not be enabled for your key/model)."""
    p = provider_name()
    if p == "openai":
        # The Responses API runs `web_search` server-side and returns the final
        # text directly. Search steps appear as `web_search_call` items in the
        # output (we count them) but you never handle a tool result yourself.
        resp = _openai_client().responses.create(
            model=_OPENAI_CHAT, tools=[{"type": "web_search"}], input=query
        )
        calls = sum(1 for item in resp.output if getattr(item, "type", "") == "web_search_call")
        return HostedResult(text=resp.output_text, server_tool_calls=calls)

    if p == "claude":
        # Declaring the web_search tool lets Claude run it during the turn. The
        # response interleaves `server_tool_use` / `web_search_tool_result` blocks
        # with the text; again, no round-trip for your code.
        resp = _anthropic_client().messages.create(
            model=_CLAUDE_CHAT,
            max_tokens=1024,
            messages=[{"role": "user", "content": query}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
        calls = sum(1 for b in resp.content if b.type == "server_tool_use")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return HostedResult(text=text, server_tool_calls=calls)

    raise ValueError(f"Unknown PROVIDER={p!r}.")


def format_tool_results(results: list[tuple[str, str]]) -> list:
    """Turn (tool_call_id, result_text) pairs into provider-native messages.

    OpenAI wants one `role:"tool"` message per result; Claude wants a SINGLE user
    message containing all the `tool_result` blocks. Getting this wrong (e.g.
    splitting Claude's results across messages) is a classic agent bug, so it
    lives in exactly one place."""
    p = provider_name()
    if p == "openai":
        return [{"role": "tool", "tool_call_id": cid, "content": content} for cid, content in results]
    if p == "claude":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": cid, "content": content}
                    for cid, content in results
                ],
            }
        ]
    raise ValueError(f"Unknown PROVIDER={p!r}.")
