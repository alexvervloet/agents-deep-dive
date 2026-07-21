# Exercises: make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts.

How to use it: work the section first, then come back. **Commit to an answer
before you run or reveal.** The prediction is where the learning happens. Answers
are hidden behind ▸ toggles.

> Examples 01 and 10 are **(offline)**: no API call, no cost. The rest make
> small, cheap calls.

---

## Section 2: What a tool is **(offline)**

**Recall.** When the model "uses a tool," does it run your function? What does it
actually do?

<details><summary>▸ Answer</summary>

No: the model never runs anything. It emits a *request*: a tool name plus
arguments. Your code decides whether and how to execute it. That gap is where all
of an agent's safety lives (approval, sandboxing, validation).
</details>

**Do (offline).** In `examples/01_tools.py`, the model only ever sees a tool's
name, description, and parameter schema, not the function body. Why does that
make the description a piece of prompt engineering?

<details><summary>▸ Answer</summary>

Because the description is the model's *only* basis for deciding when and how to
call the tool. A vague description ("does stuff") leads to misuse; a precise one
("use this for arithmetic; expression like '2+2'") leads to correct calls. You're
programming the model's behavior through that text.
</details>

---

## Section 3: One tool call

**Predict, then run.** In `examples/02_one_tool_call.py`, you ask "What is 23 *
47?" with the calculator available. Will the model reply with the number, or with
something else?

<details><summary>▸ Answer</summary>

With a *tool call request* (calculator, expression="23 * 47"), not the number. It
defers the math to the tool. You then run it, and to turn that result into a
final answer you'd feed it back and ask again, which is the loop.
</details>

---

## Section 4: The agent loop

**Recall.** Describe the agent loop in one sentence. What makes it stop?

<details><summary>▸ Answer</summary>

Ask the model; if it requests tools, run them, append the results, and ask again;
repeat until it responds with no tool calls; that final, tool-free response is
the answer. (A step cap is the backstop in case it never stops.)
</details>

**Do.** In `examples/03_agent_loop.py`, give it a question needing two or three
calculations and watch the trace. Does it make all the calls up front, or use each
result to decide the next?

<details><summary>▸ Answer</summary>

Usually one (or a few) at a time, using earlier results to inform later steps 
that step-by-step, result-dependent reasoning is exactly what the loop enables and
a single call can't do.
</details>

---

## Section 5: Multiple tools

**Predict.** Give the agent `calculator` and `search_notes` and ask "what does the
Plus plan cost per year?" Which tool runs first, and why?

<details><summary>▸ Answer</summary>

search_notes first (to find the monthly price, a product fact), then calculator
(to multiply by 12). The model routes each sub-task to the tool whose description
fits. Nothing hard-codes that order; it's the model's choice, guided by the tool
descriptions.
</details>

---

## Section 6: Limits & error recovery

**Recall.** Why feed a tool's *error* back to the model instead of crashing? And
what does `max_steps` protect against?

<details><summary>▸ Answer</summary>

Feeding the error back lets the model see what went wrong and adapt (retry with
different inputs, or explain): robustness instead of a dead program. `max_steps`
protects against an infinite loop: a model that keeps calling tools and never
finishes would otherwise run forever (and run up a bill).
</details>

**Do.** Set `max_steps=1` on a task that clearly needs several steps. What does the
agent return, and what does `stopped_early` tell you?

<details><summary>▸ Answer</summary>

It returns a "stopped early" message and `stopped_early=True`, because it hit the
ceiling before reaching a final answer. That flag is how your code knows the result
is incomplete rather than a real answer.
</details>

---

## Section 7: Human-in-the-loop

**Recall.** What makes a tool require approval, and what happens to the agent when
you deny one?

<details><summary>▸ Answer</summary>

You mark it `dangerous=True`; the loop then calls your `approve` callback before
running it. A denial is returned to the model as a normal tool result ("permission
denied"), so the agent acknowledges it and adapts instead of forcing the action.
</details>

---

## Section 8: Observability

**Recall.** Why is a step trace essential for agents specifically, more than for a
single LLM call?

<details><summary>▸ Answer</summary>

Because an agent makes its own multi-step decisions (which tools, in what order,
with what arguments. When it goes wrong, the *why* is in that sequence. Without a
trace you're debugging a black box; with one you can see (and later eval) exactly
what it did.
</details>

---

## Section 9: Memory

**Predict, then run.** In `examples/08_memory.py`, ask it to "search the plans,"
then ask "which is the cheapest paid one?" Does the second question work? Why?

<details><summary>▸ Answer</summary>

It works, because the same `history` list is passed back each turn, so the earlier
search and its results are still in the conversation the model sees. The API is
stateless; the memory is the growing list you choose to resend.
</details>

---

## Section 10: Multi-agent

**Recall.** In `examples/09_multi_agent.py`, what *is* the research sub-agent, from
the orchestrator's point of view?

<details><summary>▸ Answer</summary>

Just another tool. Calling `research(question=...)` looks identical to calling any
tool, but its function body runs a second agent loop with its own prompt and
tools. Sub-agents are agents wrapped as tools; that's how large systems decompose.
</details>

---

## Capstone: `agent_cli.py`

**Do.** Run `secrun python hands_on/agent_cli.py "Save a note titled 'todo' with body 'x'"`
and deny the approval prompt. Then run it again with `--yes`. Then add `--trace` to
either. You've now exercised the loop, the approval gate, and observability in one
tool.

**Stretch.** Add a new tool to `agent/tools.py` (say, a `word_count(text)` tool),
include it in `default_tools()`, and ask the capstone to use it. Adding a
capability is just: write a function, describe it, register it. That's the whole
extensibility story.

---

## Going further: three more agent patterns

**Recall (`11`).** You can solve a support task with a hard-coded workflow or with
the agent loop. When should you *not* reach for an agent?

<details><summary>▸ Answer</summary>

When you can already **draw the flowchart**. If the steps are known, a workflow
(classify → route → handle, orchestrated in code) is cheaper, more predictable, and
easier to test. An agent earns its cost and unpredictability only when the path is
genuinely open-ended.
</details>

**Recall (`12`).** Planning and reflection are both extra LLM passes around the loop.
What does each buy you, and what's the weakness of self-reflection?

<details><summary>▸ Answer</summary>

**Plan** (before) keeps a multi-step task from dropping a sub-goal. **Reflect**
(after) catches half-answers and slips, then revises. The weakness: a model grading
itself is fallible; the strong version grounds the critic in a real check (tests, a
schema, a verifier), as in the prompt-engineering "reflexion" lesson.
</details>

**Predict (`13`).** The model asks for three independent `get_weather` calls in one
turn. Why does running them concurrently help, and roughly how much?

<details><summary>▸ Answer</summary>

Independent calls don't depend on each other, so there's no reason to wait. Run them
concurrently and the turn finishes in the time of the **slowest** call, not the
**sum**. Three ~0.8s calls: ~2.4s sequential vs ~0.8s parallel (~3×). The final
answer is a normal completion, so you can stream it on top.
</details>

**Recall (`14`).** Example 13 streams the final answer; example 14 streams every
turn. What has to change in the loop, and what's the one fiddly part of streaming a
turn that *also* calls a tool?

<details><summary>▸ Answer</summary>

Almost nothing changes. You swap `run_turn` for `stream_turn` and print the text
deltas via a callback; the rest of the loop (run tools, feed results back, repeat) is
identical. The fiddly part is reassembling the tool call: OpenAI streams the
`arguments` as JSON *fragments* you must concatenate by index before you can parse
them. That lives in `agent/providers.py` so the loop stays clean.
</details>

---

## Section: Provider-hosted tools

**Predict (`15`).** You run `examples/15_hosted_tools.py` with the hosted web-search
tool declared. How many client-side tool rounds does *your* loop handle, and does
that mean no search happened?

<details><summary>▸ Answer</summary>

Zero client-side rounds, and search *did* happen. A hosted tool is run by the
provider *inside the turn*, not by your loop. You send one request and get one final
answer already grounded in search; there's no tool_use/tool_result round-trip for
your code, even though the provider ran search one or more times (the example prints
that count). Client-executed and hosted tools are different in kind.
</details>

**Recall.** What do you give up by using a hosted tool instead of a client-executed
one, and what do you gain?

<details><summary>▸ Answer</summary>

You give up **control**: the call runs on the provider's side, so Section 7's
approval gate can't reach it, and you can't custom-log, validate, or sandbox it. You
gain **zero plumbing**: no schema, no execution, no result round-trip. Real agents
mix both: client tools for actions you must govern, hosted tools (web search, code
execution) for capability you're happy to rent.
</details>

---

## Bonus: MCP: a tool you didn't ship with **(offline)**

**Predict (`10`).** The client turns each remote tool descriptor into an ordinary
`Tool` object and hands it to the loop. When the agent later "runs" one, what does
that tool's `func` actually do, and could `run_agent` from example 03 tell the
difference from a local tool?

<details><summary>▸ Answer</summary>

The `func` is a closure that sends a `tools/call` request over the protocol (JSON
lines over the server's stdin/stdout) and reads the text content blocks back. The
loop *can't* tell the difference; the converted tool has the same name,
description, schema, and callable shape as a local one, so tracing and approval
gates apply unchanged. That's the payoff of the conversion step: the transport is
invisible to the loop.
</details>

**Recall.** Example 01 said a tool is "a name, a description, and a JSON Schema."
What does MCP add to that picture, and what still never happens, whether the tool
is local or served across the world?

<details><summary>▸ Answer</summary>

MCP adds **discovery and a wire format**: `tools/list` advertises those same three
things from another process (another team, another language, another machine), and
`tools/call` invokes one by name with JSON arguments, with no bespoke glue per tool.
What never changes: the *model* still never runs anything. It only ever sees the
menu entry and emits a request. Now even your client doesn't hold the
implementation; the server does. (For the protocol's other primitives, like resources,
prompts, HTTP transport, and security, see the MCP deep dive.)
</details>

---

### Where to take it next

Invent your own. Wire one of your real functions in as a tool, something that
reads a file, hits an API you own, or queries a database. Mark it dangerous if it
writes, and let the agent drive it. The first time an agent completes a multi-step
task you didn't script step-by-step, the idea has landed.
