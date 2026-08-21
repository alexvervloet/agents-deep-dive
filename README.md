# Agents: A Guided Deep Dive

A hands-on playground for learning how **LLM agents** actually work, by building
one from scratch. You'll write the agentic loop yourself and understand every
moving part: tools, the loop, multi-tool routing, step limits, error recovery,
human-in-the-loop approval, enforceable tool contracts, observability, memory,
and multi-agent delegation. No
LangChain, no SDK tool-runners, no framework magic, just enough code to see how
an agent thinks.

This is the sixth of eight core repos in the series, and the one where the building blocks converge. The
first two teach the API calls ([OpenAI](https://github.com/alexvervloet/openai-api-deep-dive),
[Claude](https://github.com/alexvervloet/claude-api-deep-dive)); [prompt engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive) sharpens how you ask; [RAG](https://github.com/alexvervloet/rag-deep-dive) adds
retrieval; [evals](https://github.com/alexvervloet/evals-deep-dive) measures quality. An agent *uses*
all of it: it calls the API in a loop, its tools can include RAG retrieval, and
its step-by-step behavior is exactly what you'd evaluate. Tools + loop is the
pattern under "AI agents," and once you've written it by hand, the frameworks stop
being magic.

Like its siblings, it's meant to be *walked through*. Each section ends with
something to run; examples 01, 10, and 18 run **offline and free**.
[EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for each section.

---

## 0. The one big idea

> **An agent is a loop: the model picks a tool, you run it, you feed the result
> back, until it's done.**

That's the entire concept. A model on its own can only produce text; give it
*tools* and a *loop* and it can take actions, observe results, and decide what to
do next. Everything in this repo (multiple tools, error handling, approval,
memory, sub-agents) is a small addition to that loop, not a new idea. Hold onto
it and none of this feels complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Choose your provider (set PROVIDER in .env); your key loads separately
cp .env.example .env
#    Your API key does NOT go in .env. Store it in your OS keychain and run
#    lessons with `secrun`: 2-minute setup in ../SECRETS.md.

# 4. Confirm everything is wired up (makes no API call, costs nothing)
secrun python check_setup.py       # secrun injects your key so the check can see it
```

Agents are provider-agnostic, so this repo is too. Pick whichever stack you set up
in the sibling repos with `PROVIDER` in `.env`:

| `PROVIDER` | Chat model | Key needed |
|------------|-----------|------------|
| `openai` (default) | OpenAI `gpt-5.4-nano` | `OPENAI_API_KEY` |
| `claude` | Claude `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |

Tool-calling has a genuinely different *shape* per provider (OpenAI's
`function`/`tool_calls` vs Claude's `tool_use`/`tool_result` blocks). The one file
that knows the difference is [agent/providers.py](agent/providers.py); the loop and
everything above it stay identical. That's the whole point: agents are an
architecture, not a provider feature.

> **Start before spending anything.** Examples 01, 10, and 18 are completely
> offline. They cover tool shape, protocol transport, and the execution contract
> with no key and no cost; provider-backed examples make small, cheap calls.

---

## 2. What a tool is

A tool has two faces: to your code it's a plain function; to the model it's just a
name, a description, and a JSON Schema of inputs. The model never runs anything 
it only *asks*, and your code decides whether to execute. That gap is where every
bit of an agent's safety lives.

```bash
python examples/01_tools.py          # offline
```

See [agent/tools.py](agent/tools.py) for the toolbox: a safe `calculator`, a
read-only `search_notes` (over a tiny knowledge base, where a real agent would
call your RAG pipeline), and `save_note`, which writes a file and is therefore
flagged **dangerous**. The description and parameter names are the model's only
clues for when and how to call a tool; they're prompt engineering, not
afterthoughts. The schema is also enforced locally before execution in Section
7A. Describing allowed inputs to a model and accepting an untrusted request are
two different jobs.

---

## 3. One tool call

The core mechanic in isolation: hand the model tools and a question, and instead
of answering it replies "please run `calculator` with `expression='23 * 47'`."
That's a request, and you run it.

```bash
secrun python examples/02_one_tool_call.py
```

This does exactly one turn so you can see the request shape clearly (normalized to
the same `ToolCall` on either provider). It doesn't feed the result back yet 
that's the loop, next.

---

## 4. The agent loop

Repeat that one turn (run the tool, feed the result back, ask again) until the
model stops asking. That loop is the agent.

```bash
secrun python examples/03_agent_loop.py
```

`run_agent` in [agent/loop.py](agent/loop.py) keeps that control flow small and
routes every requested action through the contract boundary. Watch the trace:
given a multi-step question, the model chains calls, using each result to decide
the next, something a single call can't do. **This is the example to really
understand; everything after it is a small addition.**

---

## 5. Multiple tools: the model chooses

Give the agent more than one tool and it routes each sub-task to the right one and
chains them.

```bash
secrun python examples/04_multiple_tools.py
```

Asked "what does the Plus plan cost per year?", it calls `search_notes` for the
price, then `calculator` to multiply, with no hard-coded plan. The tool *descriptions*
are what make that routing reliable.

---

## 6. Control: step limits and error recovery

An unsupervised loop needs guardrails. Two essential ones are built into
`run_agent`:

```bash
secrun python examples/05_limits_and_errors.py
```

- **`max_steps`**: a hard ceiling, so a confused agent stops and says so instead
  of looping forever.
- **Error recovery**: when a tool raises, the error text goes back to the model
  *as the result*, so it can adapt instead of crashing the program.

These two are the difference between a toy loop and one you'd run unattended.

---

## 7. Human-in-the-loop approval

Some actions have consequences (writing files, sending email, spending money).
Mark those tools `dangerous=True` and pass an `approve` callback; the loop asks
before running them. Dangerous tools fail closed when that callback is absent.

```bash
secrun python examples/06_human_in_the_loop.py     # interactive
```

`save_note` is dangerous, so you're prompted; the calculator and search run freely.
Deny the call and the agent adapts; a denial is just another tool result. Which
tools are "dangerous" is your policy, declared on the tool.

---

## 7A. Tool contracts: validate before effects

A tool call is an untrusted proposal, even when a provider generated it in strict
mode. The model may be confused, compromised by prompt injection, or connected
through a provider or MCP server with different schema guarantees. The local
application therefore owns the execution decision:

```text
model proposal
  -> reject forged identity/tenant fields
  -> validate the complete JSON Schema
  -> authorize trusted session roles
  -> replay a settled result, or reject the key if the payload changed
  -> require approval
  -> execute with time/output limits
  -> record a structured outcome and content digests
```

```bash
python examples/18_tool_contracts.py          # offline
python -m unittest discover -s tests -v       # adversarial checks
```

The example uses a refund tool with an enum, identifier pattern, numeric range,
exact properties, billing role, trusted tenant/subject injection, approval,
bounded replay storage, a timeout, and a UTF-8 output-byte limit. Five independent
probes show why each control exists. Only the allowed proposal creates an effect;
replaying its request returns the stored outcome without issuing another refund.

[agent/contracts.py](agent/contracts.py) is the reusable implementation. Its
in-process replay cache deliberately stores even errors that happened after
dispatch: a timeout or connection failure cannot tell you whether the remote
effect committed. A settled key is a promise about one specific payload, so a
repeat carrying *different* arguments is denied (`idempotency_key_reuse`) rather
than answered from the cache; otherwise the audit record for the second attempt
would carry the first call's digest and the attempt would leave no trace. This prevents a duplicate retry in the teaching process, but a
production write must enforce a durable idempotency key transactionally at its
sink. The thread timeout also stops waiting rather than killing already-running
Python code; hard cancellation needs a subprocess, worker, or remote deadline.

OpenAI recommends strict function schemas and requires exact object properties
and required fields for that mode; this repo enables it when a schema is
compatible, then validates locally anyway. See the official
[OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)
and [JSON Schema 2020-12 validation vocabulary](https://json-schema.org/draft/2020-12/json-schema-validation).
Strict generation is defense in depth, not authorization.

---

## 8. Observability: see what it did

An agent makes its own decisions, so when it misbehaves you need the trace: which
tool, what arguments, what result, at each step.

```bash
secrun python examples/07_observability.py
```

You've seen the live `Tracer`; this also shows the same steps after the fact from
`result.steps`, the structured record you'd log in production and feed to an eval
(see the [evals repo](https://github.com/alexvervloet/evals-deep-dive)) to score whether the agent called the
right tools in a sensible order.

---

## 9. Memory: remembering the conversation

The API is stateless (the same lesson as the sibling repos): "memory" is just you
re-sending the growing message list. `run_agent` takes an optional `history` list
and appends to it in place, so passing the same list across turns gives the agent
memory.

```bash
secrun python examples/08_memory.py          # interactive REPL
```

Ask it to "search the plans," then "which is cheapest?" The follow-up only works
because the earlier turn is still in the history you resend.

---

## 10. Multi-agent: agents that call agents

As tasks grow, one agent with twenty tools gets unfocused. Delegate instead. A
sub-agent is not a new mechanism. It's a tool whose function happens to run its
own loop, with its own prompt and toolset.

```bash
secrun python examples/09_multi_agent.py
```

An orchestrator delegates factual questions to a `research` sub-agent (tools:
`search_notes`) and does math itself. To the orchestrator, `research` is just a
tool; underneath, it's a whole second loop. That's how large agent systems are
built: focused agents calling each other through the same tool interface.

---

## 11. The capstone: `agent_cli.py`

Everything assembled into a CLI agent you can actually use: the full toolbox, the
loop with a step cap, approval for the dangerous tool, an optional trace, and a
memory-keeping interactive mode.

```bash
# One-off task
secrun python hands_on/agent_cli.py "What's a year of the Plus plan, and is offline editing included?"

# Watch every step
secrun python hands_on/agent_cli.py "What is 19% of 240?" --trace

# Interactive chat with memory (type 'quit' to exit)
secrun python hands_on/agent_cli.py

# Save notes without being prompted each time
secrun python hands_on/agent_cli.py "Save a note titled 'todo' with body 'ship the repo'" --yes
```

Read [hands_on/agent_cli.py](hands_on/agent_cli.py); it's just the library wired to a CLI.
**Suggested exercise:** add a new tool to `agent/tools.py` (say `word_count`),
register it in `default_tools()`, and watch the agent pick it up. Adding a
capability is: write a function, describe it, register it.

---

## Going further: five more agent patterns

The loop is the core; these are the patterns you layer on it in real systems.

### Workflows vs. agents
"Agent" isn't always the answer. If you can draw the flowchart, build a **workflow** 
fixed steps you orchestrate in code (classify → route → handle). It's cheaper, more
predictable, and easier to test. Reach for an **agent** (the model drives the loop)
only when the path genuinely can't be known up front. The example does one support
task both ways.
```bash
secrun python examples/11_workflows_vs_agents.py
```

### Planning & reflection
Two cheap wrappers around the loop that boost reliability on multi-part tasks: ask
the model to write a short **plan** before it acts (keeps long tasks on track), and
run a **reflection** pass after (a critic catches half-answers, then revises). Best
when the critic is grounded in a real check. See the prompt-engineering "reflexion"
lesson and the evals dive.
```bash
secrun python examples/12_planning_reflection.py
```

### Parallel tool calls & streaming
When the model requests several *independent* tool calls in one turn, run them
**concurrently**, so the turn costs the slowest call, not the sum. And the final answer
is an ordinary completion, so **stream** it token by token for instant, responsive
output. The example times sequential vs. parallel execution, then streams the answer.
```bash
secrun python examples/13_parallel_and_streaming.py
```

### Streaming *inside* the loop
Example 13 streams the *final* answer; this streams *every* turn, including the ones
that request tools, so the user watches the agent narrate ("let me look that up...")
between tool calls instead of staring at a spinner. The loop is unchanged; you just
swap `run_turn` for `stream_turn`, which prints text deltas live and still hands back
the normalized tool calls (reassembling streamed tool-call fragments is the one fiddly
bit, kept in `agent/providers.py`). This is the pattern most production assistants use.
```bash
secrun python examples/14_streaming_tool_loop.py
```

### Provider-hosted tools: the loop never sees it
Every tool so far was **client-executed**: the model asks, *your* loop runs the
function, you feed the result back. A **hosted** tool is different in kind: you
*declare* it and the provider runs it *inside the turn*, on its own infrastructure.
You send one request and get one final answer; there's no tool_use/tool_result
round-trip for your loop to manage, because your loop isn't in the middle. The
example asks a question with hosted **web search** declared and shows the gap:
search really ran (the provider did it), but your code handled **zero** tool rounds.
The tradeoff is control for plumbing. A hosted tool can't be gated (Section 7),
custom-logged, or sandboxed, but it needs no glue. Real agents mix both.
```bash
secrun python examples/15_hosted_tools.py       # small real call; degrades cleanly if the tool isn't enabled
```

---

## Bonus: MCP: a tool you didn't ship with

Every example so far imported its tools straight from `agent/tools.py`. Real
agents often can't: the tool lives in another team's service, a vendor's product,
or a process in another language. **MCP (the Model Context Protocol)** is the
standard that makes that work: a tool *server* advertises what it offers, and the
agent *client* discovers and calls those tools over one agreed wire format, with
no bespoke glue per tool. It's the same idea as Section 2 (*a tool is a name, a
description, and a JSON Schema*), now spoken over a protocol instead of an import.

This repo ships a real, from-scratch MCP server and client: JSON-RPC over stdio,
the actual `tools/list` / `tools/call` methods, no SDK, so you can see the
protocol rather than import it. It's fully offline (no model, no key):

```bash
python examples/10_mcp.py
```

The payoff is the conversion step in the client: each remote tool descriptor
becomes an ordinary `Tool` object, so an MCP-served tool drops into the loop from
Section 4 unchanged. The agent can't tell a local function from a tool served
across the world. [agent/mcp_server.py](agent/mcp_server.py) is the server (it
serves the very same `calculator` and `search_notes` functions, now over the
wire); [examples/10_mcp.py](examples/10_mcp.py) is the client.

**A protocol moves the tool, not the trust.** Section 7A's contract applies twice
here, once on each side of the pipe, and both directions are easy to skip:

- *The client seals what it adopts.* A discovered schema was written by someone
  else, and MCP does not require `additionalProperties: false`, so most schemas
  in the wild leave it unset. `ToolExecutor` refuses a schema that loose, on
  purpose: the omission should be impossible to ignore. `seal_schema()` closes a
  copy at the point of adoption, which is where a human is actually deciding to
  trust this server. The tradeoff is real and worth stating: sealing can reject a
  call a sloppy server would have accepted, because it refuses to forward fields
  nobody declared. That is the better failure.
- *The server distrusts its clients.* The one in this repo runs every
  `tools/call` through the same `ToolExecutor` before dispatch, so a client that
  invents an argument gets a contract denial rather than a Python call. Your
  server has no idea whose model is on the other end, or whether that model just
  read a prompt-injected web page. "The client already validated" is not
  something a server can ever know.

`tests/test_mcp_contracts.py` holds both halves down.

In production you'd use the official `mcp` SDK and a real transport (HTTP/SSE),
and your provider can often skip the client entirely: the Claude API connects to
remote MCP servers for you (its MCP connector), and the OpenAI stack has an
equivalent. The protocol shape you just built by hand is exactly what those use.

---

## Bonus: the tool surface at scale (tool search & PTC)

```bash
secrun python examples/16_tool_search_and_ptc.py
```

Everything above assumed a handful of tools and a handful of calls. Both break
in the same place: the context window. Thirty tools means thousands of tokens of
schema on every request, most of it irrelevant. Forty tool calls means forty
results in context when you wanted one number.

**Tool search** fixes the first. Mark tools `defer_loading: True`, add a search
tool, and the model loads only the schemas it needs. The loaded schemas are
*appended* rather than swapped, so the cached prefix survives. Never defer
everything: the search tool must stay loaded and at least one tool must be
non-deferred, or you get a 400.

**Programmatic tool calling** fixes the second. Give a tool
`allowed_callers: ["code_execution_20260120"]` and Claude can call it from inside
a script running in the code-execution container. Results return to the running
program, not to the context window, so cost scales with the size of the answer
instead of the number of calls. Standard tool use is "model asks, you answer,
model reads"; PTC is "model writes a program, the program asks, the program
reads". PTC needs Sonnet 4.5 / Opus 4.5 or newer; tool search runs on Haiku 4.5.

---

## Bonus: memory that outlives the process

```bash
secrun python examples/17_memory_tool.py    # run it twice
```

Section 9's memory is the message list, which is the right default and dies with
the process. The **memory tool** is the other kind: Claude gets a `/memories`
directory it reads and writes through tool calls, and because it is a
*client-side* tool, you implement the storage and decide how long it lives.

Declaring the tool does not give you storage; it tells Claude the commands
exist. That is what lets you scope memory per user and delete it on request.

Two rules the example enforces rather than merely mentions. **Validate every
path**: they are model-generated, the agent reads untrusted content (see the
[Prompt Injection dive](https://github.com/alexvervloet/prompt-injection-deep-dive)),
and a six-line guard is the difference between a memory directory and an
arbitrary file write. **Never store secrets**: memory is replayed verbatim into
future contexts, so a key written once leaks into every later session.

---

## Where to go next

You've built a real agent from scratch. The frontier is more of the same loop, with
more capability and rigor:

- **Build on a harness**: the whole next step. Most agent work in 2026 is building
  *on* a harness (hooks, permission policies, sandboxing, subagents, headless runs)
  rather than hand-rolling the loop. The **[Agent Harnesses dive](https://github.com/alexvervloet/agent-harness-deep-dive)**
  builds one from scratch and covers when to throw away your loop for the SDK,
  plus computer use and hosted sandboxes.
- **MCP at scale**: you built the protocol by hand above; the official `mcp` SDK,
  remote (HTTP/SSE) transports, auth, and provider-side connectors are the
  production version.
- **Managed / hosted agents**: let the provider run the loop and host a sandbox
  for tool execution (Anthropic's Managed Agents, OpenAI's Agents/Assistants).
- **Server-side & computer-use tools**: web search, code execution, and driving a
  real GUI, where the provider runs the tool for you.
- **Planning & reflection**: having the agent draft a plan, critique its own work,
  or retry failed sub-tasks, on top of the basic loop.
- **Production hardening**: sandboxing execution, durable idempotency,
  concurrency-safe replay coordination, cost budgets, retries, and centralized
  logging/tracing beyond Section 7A's single-process boundary.
- **Evaluating agents**: scoring trajectories (right tools, right order, no wasted
  steps), not just final answers, exactly what the [evals repo](https://github.com/alexvervloet/evals-deep-dive)
  is for.
- **SDK tool-runners**: now that you've written the loop by hand, the official
  SDKs' tool-runner helpers will read as conveniences, not magic.

Each is a variation on the one idea you started with: the model picks a tool, you
run it, you feed the result back.

---

## From teaching code to production

An agent is the riskiest thing to put in production: it loops, calls tools, and
spends on its own. Every shortcut that's fine in a demo becomes a liability once
it runs unattended:

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| The loop runs until it's done | A **cost budget** *and* step ceiling per run, so a stuck loop can't rack up a bill |
| Section 8's observability is `print()` | A **structured trace** with a **span per step**: which tool, which args, how long, how many tokens |
| Tool/model errors handled inline (Section 6) | **Retries + backoff** and a **circuit breaker** around every model and tool call |
| Section 7A keeps replay/audit state in one process | A **durable, transactional tool gateway** shared by every worker, with sink-enforced idempotency and hard cancellation |
| The system prompt is a literal in the script | A **versioned prompt** promoted only past an **eval gate** on agent behavior |
| Every step re-calls the model | A **response cache** for repeated sub-calls |

These shortcuts are right for learning and wrong for production. All seven
concerns (observability, cost, reliability, caching, guardrails, prompt
versioning, and eval gates) are built from scratch and wired into one running
app in **[Production](https://github.com/alexvervloet/ai-in-production-deep-dive)** (#8 in the
series). It runs **offline on a mock provider**, so you can see the whole ops
machinery with no key and no cost.

---

## File map

```
check_setup.py              ← run first: verifies Python, packages, provider, key
README.md                   ← this guide
EXERCISES.md                ← predict-then-run prompts, one per section
agent/                      ← the from-scratch agent library (read it!)
  tools.py                  ← what a tool is + the safe default toolbox
  contracts.py              ← local validation, policy, execution limits, replay, audit
  providers.py              ← the ONLY provider-specific file: normalizes a turn
  loop.py                   ← run_agent (the loop) + Tracer + AgentResult
  mcp_server.py             ← a from-scratch MCP tool server (JSON-RPC over stdio)
hands_on/
  agent_cli.py              ← capstone: a CLI agent (one-off or interactive)
examples/
  01_tools.py               ← what a tool is (offline, no key)
  02_one_tool_call.py       ← one turn: the model requests a tool call
  03_agent_loop.py          ← the loop, the whole idea
  04_multiple_tools.py      ← the model routes between tools
  05_limits_and_errors.py   ← max_steps + feeding errors back
  06_human_in_the_loop.py   ← approval gate for dangerous tools
  07_observability.py       ← tracing each step, live and after the fact
  08_memory.py              ← multi-turn memory via a shared history
  09_multi_agent.py         ← an orchestrator delegating to a sub-agent
  10_mcp.py                 ← use a tool over MCP: offline client + server, no key
  11_workflows_vs_agents.py ← when to hard-code a workflow vs. let the model drive
  12_planning_reflection.py ← plan before acting; reflect & revise after
  13_parallel_and_streaming.py ← run independent tool calls concurrently; stream the answer
  14_streaming_tool_loop.py    ← stream every turn (incl. tool turns), not just the final answer
  15_hosted_tools.py        ← a provider-hosted tool (web search): the provider runs it inside the turn
  16_tool_search_and_ptc.py ← many tools, many calls: keeping both out of context
  17_memory_tool.py        ← memory that survives the process (client-side storage)
  18_tool_contracts.py     ← validate and authorize before effects (offline)
tests/
  test_tool_contracts.py   ← adversarial and counterfactual contract checks
  test_mcp_contracts.py    ← protocol calls cross the same boundary
```

(`workspace/` is created by the `save_note` tool and is git-ignored.)

---

## Troubleshooting

Run `secrun python check_setup.py` first; it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `PROVIDER=... needs ... in the environment` | Set `PROVIDER` in `.env`, then load the key from your keychain by running under `secrun`. See [SECRETS.md](../SECRETS.md). |
| `ModuleNotFoundError` (openai / anthropic / rich) | Dependencies aren't installed or the venv isn't active. `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| The agent answers math wrong / makes things up | It's not using its tools. Strengthen the system prompt ("use the calculator for arithmetic; don't guess product facts"). Tool *descriptions and instructions* drive tool use. |
| "(stopped: reached the step limit...)" | The task needed more steps than `max_steps`. Raise it (`--max-steps` on the capstone), or simplify the task. |
| A dangerous tool returns `approval_required` | Pass an `approve` callback, or use `--yes` in the capstone when you intentionally want to allow it. Dangerous tools fail closed without a callback. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained. Open it, read the docstring
at the top, and run it directly. The loop in `agent/loop.py` is the whole story.

---

## The series

This is one of the standalone, hands-on deep dives into building with LLM APIs: eight core, plus the bonus dives listed below.
Each one stands on its own, with its own setup, examples, and capstone, and they
all share the same house style: provider-agnostic, built from scratch (no
frameworks), offline-first examples, and a real capstone. Do them in any order;
this sequence builds naturally:

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive): the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive): the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive): shape model behavior with better prompts (zero/few-shot, chain-of-thought, roles)
4. [RAG](https://github.com/alexvervloet/rag-deep-dive): answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive): measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive): give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive): attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive): operate one app end to end: observability, cost, reliability, caching, guardrails, prompt versioning, eval gates

**Bonus dives**, standalone and slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive): manage what's in the window: memory, compaction, assembly
- [AI Data Engineering](https://github.com/alexvervloet/ai-data-engineering-deep-dive): the corpus behind the index: versions, lineage, ACLs, deletes
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive): images & audio, not just text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive): teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive): serve tools, data & prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive): run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive): build on the loop: hooks, permissions, sandboxing, subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive): low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive): watch a running app over time: drift, quality, alerting, the flywheel
- [Architecture](https://github.com/alexvervloet/architecture-deep-dive): the seams between the components, each decision measured rather than asserted
- [GenAI Security](https://github.com/alexvervloet/genai-security-deep-dive): treat the model as an untrusted principal: identity, supply chain, isolation, budgets, release gates
- [Inference Platform Engineering](https://github.com/alexvervloet/inference-platform-deep-dive): turn finite GPU memory and a request queue into latency, throughput, and a fleet size you can defend
- [Testing & Delivery](https://github.com/alexvervloet/testing-and-delivery-deep-dive): decide whether a build has earned promotion: evidence, gates, staged rollout, rollback
- [Professional Tools](https://github.com/alexvervloet/professional-tools-deep-dive): rebuild each from-scratch primitive with the tool professionals reach for, and measure both

And the whole series lands in one codebase in the
[capstone](https://github.com/alexvervloet/deep-dive-capstone): a codebase Q&A tool
built step by step, one tag per dive.

**You are here: #6, Agents.**
