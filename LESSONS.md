# Lessons

## 2026-08-21: Enforcing exact schemas exposed two older declarations

Expected: routing the existing examples through `ToolExecutor` would be a local
loop change because every `Tool` already carried a JSON Schema.

Actual: the sub-agent tool in example 09 and weather tool in example 13 did not
set `additionalProperties: false`. The former would fail when the new executor
compiled its contract, and the latter could not honestly advertise OpenAI strict
mode. The built-in and MCP-served tools were already exact.

Next time: before making a declaration invariant executable, inventory every
constructor call across examples and adapters, not only the library defaults.
Add a repository-wide contract-definition test if the tool catalog grows beyond
the current small set.

## 2026-08-21: An idempotency key without its payload is not a key

Expected: keying the replay cache on `(request_id, call.id, tool)` was enough,
since providers mint unique call IDs.

Actual: an audit of the finished work showed that a repeat of that key carrying
*different* arguments returned the settled result, and recorded the first call's
argument digest against the second attempt. The evidence trail said nothing had
happened. The digest was already being computed; it just was not being compared.

Next time: when a cache key stands in for "the same operation", write the test
that reuses the key with a different payload before writing the one that reuses
it with the same payload. The happy-path replay test passes either way, so it
proves less than it appears to.

## 2026-08-21: Strictness that is right for your tools is fatal for discovered ones

Expected: requiring `additionalProperties: false` on every tool schema was a
uniformly good invariant.

Actual: it made every third-party MCP tool unusable. `ToolExecutor` raised before
a single model call, while the README still promised discovered tools "drop into
the loop unchanged". CI never caught it because the only server under test is this
repo's own, which writes exact schemas. Two other files (`LESSONS.md` and the
OpenAI strict-mode check) had already written down that remote schemas are looser;
the executor was the one place that had not absorbed it.

Next time: an invariant about *code you wrote* needs a stated answer for input you
did not write, before it ships. Here the answer was `seal_schema()`, applied
visibly at the point of adoption. Also: when a doc makes a promise about
third-party input, test against a fixture that is actually shaped like third-party
input, not against your own well-behaved server.

## 2026-08-21: Docs can be more current than the code they describe

Expected: the OpenAI strict-mode gate would need a live check mainly for the
object-shape rules.

Actual: the published supported-keyword list omits `minLength` and `maxLength`,
which the repo's own calculator and search schemas both use, so reading the docs
suggested every OpenAI request would 400. A single live call accepted the schema
and returned a normal tool call. The docs were behind the API.

Next time: for a claim about a provider's runtime behaviour, one real request
settles it faster and more reliably than any amount of documentation reading, and
this repo's `secrun` makes that cheap. Reserve the docs for questions live calls
cannot answer.
