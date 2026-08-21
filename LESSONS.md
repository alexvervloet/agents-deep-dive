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
