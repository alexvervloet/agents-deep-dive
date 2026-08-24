"""Enforce tool calls as untrusted proposals before they reach Python code.

A provider schema helps the model produce well-shaped arguments; it is not an
authorization boundary. ``ToolExecutor`` repeats validation locally, derives
identity and tenant data from ``ExecutionContext``, checks roles and approval,
deduplicates mutating calls, bounds execution, and records structured evidence.

The replay cache is intentionally bounded, in-process, and optimized for this
single-process teaching loop. It does not coordinate duplicate calls already in
flight. It caches every post-dispatch outcome because an exception or timeout may
have happened after a side effect committed. A real service should enforce the
same idempotency key transactionally at the sink in a durable store. Likewise, a
thread timeout stops waiting but cannot kill work that is already running; use a
subprocess, job worker, or remote service deadline for hard cancellation.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from enum import Enum
from threading import Lock
from typing import Callable

from jsonschema import Draft202012Validator

from .providers import ToolCall
from .tools import Tool

ApprovalCallback = Callable[[ToolCall], bool]
_CONTEXT_FIELDS = frozenset({"request_id", "subject_id", "tenant_id"})


class ApprovalState(str, Enum):
    """Describe exactly how far a call progressed through the approval gate."""

    NOT_REACHED = "not_reached"
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPROVED = "approved"
    DENIED = "denied"
    ERROR = "error"


@dataclass(frozen=True)
class ExecutionContext:
    """Application-owned identity and request data that the model cannot choose."""

    request_id: str
    subject_id: str
    tenant_id: str
    roles: frozenset[str]

    def __post_init__(self) -> None:
        """Require stable identifiers so policy and replay keys are meaningful."""

        for name in ("request_id", "subject_id", "tenant_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")

    @classmethod
    def local(cls, request_id: str | None = None) -> ExecutionContext:
        """Create the explicit single-user context used by the teaching CLI."""

        return cls(
            request_id=request_id or str(uuid.uuid4()),
            subject_id="local-user",
            tenant_id="local-workspace",
            roles=frozenset({"user"}),
        )


@dataclass(frozen=True)
class ToolOutcome:
    """A stable result category plus provenance safe to feed to logs and tests.

    ``code`` is the field to read when you want to know what happened; it is the
    only one that distinguishes the reasons. ``approval`` separately records
    whether the gate was not reached, unnecessary, missing, approved, denied, or
    failed. It is deliberately not a Boolean: "schema rejected" and "human said
    no" are different states and must not share a value consumers can misread.
    """

    ok: bool
    code: str
    message: str
    output: str | None
    approval: ApprovalState
    replayed: bool
    arguments_sha256: str
    output_sha256: str | None
    duration_ms: float

    def for_model(self) -> str:
        """Render the useful result while preserving a machine-readable category."""

        if self.ok:
            return self.output or ""
        return f"Error [{self.code}]: {self.message}"


@dataclass(frozen=True)
class ToolAuditRecord:
    """Bounded metadata for one decision; raw arguments and outputs stay out."""

    request_id: str
    tool_call_id: str
    tool: str
    subject_id: str
    tenant_id: str
    code: str
    approval: ApprovalState
    replayed: bool
    arguments_sha256: str
    output_sha256: str | None
    duration_ms: float


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# JSON Schema keys whose values are themselves schemas. Listing them explicitly
# keeps `seal_schema` from wandering into `enum`/`const`, where a literal value
# may happen to look like a schema.
_SCHEMA_MAP_KEYS = ("properties", "patternProperties", "$defs", "definitions")
_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_KEYS = ("items", "not", "if", "then", "else")


def seal_schema(schema: dict) -> dict:
    """Return a copy of a schema with every object closed to undeclared fields.

    Tools you write should set ``additionalProperties: false`` themselves, and
    ``ToolExecutor`` refuses anything looser so the omission cannot pass silently.
    Schemas you *discover* are different: an MCP server, a plugin registry, or a
    partner API publishes a schema you did not write and often did not close.
    Sealing is how you adopt one, done once and visibly at the boundary rather
    than by weakening the executor for every tool.

    Sealing only narrows what the model may propose, which is the safe direction.
    The tradeoff is real: if a remote tool accepts a field it never
    declared, a sealed schema rejects a call that server would have honored. That
    is the price of refusing to forward arguments nobody described, and it is a
    better failure than forwarding an injected field to someone else's system.
    """

    sealed: dict = {}
    for key, value in schema.items():
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            sealed[key] = {
                name: seal_schema(sub) if isinstance(sub, dict) else sub
                for name, sub in value.items()
            }
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            sealed[key] = [
                seal_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif key in _SCHEMA_KEYS and isinstance(value, dict):
            sealed[key] = seal_schema(value)
        else:
            sealed[key] = value
    if sealed.get("type") == "object":
        sealed["additionalProperties"] = False
    return sealed


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ToolExecutor:
    """Apply one deterministic policy sequence to every client-executed tool.

    The ordering is deliberate: resolve the tool, reject forged trusted fields,
    validate the complete JSON Schema, authorize the trusted principal, return a
    prior mutating outcome, obtain approval, and only then dispatch the callable.
    This class owns policy decisions; the callable owns its domain operation.
    """

    def __init__(
        self,
        tools: list[Tool],
        *,
        maximum_replay_entries: int = 256,
        maximum_audit_entries: int = 1_000,
    ) -> None:
        """Compile schemas and create bounded replay and audit storage."""

        if maximum_replay_entries <= 0:
            raise ValueError("maximum_replay_entries must be positive")
        if maximum_audit_entries <= 0:
            raise ValueError("maximum_audit_entries must be positive")

        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("tool names must be unique")
        self._validators: dict[str, Draft202012Validator] = {}
        for tool in tools:
            Draft202012Validator.check_schema(tool.parameters)
            if tool.parameters.get("type") != "object":
                raise ValueError(f"{tool.name}: parameters must describe an object")
            if tool.parameters.get("additionalProperties") is not False:
                raise ValueError(
                    f"{tool.name}: parameters must set additionalProperties to false"
                )
            exposed = frozenset(tool.parameters.get("properties", {}))
            trusted = frozenset(tool.trusted_context_args)
            overlap = exposed & trusted
            if overlap:
                raise ValueError(
                    f"{tool.name}: trusted arguments must not be model-visible: "
                    f"{sorted(overlap)}"
                )
            unknown_context = set(tool.trusted_context_args.values()) - _CONTEXT_FIELDS
            if unknown_context:
                raise ValueError(
                    f"{tool.name}: unknown execution context fields: "
                    f"{sorted(unknown_context)}"
                )
            self._validators[tool.name] = Draft202012Validator(tool.parameters)

        self._maximum_replay_entries = maximum_replay_entries
        self._replays: OrderedDict[tuple[str, str, str], ToolOutcome] = OrderedDict()
        self._audit: deque[ToolAuditRecord] = deque(maxlen=maximum_audit_entries)
        self._lock = Lock()

    @property
    def audit_records(self) -> tuple[ToolAuditRecord, ...]:
        """Return an immutable snapshot of the currently retained audit records."""

        with self._lock:
            return tuple(self._audit)

    def _record(
        self,
        context: ExecutionContext,
        call: ToolCall,
        outcome: ToolOutcome,
    ) -> ToolOutcome:
        with self._lock:
            self._audit.append(
                ToolAuditRecord(
                    request_id=context.request_id,
                    tool_call_id=call.id,
                    tool=call.name,
                    subject_id=context.subject_id,
                    tenant_id=context.tenant_id,
                    code=outcome.code,
                    approval=outcome.approval,
                    replayed=outcome.replayed,
                    arguments_sha256=outcome.arguments_sha256,
                    output_sha256=outcome.output_sha256,
                    duration_ms=outcome.duration_ms,
                )
            )
        return outcome

    def _deny(
        self,
        context: ExecutionContext,
        call: ToolCall,
        code: str,
        message: str,
        arguments_sha256: str,
        *,
        approval: ApprovalState = ApprovalState.NOT_REACHED,
    ) -> ToolOutcome:
        return self._record(
            context,
            call,
            ToolOutcome(
                ok=False,
                code=code,
                message=message,
                output=None,
                approval=approval,
                replayed=False,
                arguments_sha256=arguments_sha256,
                output_sha256=None,
                duration_ms=0.0,
            ),
        )

    def _remember(
        self,
        key: tuple[str, str, str],
        outcome: ToolOutcome,
    ) -> None:
        with self._lock:
            self._replays[key] = outcome
            self._replays.move_to_end(key)
            while len(self._replays) > self._maximum_replay_entries:
                self._replays.popitem(last=False)

    def execute(
        self,
        call: ToolCall,
        context: ExecutionContext,
        *,
        approve: ApprovalCallback | None = None,
    ) -> ToolOutcome:
        """Validate, authorize, and execute one model-proposed call.

        Mutating calls use ``(request_id, call.id, tool name)`` as the local replay
        key. Keeping the client request ID outside model arguments binds retries to
        the application request rather than letting the model choose their scope.
        A repeat of that key must also carry the same effective arguments: reusing
        one key for a different payload is a caller bug or an attempt to have a
        settled decision stand in for a new one, so it is denied rather than
        answered from the cache.
        """

        proposal_digest = _json_digest(call.arguments)
        tool = self._tools.get(call.name)
        if tool is None:
            return self._deny(
                context,
                call,
                "unknown_tool",
                f"no tool named {call.name!r}",
                proposal_digest,
            )
        if call.parse_error is not None:
            return self._deny(
                context,
                call,
                "invalid_arguments",
                call.parse_error,
                proposal_digest,
            )
        if not isinstance(call.arguments, dict):
            return self._deny(
                context,
                call,
                "invalid_arguments",
                "tool arguments must be a JSON object",
                proposal_digest,
            )

        forged = set(call.arguments) & set(tool.trusted_context_args)
        if forged:
            return self._deny(
                context,
                call,
                "trusted_context_forgery",
                f"model supplied trusted arguments: {sorted(forged)}",
                proposal_digest,
            )

        errors = sorted(
            self._validators[tool.name].iter_errors(call.arguments),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.absolute_path) or "$"
            return self._deny(
                context,
                call,
                "schema_validation",
                f"{location}: {first.message}",
                proposal_digest,
            )

        if tool.allowed_roles and not context.roles & tool.allowed_roles:
            return self._deny(
                context,
                call,
                "not_authorized",
                "trusted principal lacks an allowed role",
                proposal_digest,
            )

        effective_arguments = dict(call.arguments)
        for argument, context_field in tool.trusted_context_args.items():
            effective_arguments[argument] = getattr(context, context_field)
        arguments_digest = _json_digest(effective_arguments)

        replay_key = (context.request_id, call.id, call.name)
        if tool.mutating:
            if not call.id.strip():
                return self._deny(
                    context,
                    call,
                    "missing_idempotency_key",
                    "mutating calls require a nonblank provider call ID",
                    arguments_digest,
                )
            with self._lock:
                cached = self._replays.get(replay_key)
                if cached is not None:
                    self._replays.move_to_end(replay_key)
            if cached is not None:
                if cached.arguments_sha256 != arguments_digest:
                    return self._deny(
                        context,
                        call,
                        "idempotency_key_reuse",
                        "this call ID already settled different arguments",
                        arguments_digest,
                    )
                return self._record(
                    context,
                    call,
                    replace(cached, replayed=True, duration_ms=0.0),
                )

        approval = ApprovalState.NOT_REQUIRED
        if tool.dangerous:
            if approve is None:
                return self._deny(
                    context,
                    call,
                    "approval_required",
                    "dangerous tool has no approval callback",
                    arguments_digest,
                    approval=ApprovalState.REQUIRED,
                )
            try:
                approved = bool(approve(call))
            except Exception as exc:  # approval systems fail closed
                return self._deny(
                    context,
                    call,
                    "approval_error",
                    f"approval callback failed: {type(exc).__name__}: {exc}",
                    arguments_digest,
                    approval=ApprovalState.ERROR,
                )
            if approved:
                approval = ApprovalState.APPROVED
            else:
                return self._deny(
                    context,
                    call,
                    "approval_denied",
                    "user denied permission to run this tool",
                    arguments_digest,
                    approval=ApprovalState.DENIED,
                )

        started = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{tool.name}")
        future = pool.submit(tool.func, **effective_arguments)
        try:
            raw_output = future.result(timeout=tool.timeout_seconds)
        except FutureTimeout:
            future.cancel()
            outcome = ToolOutcome(
                False,
                "timeout",
                f"tool exceeded {tool.timeout_seconds:g}s",
                None,
                approval,
                False,
                arguments_digest,
                None,
                (time.perf_counter() - started) * 1_000,
            )
        except Exception as exc:  # turn sink failures into structured outcomes
            outcome = ToolOutcome(
                False,
                "tool_error",
                f"{type(exc).__name__}: {exc}",
                None,
                approval,
                False,
                arguments_digest,
                None,
                (time.perf_counter() - started) * 1_000,
            )
        else:
            output = str(raw_output)
            output_digest = _text_digest(output)
            output_size = len(output.encode("utf-8"))
            if output_size > tool.maximum_output_bytes:
                outcome = ToolOutcome(
                    False,
                    "output_limit",
                    f"tool returned {output_size} bytes; limit is "
                    f"{tool.maximum_output_bytes}",
                    None,
                    approval,
                    False,
                    arguments_digest,
                    output_digest,
                    (time.perf_counter() - started) * 1_000,
                )
            else:
                outcome = ToolOutcome(
                    True,
                    "ok",
                    "tool executed",
                    output,
                    approval,
                    False,
                    arguments_digest,
                    output_digest,
                    (time.perf_counter() - started) * 1_000,
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if tool.mutating:
            self._remember(replay_key, outcome)
        return self._record(context, call, outcome)
