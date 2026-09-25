"""Deterministic execution of an already-approved ``ToolCall``.

This is the low-level primitive the future Executor will use - it applies
the tool's declared timeout and retry policy and normalizes every outcome
into a ``ToolResult``. It contains NO LLM calls, NO state-machine access,
and NO orchestration loop (those belong to later phases).

Security notes:
- tools run in a worker thread with a hard timeout
- no exception ever propagates to callers: failures become structured
  ``ToolResult`` values with stable error codes
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from app.state.models import ToolCall, ToolResult
from app.tools.base import (
    DestinationExistsError,
    FileMissingError,
    FileTooLargeError,
    MalformedToolInputError,
    PathTraversalError,
    ToolError,
    ToolExecutionError,
    ToolTimeoutError,
    WebSearchUnavailableError,
)
from app.tools.registry import ToolNotFoundError, ToolRegistry

__all__ = ["execute_tool_call"]

_ERROR_CODES: dict[type[ToolError], str] = {
    ToolTimeoutError: "TOOL_TIMEOUT",
    MalformedToolInputError: "MALFORMED_INPUT",
    ToolExecutionError: "TOOL_FAILURE",
    PathTraversalError: "PATH_TRAVERSAL",
    FileMissingError: "FILE_MISSING",
    FileTooLargeError: "FILE_TOO_LARGE",
    DestinationExistsError: "FILE_EXISTS",
    WebSearchUnavailableError: "UNAVAILABLE",
}

# Deterministic failures: retrying them can only waste time.
_NON_RETRYABLE: tuple[type[ToolError], ...] = (
    MalformedToolInputError,
    PathTraversalError,
    FileMissingError,
    FileTooLargeError,
    DestinationExistsError,
    WebSearchUnavailableError,
)


def _error_code(error: ToolError) -> str:
    for klass in type(error).__mro__:
        code = _ERROR_CODES.get(klass)
        if code is not None:
            return code
    return "TOOL_FAILURE"


def _run_with_timeout(tool: object, arguments: dict) -> object:
    """Run ``tool.run`` in a worker thread bounded by ``tool.timeout_s``."""
    name = getattr(tool, "name", "?")
    timeout_s = getattr(tool, "timeout_s", 5.0)
    run = getattr(tool, "run")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"orbit-{name}")
    try:
        future = executor.submit(run, arguments)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            raise ToolTimeoutError(
                f"tool {name!r} exceeded its {timeout_s}s timeout",
                tool_name=name,
            ) from exc
    finally:
        # A timed-out worker cannot be killed; it is abandoned (documented
        # limitation of sync tools - the Phase 3 executor may revisit this).
        executor.shutdown(wait=False)


def execute_tool_call(registry: ToolRegistry, call: ToolCall) -> ToolResult:
    """Execute an approved ``ToolCall`` under its tool's timeout/retry policy.

    Always returns a ``ToolResult`` - never raises for tool-level failures:
    - success -> ``ok=True`` with validated ``data``
    - unknown tool -> ``ok=False`` / ``TOOL_NOT_FOUND``
    - malformed input -> ``ok=False`` / ``MALFORMED_INPUT``
    - tool errors -> ``ok=False`` with a stable ``error_code``
    - retries are bounded by ``tool.max_retries`` with exponential backoff
    """
    start = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - start) * 1000.0

    try:
        tool = registry.get(call.tool_name)
    except ToolNotFoundError as exc:
        return ToolResult(
            ok=False,
            error=str(exc),
            error_code="TOOL_NOT_FOUND",
            duration_ms=elapsed_ms(),
        )

    attempts = tool.max_retries + 1
    last_error = "tool execution failed"
    last_code = "TOOL_FAILURE"

    for attempt in range(attempts):
        try:
            output = _run_with_timeout(tool, dict(call.arguments))
            return ToolResult(
                ok=True,
                data=output.model_dump(mode="json"),  # type: ignore[union-attr]
                duration_ms=elapsed_ms(),
            )
        except MalformedToolInputError as exc:
            return ToolResult(
                ok=False,
                error=str(exc),
                error_code="MALFORMED_INPUT",
                duration_ms=elapsed_ms(),
            )
        except ToolError as exc:
            last_error = str(exc)
            last_code = _error_code(exc)
            if isinstance(exc, _NON_RETRYABLE):
                return ToolResult(
                    ok=False,
                    error=last_error,
                    error_code=last_code,
                    duration_ms=elapsed_ms(),
                )
        except Exception as exc:  # defensive: normalize unexpected failures
            last_error = f"{type(exc).__name__}: {exc}"
            last_code = "TOOL_FAILURE"

        if attempt < attempts - 1:
            delay = tool.retry_backoff_s * (2**attempt)
            if delay > 0:
                time.sleep(delay)

    return ToolResult(
        ok=False,
        error=last_error,
        error_code=last_code,
        duration_ms=elapsed_ms(),
    )
