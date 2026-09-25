"""Observer: converts execution outcomes into structured Observations.

The Observer answers exactly one question: **"What happened?"**

- successful tool execution -> an ``ok`` observation (source=TOOL)
- failed tool execution -> a failed observation (source=TOOL)
- rejected tool decision -> a failed observation (source=DECISION)
- reasoning-task result -> an ``ok`` observation (source=REASONING)
- typed execution/LLM errors -> a failed observation (source=SYSTEM)

It does NOT decide whether the user's goal was satisfied - that is the
future validation stage's responsibility. It never touches workflow state
or the event log; the Executor records what the Observer produces through
the Controller.

Safety: summaries are bounded, raw results are size-bounded, and
chain-of-thought/prompts are never stored.
"""

from __future__ import annotations

import json

from app.llm.schemas import TaskResult
from app.state.models import JsonValue, Observation, ObservationSource, ToolResult

__all__ = ["Observer", "MAX_RAW_BYTES"]

MAX_RAW_BYTES = 16_000
"""Bound on the serialized size of ``Observation.raw`` (bytes)."""


def _bounded_raw(value: JsonValue | None) -> JsonValue | None:
    """Return ``value`` if small enough, else a truncated preview marker."""
    if value is None:
        return None
    text = json.dumps(value, ensure_ascii=False)
    if len(text.encode("utf-8")) <= MAX_RAW_BYTES:
        return value
    preview = text[:2000]
    return {
        "truncated": True,
        "original_chars": len(text),
        "preview": preview,
    }


class Observer:
    """Pure, deterministic converter: execution outcome -> Observation."""

    def observe_tool_result(
        self, *, task_id: str, tool_name: str, result: ToolResult
    ) -> Observation:
        """Convert a ``ToolResult`` into an observation.

        A failed ``ToolResult`` becomes a FAILED observation - never a
        successful one. Failures are data for the later stages to judge.
        """
        if result.ok:
            summary = f"Tool {tool_name!r} completed successfully"
            if isinstance(result.data, dict):
                keys = ", ".join(list(result.data)[:5])
                if keys:
                    summary = f"Tool {tool_name!r} returned: {keys}"
            return Observation(
                task_id=task_id,
                content_summary=summary[:500],
                raw=_bounded_raw(result.data),
                is_error=False,
                source=ObservationSource.TOOL,
                tool_name=tool_name,
            )
        error = (result.error or "tool failed").strip() or "tool failed"
        payload: JsonValue = {
            "tool_name": tool_name,
            "error": error[:500],
        }
        if result.error_code is not None:
            payload["error_code"] = result.error_code
        return Observation(
            task_id=task_id,
            content_summary=f"Tool {tool_name!r} failed: {error}"[:500],
            raw=_bounded_raw(payload),
            is_error=True,
            source=ObservationSource.TOOL,
            tool_name=tool_name,
            error_code=result.error_code or "TOOL_FAILURE",
        )

    def observe_reasoning_result(
        self, *, task_id: str, result: TaskResult
    ) -> Observation:
        """Convert a validated ``TaskResult`` into an observation.

        Only the structured result reaches workflow state - the raw model
        text outside the schema never does.
        """
        return Observation(
            task_id=task_id,
            content_summary=result.reason_summary[:500],
            raw=_bounded_raw(result.data),
            is_error=False,
            source=ObservationSource.REASONING,
        )

    def observe_rejected_decision(
        self, *, task_id: str, tool_name: str, detail: str, error_code: str
    ) -> Observation:
        """Observation for a tool proposal that failed deterministic checks."""
        detail = (detail or "tool decision rejected").strip()
        return Observation(
            task_id=task_id,
            content_summary=f"Tool decision rejected for {tool_name!r}: {detail}"[:500],
            raw=_bounded_raw(
                {"tool_name": tool_name, "detail": detail[:500], "error_code": error_code}
            ),
            is_error=True,
            source=ObservationSource.DECISION,
            tool_name=tool_name,
            error_code=error_code,
        )

    def observe_error(
        self, *, task_id: str, detail: str, error_code: str
    ) -> Observation:
        """Observation for a typed execution/LLM error."""
        detail = (detail or "execution error").strip()
        return Observation(
            task_id=task_id,
            content_summary=detail[:500],
            raw=_bounded_raw({"detail": detail[:500], "error_code": error_code}),
            is_error=True,
            source=ObservationSource.SYSTEM,
            error_code=error_code,
        )
