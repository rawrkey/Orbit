"""Append-only event system.

Events are the single source of truth for frontend visualization. They carry
structured decision summaries only - never hidden chain-of-thought.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.state.models import JsonValue, WorkflowState, utcnow


class EventType(str, Enum):
    """Explicit event types emitted by the workflow lifecycle."""

    WORKFLOW_STARTED = "workflow.started"
    GOAL_ANALYZED = "goal.analyzed"
    CLARIFICATION_REQUESTED = "clarification.requested"
    CLARIFICATION_ANSWERED = "clarification.answered"
    WORKFLOW_PLANNED = "workflow.planned"
    PLAN_VALIDATION_FAILED = "plan.validation_failed"
    STATE_CHANGED = "state.changed"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_FINISHED = "execution.finished"
    EXECUTION_FAILED = "execution.failed"
    TASK_STARTED = "task.started"
    TOOL_DECISION_PROPOSED = "tool.decision_proposed"
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    OBSERVATION_CREATED = "observation.created"
    VALIDATION_STARTED = "validation.started"
    VALIDATION_PASSED = "validation.passed"
    VALIDATION_FAILED = "validation.failed"
    REPLANNING_STARTED = "replanning.started"
    REPLAN_PROPOSED = "replan.proposed"
    REPLAN_ACCEPTED = "replan.accepted"
    REPLAN_REJECTED = "replan.rejected"
    REPLANNING_EXHAUSTED = "replanning.exhausted"
    WORKFLOW_REPLANNED = "workflow.replanned"
    TASK_COMPLETED = "task.completed"
    SYNTHESIS_COMPLETED = "synthesis.completed"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"


class Event(BaseModel):
    """One immutable, serializable workflow event.

    ``decision_summary`` is a short human-readable explanation such as
    "Action: search accommodation". Raw reasoning is never stored.
    """

    seq: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=utcnow)
    workflow_id: str = Field(min_length=1)
    type: EventType
    state: WorkflowState
    task_id: str | None = None
    decision_summary: str | None = Field(default=None, max_length=500)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class EventLog:
    """Append-only, per-workflow sequence of events.

    Sequence numbers are assigned on append, start at 0, and never change.
    The log exposes no API that can remove or rewrite an event.
    """

    def __init__(self, workflow_id: str) -> None:
        if not workflow_id:
            raise ValueError("workflow_id must be a non-empty string")
        self._workflow_id = workflow_id
        self._events: list[Event] = []

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    def append(
        self,
        *,
        event_type: EventType,
        state: WorkflowState,
        task_id: str | None = None,
        decision_summary: str | None = None,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> Event:
        """Append an event and return it. Sequence number is assigned here."""
        event = Event(
            seq=len(self._events),
            workflow_id=self._workflow_id,
            type=event_type,
            state=state,
            task_id=task_id,
            decision_summary=decision_summary,
            payload=dict(payload) if payload else {},
        )
        self._events.append(event)
        return event

    def all(self) -> tuple[Event, ...]:
        """Immutable snapshot of every event, in order."""
        return tuple(self._events)

    def load(self, events: Sequence[Event]) -> None:
        """Restore persisted events (persistence backends only).

        Keeps each event's original ``seq`` and ``timestamp``. The log must
        be empty, every event must belong to this workflow, and sequence
        numbers must be contiguous from 0 - append-only semantics hold.
        """
        if self._events:
            raise ValueError("cannot load events into a non-empty log")
        expected = 0
        for event in events:
            if event.workflow_id != self._workflow_id:
                raise ValueError(
                    f"event for {event.workflow_id!r} does not belong to "
                    f"{self._workflow_id!r}"
                )
            if event.seq != expected:
                raise ValueError(
                    f"event sequence must be contiguous from 0 "
                    f"(expected {expected}, got {event.seq})"
                )
            self._events.append(event)
            expected += 1

    def after(self, seq: int) -> list[Event]:
        """Events strictly after ``seq`` (for streaming/polling)."""
        return [event for event in self._events if event.seq > seq]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(tuple(self._events))

    def __repr__(self) -> str:
        return f"EventLog(workflow_id={self._workflow_id!r}, events={len(self._events)})"
