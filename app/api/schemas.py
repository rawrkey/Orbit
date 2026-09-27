"""HTTP request/response contracts for the API layer.

The API never exposes raw observation payloads, prompts, API keys, or
chain-of-thought - only structured status, decision summaries, digests,
and the user's own final answer.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.state.events import Event, EventType
from app.state.models import (
    FailureContext,
    JsonValue,
    Plan,
    Workflow,
    WorkflowState,
)

__all__ = [
    "CreateWorkflowRequest",
    "CreateWorkflowResponse",
    "EventView",
    "EventsResponse",
    "ObservationDigest",
    "RunWorkflowRequest",
    "RunWorkflowResponse",
    "WorkflowResultResponse",
    "WorkflowStatusResponse",
]


class CreateWorkflowRequest(BaseModel):
    """POST body for submitting a new goal.

    ``demo`` is the explicit offline-demo signal: when ``true``, THIS
    workflow runs on the scripted demo provider. It is optional and
    defaults to ``false`` - normal creation omits it and keeps the
    standard provider-selection rules. The response never reveals which
    provider was selected.
    """

    goal: str = Field(min_length=1, max_length=4000)
    demo: bool = False

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("goal must not be blank")
        return cleaned


class CreateWorkflowResponse(BaseModel):
    """Returned when a workflow is accepted."""

    workflow_id: str
    state: WorkflowState


class WorkflowStatusResponse(BaseModel):
    """Snapshot of a workflow's current status."""

    workflow_id: str
    goal: str
    state: WorkflowState
    replan_count: int
    created_at: datetime
    updated_at: datetime
    task_count: int
    observation_count: int
    event_count: int
    plan_version: int | None = None
    clarification_questions: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    failure: FailureContext | None = None


class ObservationDigest(BaseModel):
    """Observation WITHOUT its raw payload (bounded, leak-proof view)."""

    task_id: str
    content_summary: str
    source: str
    is_error: bool
    tool_name: str | None = None
    error_code: str | None = None


class WorkflowResultResponse(BaseModel):
    """Final result of a workflow: answer, plan, observation digests."""

    workflow_id: str
    state: WorkflowState
    final_answer: str | None = None
    plan: Plan | None = None
    observations: list[ObservationDigest] = Field(default_factory=list)
    synthesis: dict[str, JsonValue] | None = None
    failure: FailureContext | None = None

    @classmethod
    def from_workflow(cls, workflow: Workflow) -> WorkflowResultResponse:
        """Build the view from a Workflow - digests only, no raw payloads."""
        digests = [
            ObservationDigest(
                task_id=observation.task_id,
                content_summary=observation.content_summary,
                source=observation.source.value,
                is_error=observation.is_error,
                tool_name=observation.tool_name,
                error_code=observation.error_code,
            )
            for observation in workflow.observations
        ]
        return cls(
            workflow_id=workflow.id,
            state=workflow.state,
            final_answer=workflow.final_answer,
            plan=workflow.plan,
            observations=digests,
            synthesis=(
                workflow.synthesis.model_dump(mode="json")
                if workflow.synthesis is not None
                else None
            ),
            failure=workflow.failure,
        )


class EventView(BaseModel):
    """One event as served over HTTP (append-only decision record)."""

    seq: int
    timestamp: datetime
    type: EventType
    state: WorkflowState
    task_id: str | None = None
    decision_summary: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def from_event(cls, event: Event) -> EventView:
        return cls(
            seq=event.seq,
            timestamp=event.timestamp,
            type=event.type,
            state=event.state,
            task_id=event.task_id,
            decision_summary=event.decision_summary,
            payload=event.payload,
        )


class EventsResponse(BaseModel):
    """Event log page for a workflow."""

    workflow_id: str
    events: list[EventView]
    count: int


class RunWorkflowRequest(BaseModel):
    """POST body for /workflows/{id}/run - clarification answers only."""

    answers: dict[str, str] = Field(default_factory=dict)

    @field_validator("answers")
    @classmethod
    def answers_must_be_bounded(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 20:
            raise ValueError("at most 20 answers")
        for question, answer in value.items():
            if not question.strip():
                raise ValueError("answer keys must be non-empty questions")
            if not answer.strip():
                raise ValueError("answer values must be non-empty")
            if len(answer) > 500:
                raise ValueError("answers are limited to 500 characters")
        return value


class RunWorkflowResponse(BaseModel):
    """Outcome of driving a workflow (mirrors the adaptive run report)."""

    workflow_id: str
    success: bool
    final_state: WorkflowState
    executions: int
    validations: int
    replans: int
    termination: str
    failure: FailureContext | None = None
