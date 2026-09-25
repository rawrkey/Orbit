"""Structured-output schemas for LLM decision points.

Each schema is the typed contract for one LLM-controlled operation. The
application validates these before acting on them - free-form text is never
used for orchestration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.state.models import EvidenceRef, JsonValue

__all__ = [
    "ConstraintVerdict",
    "LLMDecision",
    "TaskResult",
    "ToolDecision",
    "ValidationProposal",
]


class LLMDecision(BaseModel):
    """Base shape for every LLM decision: a short, human-readable summary.

    Only a concise ``reason_summary`` is allowed - never chain-of-thought.
    """

    reason_summary: str = Field(min_length=1, max_length=500)


class ToolDecision(LLMDecision):
    """LLM proposal of which tool to call and with which arguments.

    This is a *proposal*: the application validates it against the tool
    registry and input schema before the executor runs anything.
    (The ToolSelector component that requests this arrives in a later phase.)
    """

    tool_name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class TaskResult(LLMDecision):
    """Structured outcome of a reasoning-only task (no tool involved).

    ``reason_summary`` is the concise summary of what was determined and
    ``data`` is any structured result. The Observer converts this into an
    Observation; raw model prose is never stored as workflow state.
    """

    data: JsonValue | None = None


class ConstraintVerdict(BaseModel):
    """LLM *proposal* for one constraint - Python has the final say.

    For constraints carrying a numeric ``rule``, the Validator independently
    extracts ``evidence_field`` from the referenced observation and applies
    the operator itself: the claimed ``outcome`` can never override a
    deterministic computation.
    """

    constraint_id: str = Field(min_length=1)
    outcome: Literal["satisfied", "violated", "insufficient"]
    evidence: list[EvidenceRef] = Field(default_factory=list)
    evidence_field: str | None = Field(
        default=None,
        max_length=200,
        description="Dot path into the observation's raw JSON holding the value",
    )
    note: str = Field(default="", max_length=300)


class ValidationProposal(LLMDecision):
    """LLM proposal of goal-satisfaction - validated before becoming a result.

    ``goal_outcome`` covers goals without extractable constraints; the
    Validator composes the final status deterministically from verified
    per-constraint outcomes first, then this assessment.
    """

    constraint_verdicts: list[ConstraintVerdict] = Field(default_factory=list)
    goal_outcome: Literal["satisfied", "not_satisfied", "insufficient"]
