"""Explicit workflow state machine.

The transition table is the single source of truth for which state changes
are legal. Only the Controller performs transitions; this module defines the
rules and raises ``IllegalTransitionError`` on any violation.
"""

from __future__ import annotations

from app.state.models import WorkflowState

__all__ = [
    "IllegalTransitionError",
    "LEGAL_TRANSITIONS",
    "assert_legal",
    "is_legal",
    "is_terminal",
    "legal_targets",
]


class IllegalTransitionError(Exception):
    """Raised when a state transition is not permitted. Never mutates state."""

    def __init__(
        self,
        source: WorkflowState,
        target: WorkflowState,
        reason: str | None = None,
    ) -> None:
        self.source = source
        self.target = target
        message = f"illegal workflow transition: {source.value} -> {target.value}"
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)


# The complete transition table. Missing source keys would be a bug; tests
# assert that every state appears here.
#
# Normal flow:
#   RECEIVED -> ANALYZING -> PLANNING -> READY -> EXECUTING
#   -> ... -> AWAITING_VALIDATION -> VALIDATING -> COMPLETED
# Observing/validating loop (later phases):
#   EXECUTING -> OBSERVING -> VALIDATING -> EXECUTING ...
# Replan flow:
#   VALIDATING -> REPLANNING -> PLANNING -> READY -> EXECUTING
#
# Phase 4 note: a full execution pass ends at AWAITING_VALIDATION - never at
# COMPLETED. Executing every planned task does NOT prove the user's goal was
# satisfied; the future validation stage owns the COMPLETED decision.
# AWAITING_VALIDATION may still move to FAILED for unrecoverable errors.
LEGAL_TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.RECEIVED: frozenset(
        {WorkflowState.ANALYZING, WorkflowState.FAILED}
    ),
    WorkflowState.ANALYZING: frozenset(
        {
            WorkflowState.PLANNING,
            WorkflowState.AWAITING_CLARIFICATION,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.AWAITING_CLARIFICATION: frozenset(
        {WorkflowState.PLANNING, WorkflowState.FAILED}
    ),
    WorkflowState.PLANNING: frozenset(
        {WorkflowState.READY, WorkflowState.FAILED}
    ),
    WorkflowState.READY: frozenset(
        {WorkflowState.EXECUTING, WorkflowState.FAILED}
    ),
    WorkflowState.EXECUTING: frozenset(
        {
            WorkflowState.OBSERVING,
            WorkflowState.AWAITING_VALIDATION,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.OBSERVING: frozenset(
        {WorkflowState.VALIDATING, WorkflowState.FAILED}
    ),
    WorkflowState.VALIDATING: frozenset(
        {
            WorkflowState.EXECUTING,
            WorkflowState.REPLANNING,
            WorkflowState.COMPLETED,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.AWAITING_VALIDATION: frozenset(
        {WorkflowState.VALIDATING, WorkflowState.FAILED}
    ),
    WorkflowState.REPLANNING: frozenset(
        {WorkflowState.PLANNING, WorkflowState.FAILED}
    ),
    WorkflowState.COMPLETED: frozenset(),
    WorkflowState.FAILED: frozenset(),
}


def legal_targets(source: WorkflowState) -> frozenset[WorkflowState]:
    """States reachable from ``source`` (empty for terminal states)."""
    return LEGAL_TRANSITIONS[source]


def is_legal(source: WorkflowState, target: WorkflowState) -> bool:
    """True if ``source -> target`` is a permitted transition."""
    return target in LEGAL_TRANSITIONS[source]


def is_terminal(state: WorkflowState) -> bool:
    """True for COMPLETED and FAILED: no further transitions allowed."""
    return not LEGAL_TRANSITIONS[state]


def assert_legal(
    source: WorkflowState, target: WorkflowState, reason: str | None = None
) -> None:
    """Raise ``IllegalTransitionError`` unless the transition is permitted."""
    if source == target:
        raise IllegalTransitionError(source, target, "self-transition not allowed")
    if not is_legal(source, target):
        raise IllegalTransitionError(source, target, reason)
