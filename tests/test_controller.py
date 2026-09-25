"""Controller skeleton tests: transitions, replan limits, events, failures."""

from __future__ import annotations

import pytest

from app.agent.controller import (
    MaxReplansExceededError,
    WorkflowController,
)
from app.agent.state_machine import IllegalTransitionError
from app.state.events import EventType
from app.state.models import FailureContext, WorkflowState
from app.state.store import InMemoryStateStore, WorkflowNotFoundError

S = WorkflowState


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


def create(controller: WorkflowController, goal: str = "Plan a trip") -> str:
    return controller.create_workflow(goal).id


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_create_workflow_starts_in_received_with_started_event(controller) -> None:
    workflow = controller.create_workflow("Plan a 3-day Mumbai trip")
    assert workflow.state == S.RECEIVED
    assert workflow.event_count == 1
    events = controller.events(workflow.id)
    assert events[0].type == EventType.WORKFLOW_STARTED
    assert events[0].state == S.RECEIVED
    assert events[0].seq == 0


def test_create_workflow_rejects_blank_goal(controller) -> None:
    with pytest.raises(ValueError):
        controller.create_workflow("   ")


def test_unknown_workflow_raises(controller) -> None:
    with pytest.raises(WorkflowNotFoundError):
        controller.get_workflow("does-not-exist")


def test_negative_max_replans_is_rejected() -> None:
    with pytest.raises(ValueError):
        WorkflowController(InMemoryStateStore(), max_replans=-1)


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def test_legal_transition_updates_state_and_emits_event(controller) -> None:
    workflow_id = create(controller)
    updated = controller.transition(workflow_id, S.ANALYZING)
    assert updated.state == S.ANALYZING

    state_events = [
        e for e in controller.events(workflow_id) if e.type == EventType.STATE_CHANGED
    ]
    assert len(state_events) == 1
    assert state_events[0].payload == {"from": "RECEIVED", "to": "ANALYZING"}


def test_illegal_transition_raises_and_changes_nothing(controller) -> None:
    workflow_id = create(controller)
    events_before = len(controller.events(workflow_id))

    with pytest.raises(IllegalTransitionError):
        controller.transition(workflow_id, S.COMPLETED)

    assert controller.get_workflow(workflow_id).state == S.RECEIVED
    assert len(controller.events(workflow_id)) == events_before


def test_terminal_state_protection_via_controller(controller) -> None:
    workflow_id = create(controller)
    with pytest.raises(IllegalTransitionError):  # cannot skip to COMPLETED
        controller.transition(workflow_id, S.COMPLETED)
    controller.transition(workflow_id, S.ANALYZING)
    failure = FailureContext(
        stage=S.ANALYZING, error_code="LLM_TIMEOUT", detail="provider timed out"
    )
    controller.fail(workflow_id, failure)
    assert controller.get_workflow(workflow_id).state == S.FAILED

    with pytest.raises(IllegalTransitionError):
        controller.transition(workflow_id, S.PLANNING)
    assert controller.get_workflow(workflow_id).state == S.FAILED


def test_happy_path_walk_through(controller) -> None:
    workflow_id = create(controller)
    for state in (S.ANALYZING, S.PLANNING, S.READY, S.EXECUTING, S.OBSERVING, S.VALIDATING):
        controller.transition(workflow_id, state)
    assert controller.get_workflow(workflow_id).state == S.VALIDATING


# ---------------------------------------------------------------------------
# Completion and failure
# ---------------------------------------------------------------------------


def test_complete_sets_answer_and_emits_completed(controller) -> None:
    workflow_id = create(controller)
    for state in (S.ANALYZING, S.PLANNING, S.READY, S.EXECUTING, S.OBSERVING, S.VALIDATING):
        controller.transition(workflow_id, state)

    finished = controller.complete(workflow_id, "Here is your 3-day plan.")
    assert finished.state == S.COMPLETED
    assert finished.final_answer == "Here is your 3-day plan."
    assert any(e.type == EventType.WORKFLOW_COMPLETED for e in controller.events(workflow_id))


def test_complete_rejects_empty_answer_and_partial_state(controller) -> None:
    workflow_id = create(controller)
    with pytest.raises(ValueError):
        controller.complete(workflow_id, "  ")


def test_complete_from_wrong_state_does_not_mutate(controller) -> None:
    workflow_id = create(controller)
    with pytest.raises(IllegalTransitionError):
        controller.complete(workflow_id, "answer")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.final_answer is None
    assert workflow.state == S.RECEIVED


def test_fail_records_structured_failure(controller) -> None:
    workflow_id = create(controller)
    failure = FailureContext(
        stage=S.RECEIVED, error_code="MISSING_INFO", detail="no budget given"
    )
    failed = controller.fail(workflow_id, failure)
    assert failed.state == S.FAILED
    assert failed.failure is not None
    assert failed.failure.error_code == "MISSING_INFO"

    failure_events = [
        e for e in controller.events(workflow_id) if e.type == EventType.WORKFLOW_FAILED
    ]
    assert len(failure_events) == 1
    assert failure_events[0].payload["error_code"] == "MISSING_INFO"


# ---------------------------------------------------------------------------
# Replan accounting
# ---------------------------------------------------------------------------


def test_record_replan_increments_up_to_the_limit(controller) -> None:
    workflow_id = create(controller)
    assert controller.record_replan(workflow_id) == 1
    assert controller.record_replan(workflow_id) == 2
    assert controller.get_workflow(workflow_id).replan_count == 2

    replanned = [
        e for e in controller.events(workflow_id)
        if e.type == EventType.WORKFLOW_REPLANNED
    ]
    assert [e.payload["replan_count"] for e in replanned] == [1, 2]


def test_record_replan_beyond_limit_raises_without_mutation(controller) -> None:
    workflow_id = create(controller)
    controller.record_replan(workflow_id)
    controller.record_replan(workflow_id)
    events_before = len(controller.events(workflow_id))

    with pytest.raises(MaxReplansExceededError) as exc_info:
        controller.record_replan(workflow_id)

    assert exc_info.value.limit == 2
    assert exc_info.value.replan_count == 2
    workflow = controller.get_workflow(workflow_id)
    assert workflow.replan_count == 2  # unchanged
    assert len(controller.events(workflow_id)) == events_before  # unchanged


def test_max_replans_property(controller) -> None:
    assert controller.max_replans == 2


# ---------------------------------------------------------------------------
# Event access
# ---------------------------------------------------------------------------


def test_events_are_sequenced_and_filterable(controller) -> None:
    workflow_id = create(controller)
    controller.transition(workflow_id, S.ANALYZING)
    controller.transition(workflow_id, S.PLANNING)

    all_events = controller.events(workflow_id)
    assert [e.seq for e in all_events] == [0, 1, 2]
    assert [e.seq for e in controller.events_after(workflow_id, 0)] == [1, 2]
    assert controller.events_after(workflow_id, 2) == []


def test_events_for_unknown_workflow_raise(controller) -> None:
    with pytest.raises(WorkflowNotFoundError):
        controller.events("missing")


def test_emit_event_appends_domain_event(controller) -> None:
    workflow_id = create(controller)
    controller.transition(workflow_id, S.ANALYZING)
    event = controller.emit_event(
        workflow_id,
        EventType.GOAL_ANALYZED,
        decision_summary="Constraints: budget 8000 INR, 3 days.",
        payload={"constraint_count": 1},
    )
    assert event.seq == 2
    assert event.decision_summary is not None
    assert controller.get_workflow(workflow_id).event_count == 3


def test_can_transition_is_non_raising(controller) -> None:
    workflow_id = create(controller)
    assert controller.can_transition(workflow_id, S.ANALYZING) is True
    assert controller.can_transition(workflow_id, S.COMPLETED) is False
