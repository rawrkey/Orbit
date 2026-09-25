"""Phase 5 controller tests: validation, replanning, and plan replacement.

The Controller owns every mutation: these tests pin the legal/illegal
transitions, the replan budget (checked BEFORE mutation), plan versioning
with preserved completed work, and the exact event sequences. No API key.
"""

from __future__ import annotations

import pytest

from app.agent.controller import (
    ControllerError,
    MaxReplansExceededError,
    WorkflowController,
)
from app.agent.state_machine import IllegalTransitionError, is_legal
from app.state.events import EventType
from app.state.models import (
    Constraint,
    FailureContext,
    GoalAnalysis,
    Observation,
    ObservationSource,
    Plan,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    WorkflowState,
)
from app.state.store import InMemoryStateStore

S = WorkflowState
GOAL = "Plan a 3-day Mumbai trip for 2 people under 8000 INR"

ANALYSIS = GoalAnalysis(
    objective="Plan a 3-day Mumbai trip under 8000 INR",
    reason_summary="Duration and budget are explicit.",
    constraints=[
        Constraint(
            id="c1", name="budget", type="BUDGET", description="Under 8000 INR"
        )
    ],
)

PLAN = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Compute total",
            description="Compute the trip total",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="total computed",
        ),
        Task(
            id="t2",
            title="Draft itinerary",
            description="Draft the day-by-day itinerary",
            success_criteria="itinerary drafted",
        ),
    ],
)

PLAN_V2 = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Compute cheaper total",
            description="Recompute the total with cheaper stays",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="revised total computed",
        ),
        PLAN.tasks[1],  # unchanged definition
        Task(
            id="t3",
            title="Check transit",
            description="Check local transit options",
            success_criteria="transit checked",
        ),
    ],
)


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


def ready(controller: WorkflowController) -> str:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, PLAN)
    return workflow.id


def awaiting(controller: WorkflowController) -> str:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    for task in PLAN.tasks:
        controller.begin_task(workflow_id, task.id)
        controller.record_observation(
            workflow_id,
            Observation(
                task_id=task.id,
                content_summary="Done.",
                source=ObservationSource.REASONING,
            ),
        )
        controller.complete_task(workflow_id, task.id)
    controller.finish_execution(workflow_id)
    return workflow_id


def validating(controller: WorkflowController) -> str:
    workflow_id = awaiting(controller)
    controller.begin_validation(workflow_id)
    return workflow_id


def replanning(controller: WorkflowController) -> str:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, fail_result())
    controller.begin_replanning(workflow_id)
    return workflow_id


def pass_result() -> ValidationResult:
    return ValidationResult(
        status=ValidationStatus.PASS,
        reason_summary="Budget respected: 7500 <= 8000.",
        satisfied_constraints=["c1"],
        replanning_required=False,
    )


def fail_result() -> ValidationResult:
    return ValidationResult(
        status=ValidationStatus.FAIL,
        reason_summary="Trip total 9000 INR exceeds the 8000 INR budget.",
        violated_constraints=["c1"],
        replanning_required=True,
    )


def insufficient_result() -> ValidationResult:
    return ValidationResult(
        status=ValidationStatus.INSUFFICIENT,
        reason_summary="Price evidence covers only 3 of 10 bookings.",
        unresolved_constraints=["c1"],
        replanning_required=True,
    )


def event_types(controller: WorkflowController, workflow_id: str) -> list[EventType]:
    return [e.type for e in controller.events(workflow_id)]


def one_event(controller, workflow_id, event_type):
    matches = [e for e in controller.events(workflow_id) if e.type is event_type]
    assert len(matches) == 1, f"expected exactly one {event_type}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# begin_validation
# ---------------------------------------------------------------------------


def test_begin_validation_moves_to_validating_and_emits(controller) -> None:
    workflow_id = awaiting(controller)
    updated = controller.begin_validation(workflow_id)
    assert updated.state is S.VALIDATING
    assert event_types(controller, workflow_id)[-2:] == [
        EventType.STATE_CHANGED,
        EventType.VALIDATION_STARTED,
    ]
    started = one_event(controller, workflow_id, EventType.VALIDATION_STARTED)
    assert started.state is S.VALIDATING
    assert started.payload["replan_count"] == 0


def test_begin_validation_illegal_before_execution(controller) -> None:
    workflow_id = ready(controller)
    events_before = len(controller.events(workflow_id))
    with pytest.raises(IllegalTransitionError):
        controller.begin_validation(workflow_id)
    assert controller.get_workflow(workflow_id).state is S.READY
    assert len(controller.events(workflow_id)) == events_before


# ---------------------------------------------------------------------------
# record_validation
# ---------------------------------------------------------------------------


def test_record_pass_stores_result_and_emits_passed(controller) -> None:
    workflow_id = validating(controller)
    updated = controller.record_validation(workflow_id, pass_result())
    assert updated.state is S.VALIDATING  # recording is not a transition
    assert updated.validation is not None
    assert updated.validation.status is ValidationStatus.PASS
    event = one_event(controller, workflow_id, EventType.VALIDATION_PASSED)
    assert event.payload["status"] == "PASS"
    assert event.payload["satisfied_constraints"] == ["c1"]


def test_record_fail_stores_result_and_emits_failed(controller) -> None:
    workflow_id = validating(controller)
    updated = controller.record_validation(workflow_id, fail_result())
    assert updated.validation is not None
    assert updated.validation.status is ValidationStatus.FAIL
    assert updated.validation.replanning_required is True
    event = one_event(controller, workflow_id, EventType.VALIDATION_FAILED)
    assert event.payload["status"] == "FAIL"
    assert event.payload["violated_constraints"] == ["c1"]


def test_record_validation_rejected_outside_validating(controller) -> None:
    workflow_id = awaiting(controller)  # not VALIDATING yet
    events_before = len(controller.events(workflow_id))
    with pytest.raises(IllegalTransitionError):
        controller.record_validation(workflow_id, pass_result())
    assert controller.get_workflow(workflow_id).validation is None
    assert len(controller.events(workflow_id)) == events_before


def test_record_validation_rejects_pass_with_violated_constraints(
    controller,
) -> None:
    workflow_id = validating(controller)
    poisoned = ValidationResult(
        status=ValidationStatus.PASS,
        reason_summary="Claims pass.",
        violated_constraints=["c1"],
    )
    with pytest.raises(ControllerError, match="PASS is impossible"):
        controller.record_validation(workflow_id, poisoned)
    assert controller.get_workflow(workflow_id).validation is None
    assert (
        EventType.VALIDATION_PASSED
        not in event_types(controller, workflow_id)
    )


# ---------------------------------------------------------------------------
# begin_replanning
# ---------------------------------------------------------------------------


def test_begin_replanning_moves_and_consumes_budget(controller) -> None:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, fail_result())
    updated = controller.begin_replanning(workflow_id)
    assert updated.state is S.REPLANNING
    assert updated.replan_count == 1
    assert event_types(controller, workflow_id)[-3:] == [
        EventType.STATE_CHANGED,  # VALIDATING -> REPLANNING
        EventType.REPLANNING_STARTED,
        EventType.WORKFLOW_REPLANNED,
    ]
    started = one_event(controller, workflow_id, EventType.REPLANNING_STARTED)
    assert started.payload["validation_status"] == "FAIL"
    assert started.payload["limit"] == 2
    assert "9000" in started.payload["reason"]


def test_begin_replanning_requires_a_stored_validation(controller) -> None:
    workflow_id = validating(controller)  # VALIDATING, nothing recorded
    events_before = len(controller.events(workflow_id))
    with pytest.raises(ControllerError, match="without a stored validation"):
        controller.begin_replanning(workflow_id)
    assert controller.get_workflow(workflow_id).state is S.VALIDATING
    assert len(controller.events(workflow_id)) == events_before


def test_begin_replanning_illegal_outside_validating(controller) -> None:
    workflow_id = awaiting(controller)
    with pytest.raises(IllegalTransitionError):
        controller.begin_replanning(workflow_id)
    assert controller.get_workflow(workflow_id).state is S.AWAITING_VALIDATION


def test_replan_budget_checked_before_any_mutation() -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=0)
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, fail_result())
    events_before = len(controller.events(workflow_id))
    with pytest.raises(MaxReplansExceededError) as exc:
        controller.begin_replanning(workflow_id)
    assert exc.value.limit == 0
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.VALIDATING  # never entered REPLANNING
    assert workflow.replan_count == 0
    assert workflow.validation is not None  # verdict kept for failure record
    assert len(controller.events(workflow_id)) == events_before


# ---------------------------------------------------------------------------
# replace_plan
# ---------------------------------------------------------------------------


def test_replace_plan_versions_history_and_merges_states(controller) -> None:
    workflow_id = replanning(controller)
    workflow = controller.get_workflow(workflow_id)
    workflow.task_states["t1"] = TaskStatus.COMPLETED
    workflow.task_states["t2"] = TaskStatus.COMPLETED
    controller._store.save_workflow(workflow)

    updated = controller.replace_plan(
        workflow_id, PLAN_V2, reason="Validation FAIL: budget exceeded."
    )

    assert updated.state is S.PLANNING
    assert updated.plan is not None
    assert updated.plan.version == 2  # old version + 1
    # t1 redefined -> reset; t2 unchanged -> preserved; t3 new -> pending.
    assert updated.task_states["t1"] is TaskStatus.PENDING
    assert updated.task_states["t2"] is TaskStatus.COMPLETED
    assert updated.task_states["t3"] is TaskStatus.PENDING
    # Append-only history keeps the superseded plan + reason.
    assert len(updated.plan_history) == 1
    assert updated.plan_history[0].plan.version == 1
    assert updated.plan_history[0].reason == "Validation FAIL: budget exceeded."


def test_replace_plan_drops_orphaned_task_states(controller) -> None:
    workflow_id = replanning(controller)
    workflow = controller.get_workflow(workflow_id)
    workflow.task_states = {"t1": TaskStatus.COMPLETED, "t2": TaskStatus.COMPLETED}
    controller._store.save_workflow(workflow)
    only_t1 = Plan(
        tasks=[
            Task(
                id="t1",
                title="Compute total",
                description="Compute the trip total",
                requires_tool=True,
                tool_hint="calculator",
                constraint_ids=["c1"],
                success_criteria="total computed",
            )
        ]
    )
    updated = controller.replace_plan(
        workflow_id, only_t1, reason="Drop itinerary task."
    )
    assert "t2" not in updated.task_states
    assert updated.task_states["t1"] is TaskStatus.COMPLETED
    assert updated.plan is not None and [t.id for t in updated.plan.tasks] == ["t1"]


def test_replace_plan_event_sequence(controller) -> None:
    workflow_id = replanning(controller)
    controller.replace_plan(workflow_id, PLAN_V2, reason="Budget exceeded.")
    assert event_types(controller, workflow_id)[-3:] == [
        EventType.REPLAN_PROPOSED,
        EventType.REPLAN_ACCEPTED,
        EventType.STATE_CHANGED,  # REPLANNING -> PLANNING
    ]
    accepted = one_event(controller, workflow_id, EventType.REPLAN_ACCEPTED)
    assert accepted.payload["version"] == 2
    assert accepted.payload["previous_version"] == 1
    proposed = one_event(controller, workflow_id, EventType.REPLAN_PROPOSED)
    assert proposed.payload["task_ids"] == ["t1", "t2", "t3"]


def test_replace_plan_rejects_identical_plan(controller) -> None:
    workflow_id = replanning(controller)
    events_before = len(controller.events(workflow_id))
    with pytest.raises(ControllerError, match="identical"):
        controller.replace_plan(workflow_id, PLAN, reason="No real change.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.REPLANNING
    assert workflow.plan is not None and workflow.plan.version == 1
    assert workflow.plan_history == []
    # rejection is diagnosed with a single append-only event, no state change
    events = controller.events(workflow_id)
    assert len(events) == events_before + 1
    assert events[-1].type is EventType.REPLAN_REJECTED
    assert events[-1].payload["reason"] == "identical_plan"


def test_replace_plan_illegal_outside_replanning(controller) -> None:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, fail_result())
    events_before = len(controller.events(workflow_id))
    with pytest.raises(IllegalTransitionError):
        controller.replace_plan(workflow_id, PLAN_V2, reason="Premature.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.VALIDATING
    assert workflow.plan is not None and workflow.plan.version == 1
    events = controller.events(workflow_id)
    assert len(events) == events_before + 1
    assert events[-1].type is EventType.REPLAN_REJECTED
    assert events[-1].payload["reason"] == "invalid_state"


def test_plan_versions_accumulate_across_replans(controller) -> None:
    workflow_id = replanning(controller)
    workflow = controller.get_workflow(workflow_id)
    workflow.task_states["t1"] = TaskStatus.COMPLETED
    workflow.task_states["t2"] = TaskStatus.COMPLETED
    controller._store.save_workflow(workflow)

    controller.replace_plan(workflow_id, PLAN_V2, reason="First replan.")
    revised = controller.get_workflow(workflow_id)
    assert revised.plan is not None and revised.plan.version == 2
    controller.record_plan(workflow_id, revised.plan)

    # Second round: READY -> ... -> VALIDATING -> REPLANNING -> replace.
    controller.begin_execution(workflow_id)
    for task in revised.plan.tasks:
        if controller.get_workflow(workflow_id).task_states[task.id] is TaskStatus.COMPLETED:
            continue
        controller.begin_task(workflow_id, task.id)
        controller.record_observation(
            workflow_id,
            Observation(task_id=task.id, content_summary="Done."),
        )
        controller.complete_task(workflow_id, task.id)
    controller.finish_execution(workflow_id)
    controller.begin_validation(workflow_id)
    controller.record_validation(workflow_id, fail_result())
    controller.begin_replanning(workflow_id)

    third = Plan(tasks=[PLAN.tasks[1]], version=1)
    controller.replace_plan(workflow_id, third, reason="Second replan.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.plan is not None and workflow.plan.version == 3
    assert [revision.plan.version for revision in workflow.plan_history] == [1, 2]
    assert workflow.replan_count == 2


# ---------------------------------------------------------------------------
# finish_with_validation
# ---------------------------------------------------------------------------


def test_finish_with_validation_completes_on_pass(controller) -> None:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, pass_result())
    updated = controller.finish_with_validation(workflow_id, "Trip under budget.")
    assert updated.state is S.COMPLETED
    assert updated.final_answer == "Trip under budget."
    assert event_types(controller, workflow_id)[-2:] == [
        EventType.STATE_CHANGED,
        EventType.WORKFLOW_COMPLETED,
    ]


def test_finish_with_validation_rejects_non_pass(controller) -> None:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, fail_result())
    with pytest.raises(ControllerError, match="not PASS"):
        controller.finish_with_validation(workflow_id, "Done?")
    assert controller.get_workflow(workflow_id).state is S.VALIDATING


def test_finish_with_validation_requires_stored_result(controller) -> None:
    workflow_id = validating(controller)
    with pytest.raises(ControllerError, match="before validation"):
        controller.finish_with_validation(workflow_id, "Done?")
    assert controller.get_workflow(workflow_id).state is S.VALIDATING


def test_finish_with_validation_illegal_outside_validating(controller) -> None:
    workflow_id = awaiting(controller)
    with pytest.raises(IllegalTransitionError):
        controller.finish_with_validation(workflow_id, "Done?")
    assert controller.get_workflow(workflow_id).state is S.AWAITING_VALIDATION


def test_completed_is_unreachable_without_validation() -> None:
    # The transition table itself: execution never completes a workflow.
    assert not is_legal(S.AWAITING_VALIDATION, S.COMPLETED)
    assert not is_legal(S.EXECUTING, S.COMPLETED)
    assert not is_legal(S.REPLANNING, S.COMPLETED)
    assert is_legal(S.VALIDATING, S.COMPLETED)


# ---------------------------------------------------------------------------
# fail_unsatisfied_goal
# ---------------------------------------------------------------------------


def test_fail_unsatisfied_goal_fails_with_evidence(controller) -> None:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, fail_result())
    failure = FailureContext(
        stage=S.VALIDATING,
        error_code="GOAL_UNSATISFIED",
        detail="Budget constraint violated after replanning.",
    )
    updated = controller.fail_unsatisfied_goal(workflow_id, failure)
    assert updated.state is S.FAILED
    assert updated.failure is not None
    assert updated.failure.error_code == "GOAL_UNSATISFIED"
    assert event_types(controller, workflow_id)[-1] is EventType.WORKFLOW_FAILED


def test_fail_unsatisfied_goal_rejects_pass(controller) -> None:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, pass_result())
    failure = FailureContext(stage=S.VALIDATING, error_code="X", detail="y")
    with pytest.raises(ControllerError, match="when validation passed"):
        controller.fail_unsatisfied_goal(workflow_id, failure)
    assert controller.get_workflow(workflow_id).state is S.VALIDATING


def test_fail_unsatisfied_goal_requires_stored_result(controller) -> None:
    workflow_id = validating(controller)
    failure = FailureContext(stage=S.VALIDATING, error_code="X", detail="y")
    with pytest.raises(ControllerError, match="before validation"):
        controller.fail_unsatisfied_goal(workflow_id, failure)
    assert controller.get_workflow(workflow_id).state is S.VALIDATING


def test_fail_unsatisfied_goal_illegal_outside_validating(controller) -> None:
    workflow_id = awaiting(controller)
    failure = FailureContext(stage=S.VALIDATING, error_code="X", detail="y")
    with pytest.raises(IllegalTransitionError):
        controller.fail_unsatisfied_goal(workflow_id, failure)
    assert controller.get_workflow(workflow_id).state is S.AWAITING_VALIDATION


def test_insufficient_result_also_triggers_failure_path(controller) -> None:
    workflow_id = validating(controller)
    controller.record_validation(workflow_id, insufficient_result())
    controller.begin_replanning(workflow_id)  # INSUFFICIENT replans too
    assert controller.get_workflow(workflow_id).state is S.REPLANNING
