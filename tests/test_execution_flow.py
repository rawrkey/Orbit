"""Phase 4 controller tests + full offline integration flow.

Covers the execution-facing controller methods (legal/illegal transitions,
events, task lifecycle) and the end-to-end run from Goal through
AWAITING_VALIDATION with an exact event sequence. No API key, no network.
"""

from __future__ import annotations

import pytest

from app.agent.controller import ControllerError, WorkflowController
from app.agent.executor import Executor
from app.agent.state_machine import IllegalTransitionError
from app.config.settings import load_settings
from app.llm.mock import MockProvider
from app.llm.schemas import TaskResult, ToolDecision
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
    WorkflowState,
)
from app.state.store import InMemoryStateStore
from app.tools.builtin import create_default_registry

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
    is_feasible=True,
)

PLAN = Plan(
    tasks=[
        Task(
            id="t1",
            title="Calculate per-person budget",
            description="Compute the per-person budget share",
            requires_tool=True,
            tool_hint="calculator",
            expected_output="numeric share",
            constraint_ids=["c1"],
            success_criteria="share computed",
        ),
        Task(
            id="t2",
            title="Draft itinerary",
            description="Draft the day-by-day itinerary",
            success_criteria="3 days drafted",
        ),
    ]
)

CALC_DECISION = ToolDecision(
    tool_name="calculator",
    arguments={"expression": "8000/2"},
    reason_summary="Compute the per-person budget share.",
)
REASONING_RESULT = TaskResult(
    reason_summary="Itinerary covers 3 days in South Mumbai.",
    data={"days": 3},
)


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


@pytest.fixture
def registry():
    return create_default_registry(load_settings({}))


def ready(controller: WorkflowController, plan: Plan = PLAN) -> str:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, plan)
    return workflow.id


def event_types(controller: WorkflowController, workflow_id: str) -> list[EventType]:
    return [e.type for e in controller.events(workflow_id)]


def one_event(controller, workflow_id, event_type) -> object:
    matches = [e for e in controller.events(workflow_id) if e.type is event_type]
    assert len(matches) == 1, f"expected exactly one {event_type}"
    return matches[0]


# ---------------------------------------------------------------------------
# Controller: begin_execution / begin_task
# ---------------------------------------------------------------------------


def test_begin_execution_moves_ready_to_executing(controller) -> None:
    workflow_id = ready(controller)
    updated = controller.begin_execution(workflow_id)
    assert updated.state is S.EXECUTING
    assert event_types(controller, workflow_id)[-2:] == [
        EventType.STATE_CHANGED,
        EventType.EXECUTION_STARTED,
    ]
    started = one_event(controller, workflow_id, EventType.EXECUTION_STARTED)
    assert started.payload["tasks"] == 2
    assert started.state is S.EXECUTING


def test_begin_execution_requires_a_plan(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    events_before = len(controller.events(workflow.id))
    with pytest.raises(ControllerError, match="without a plan"):
        controller.begin_execution(workflow.id)
    assert controller.get_workflow(workflow.id).state is S.RECEIVED
    assert len(controller.events(workflow.id)) == events_before


def test_begin_execution_illegal_when_not_ready(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)  # EXECUTING, has a plan
    events_before = len(controller.events(workflow_id))
    with pytest.raises(IllegalTransitionError):
        controller.begin_execution(workflow_id)
    assert controller.get_workflow(workflow_id).state is S.EXECUTING
    assert len(controller.events(workflow_id)) == events_before


def test_begin_task_starts_pending_task_and_emits(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    updated = controller.begin_task(workflow_id, "t1")

    assert updated.task_states["t1"] is TaskStatus.RUNNING
    started = one_event(controller, workflow_id, EventType.TASK_STARTED)
    assert started.task_id == "t1"
    assert started.payload == {"index": 1, "total": 2, "requires_tool": True}
    assert "task 1/2" in started.decision_summary.lower()


def test_begin_task_rejects_non_executing_workflow(controller) -> None:
    workflow_id = ready(controller)  # READY
    with pytest.raises(ControllerError, match="while READY"):
        controller.begin_task(workflow_id, "t1")
    assert controller.get_workflow(workflow_id).task_states["t1"] is TaskStatus.PENDING


def test_begin_task_rejects_double_start(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t1")
    events_before = len(controller.events(workflow_id))
    with pytest.raises(ControllerError, match="RUNNING"):
        controller.begin_task(workflow_id, "t1")
    assert controller.get_workflow(workflow_id).task_states["t1"] is TaskStatus.RUNNING
    assert len(controller.events(workflow_id)) == events_before


def test_begin_task_rejects_unknown_task(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    with pytest.raises(ControllerError, match="unknown task"):
        controller.begin_task(workflow_id, "ghost")


# ---------------------------------------------------------------------------
# Controller: record_observation / complete_task
# ---------------------------------------------------------------------------


def test_record_observation_success_moves_to_observed(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t2")
    updated = controller.record_observation(
        workflow_id,
        Observation(
            task_id="t2",
            content_summary="Itinerary drafted.",
            source=ObservationSource.REASONING,
        ),
    )
    assert updated.task_states["t2"] is TaskStatus.OBSERVED
    assert len(updated.observations) == 1
    event = one_event(controller, workflow_id, EventType.OBSERVATION_CREATED)
    assert event.payload["status"] == "ok"
    assert event.payload["source"] == "reasoning"


def test_record_observation_failure_marks_task_failed(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t1")
    updated = controller.record_observation(
        workflow_id,
        Observation(
            task_id="t1",
            content_summary="Tool failed.",
            is_error=True,
            source=ObservationSource.TOOL,
            error_code="UNAVAILABLE",
        ),
    )
    assert updated.task_states["t1"] is TaskStatus.FAILED
    event = one_event(controller, workflow_id, EventType.OBSERVATION_CREATED)
    assert event.payload["status"] == "error"
    assert event.payload["error_code"] == "UNAVAILABLE"


def test_record_observation_requires_running_task(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    events_before = len(controller.events(workflow_id))
    with pytest.raises(ControllerError, match="expected RUNNING"):
        controller.record_observation(
            workflow_id, Observation(task_id="t1", content_summary="x")
        )
    assert len(controller.events(workflow_id)) == events_before


def test_record_observation_requires_executing_state(controller) -> None:
    workflow_id = ready(controller)
    workflow = controller.get_workflow(workflow_id)
    workflow.task_states["t1"] = TaskStatus.RUNNING  # tampered pre-execution
    controller._store.save_workflow(workflow)
    with pytest.raises(ControllerError, match="while READY"):
        controller.record_observation(
            workflow_id, Observation(task_id="t1", content_summary="x")
        )


def test_complete_task_requires_observed_status(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t1")
    with pytest.raises(ControllerError, match="expected OBSERVED"):
        controller.complete_task(workflow_id, "t1")
    assert controller.get_workflow(workflow_id).task_states["t1"] is TaskStatus.RUNNING


def test_complete_task_requires_a_successful_observation(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    workflow = controller.get_workflow(workflow_id)
    workflow.task_states["t1"] = TaskStatus.OBSERVED  # tampered: no observation
    controller._store.save_workflow(workflow)
    with pytest.raises(ControllerError, match="no successful observation"):
        controller.complete_task(workflow_id, "t1")
    assert (
        controller.get_workflow(workflow_id).task_states["t1"] is TaskStatus.OBSERVED
    )


def test_complete_task_happy_path(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t2")
    controller.record_observation(
        workflow_id, Observation(task_id="t2", content_summary="done")
    )
    updated = controller.complete_task(workflow_id, "t2")
    assert updated.task_states["t2"] is TaskStatus.COMPLETED
    event = one_event(controller, workflow_id, EventType.TASK_COMPLETED)
    assert event.task_id == "t2"
    assert event.payload["index"] == 2


# ---------------------------------------------------------------------------
# Controller: finish_execution / record_execution_failure
# ---------------------------------------------------------------------------


def test_finish_execution_reaches_awaiting_validation_not_completed(
    controller,
) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t2")
    controller.record_observation(
        workflow_id, Observation(task_id="t2", content_summary="done")
    )
    controller.complete_task(workflow_id, "t2")

    # t1 must also finish before the pass can end.
    controller.begin_task(workflow_id, "t1")
    controller.record_observation(
        workflow_id, Observation(task_id="t1", content_summary="computed")
    )
    controller.complete_task(workflow_id, "t1")

    updated = controller.finish_execution(workflow_id)
    assert updated.state is S.AWAITING_VALIDATION
    assert updated.state is not S.COMPLETED
    finished = one_event(controller, workflow_id, EventType.EXECUTION_FINISHED)
    assert finished.payload["tasks"] == 2
    assert "awaiting validation" in finished.decision_summary.lower()


def test_finish_execution_rejects_unfinished_tasks(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    with pytest.raises(ControllerError, match="unfinished tasks"):
        controller.finish_execution(workflow_id)
    assert controller.get_workflow(workflow_id).state is S.EXECUTING


def test_record_execution_failure_emits_then_fails(controller) -> None:
    workflow_id = ready(controller)
    controller.begin_execution(workflow_id)
    failure = FailureContext(
        stage=S.EXECUTING,
        error_code="UNAVAILABLE",
        detail="search provider unavailable",
        task_id="t1",
    )
    updated = controller.record_execution_failure(workflow_id, failure)

    assert updated.state is S.FAILED
    assert updated.failure is not None
    assert updated.failure.error_code == "UNAVAILABLE"
    assert event_types(controller, workflow_id)[-3:] == [
        EventType.EXECUTION_FAILED,
        EventType.STATE_CHANGED,
        EventType.WORKFLOW_FAILED,
    ]
    exec_failed = one_event(controller, workflow_id, EventType.EXECUTION_FAILED)
    assert exec_failed.state is S.EXECUTING  # recorded before the transition
    assert exec_failed.task_id == "t1"


def test_record_execution_failure_rejected_outside_executing(controller) -> None:
    workflow_id = ready(controller)  # READY
    failure = FailureContext(stage=S.EXECUTING, error_code="X", detail="y")
    events_before = len(controller.events(workflow_id))
    with pytest.raises(IllegalTransitionError):
        controller.record_execution_failure(workflow_id, failure)
    assert controller.get_workflow(workflow_id).state is S.READY
    assert len(controller.events(workflow_id)) == events_before


# ---------------------------------------------------------------------------
# Integration: full flow with exact event sequence
# ---------------------------------------------------------------------------


PRE_EXECUTION_EVENTS = [
    EventType.WORKFLOW_STARTED,
    EventType.STATE_CHANGED,  # RECEIVED -> ANALYZING
    EventType.GOAL_ANALYZED,
    EventType.STATE_CHANGED,  # ANALYZING -> PLANNING
    EventType.WORKFLOW_PLANNED,
    EventType.STATE_CHANGED,  # PLANNING -> READY
]

HAPPY_EXECUTION_EVENTS = [
    EventType.STATE_CHANGED,  # READY -> EXECUTING
    EventType.EXECUTION_STARTED,
    EventType.TASK_STARTED,  # t1 (tool)
    EventType.TOOL_DECISION_PROPOSED,
    EventType.TOOL_CALLED,
    EventType.TOOL_COMPLETED,
    EventType.OBSERVATION_CREATED,
    EventType.TASK_COMPLETED,
    EventType.TASK_STARTED,  # t2 (reasoning)
    EventType.OBSERVATION_CREATED,
    EventType.TASK_COMPLETED,
    EventType.STATE_CHANGED,  # EXECUTING -> AWAITING_VALIDATION
    EventType.EXECUTION_FINISHED,
]

FAILURE_EXECUTION_EVENTS = [
    EventType.STATE_CHANGED,  # READY -> EXECUTING
    EventType.EXECUTION_STARTED,
    EventType.TASK_STARTED,
    EventType.TOOL_DECISION_PROPOSED,
    EventType.TOOL_CALLED,
    EventType.TOOL_FAILED,
    EventType.OBSERVATION_CREATED,
    EventType.EXECUTION_FAILED,
    EventType.STATE_CHANGED,  # EXECUTING -> FAILED
    EventType.WORKFLOW_FAILED,
]


async def test_full_flow_reaches_awaiting_validation_with_exact_events(
    controller, registry
) -> None:
    workflow_id = ready(controller)
    provider = MockProvider([CALC_DECISION, REASONING_RESULT])
    executor = Executor(controller, provider, registry)

    report = await executor.execute(workflow_id)

    assert report.success is True
    assert report.final_state is S.AWAITING_VALIDATION

    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.AWAITING_VALIDATION
    assert workflow.task_states == {
        "t1": TaskStatus.COMPLETED,
        "t2": TaskStatus.COMPLETED,
    }
    assert [o.source for o in workflow.observations] == [
        ObservationSource.TOOL,
        ObservationSource.REASONING,
    ]
    # The tool observation carries the calculator's structured result.
    assert isinstance(workflow.observations[0].raw, dict)
    assert workflow.observations[0].raw.get("result") == 4000.0

    assert event_types(controller, workflow_id) == (
        PRE_EXECUTION_EVENTS + HAPPY_EXECUTION_EVENTS
    )


async def test_full_flow_tool_failure_sequence(controller, registry) -> None:
    plan = Plan(tasks=[PLAN.tasks[0]])  # only the tool task
    workflow_id = ready(controller, plan)
    provider = MockProvider(
        [
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "stays"},
                reason_summary="Search for stays.",
            )
        ]
    )
    report = await Executor(controller, provider, registry).execute(workflow_id)

    assert report.success is False
    assert report.failure is not None
    assert report.failure.error_code == "UNAVAILABLE"

    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.FAILED
    assert workflow.task_states["t1"] is TaskStatus.FAILED
    assert workflow.observations[0].is_error is True

    assert event_types(controller, workflow_id) == (
        PRE_EXECUTION_EVENTS + FAILURE_EXECUTION_EVENTS
    )


async def test_goal_to_awaiting_validation_pipeline(controller, registry) -> None:
    """Goal -> analysis -> plan -> READY -> execution -> AWAITING_VALIDATION."""
    from app.agent.goal_analyzer import GoalAnalyzer
    from app.agent.planner import Planner

    # Phase 3 stages with scripted providers (offline).
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    analysis = await GoalAnalyzer(MockProvider([ANALYSIS])).analyze(GOAL)
    controller.record_analysis(workflow.id, analysis)
    plan = await Planner(MockProvider([PLAN])).plan(GOAL, analysis)
    controller.record_plan(workflow.id, plan)
    assert controller.get_workflow(workflow.id).state is S.READY

    # Phase 4 execution.
    provider = MockProvider([CALC_DECISION, REASONING_RESULT])
    report = await Executor(controller, provider, registry).execute(workflow.id)

    assert report.success is True
    assert report.final_state is S.AWAITING_VALIDATION
    final = controller.get_workflow(workflow.id)
    assert final.state is S.AWAITING_VALIDATION
    assert final.state is not S.COMPLETED  # goal NOT validated yet
    assert final.failure is None
    assert all(
        status is TaskStatus.COMPLETED for status in final.task_states.values()
    )
