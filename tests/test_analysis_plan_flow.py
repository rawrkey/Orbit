"""End-to-end offline flow: receive -> analyze -> plan -> ready.

Exercises GoalAnalyzer + Planner + WorkflowController together with a
scripted MockProvider - no network, no API key.
"""

from __future__ import annotations

import pytest

from app.agent.controller import WorkflowController
from app.agent.goal_analyzer import GoalAnalyzer
from app.agent.planner import Planner, PlanValidationError
from app.agent.state_machine import IllegalTransitionError
from app.llm.mock import MockProvider
from app.state.events import EventType
from app.state.models import (
    Constraint,
    ConstraintType,
    GoalAnalysis,
    Plan,
    Task,
    TaskStatus,
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
            id="c1",
            name="budget",
            type=ConstraintType.BUDGET,
            description="Total under 8000 INR",
        )
    ],
    is_feasible=True,
)

PLAN = Plan(
    tasks=[
        Task(
            id="t1",
            title="Shortlist stays",
            description="Find stays within budget",
            requires_tool=True,
            tool_hint="web_search",
            expected_output="3 stay options with prices",
            constraint_ids=["c1"],
            success_criteria="3 options under 4000 INR each",
        ),
        Task(
            id="t2",
            title="Build day-by-day itinerary",
            description="Draft the itinerary",
            success_criteria="3 days with morning/afternoon/evening",
            depends_on=["t1"],
        ),
    ]
)


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


def event_types(controller: WorkflowController, workflow_id: str) -> list[EventType]:
    return [event.type for event in controller.events(workflow_id)]


async def test_full_analysis_plan_flow_reaches_ready(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)

    analyzer = GoalAnalyzer(MockProvider([ANALYSIS]))
    analysis = await analyzer.analyze(GOAL)
    controller.record_analysis(workflow.id, analysis)
    assert controller.get_workflow(workflow.id).state is S.PLANNING

    planner = Planner(MockProvider([PLAN]))
    plan = await planner.plan(GOAL, analysis)
    controller.record_plan(workflow.id, plan)

    final = controller.get_workflow(workflow.id)
    assert final.state is S.READY
    assert final.analysis is not None and final.analysis.objective
    assert final.plan is not None and len(final.plan.tasks) == 2
    assert final.task_states == {"t1": TaskStatus.PENDING, "t2": TaskStatus.PENDING}

    assert event_types(controller, workflow.id) == [
        EventType.WORKFLOW_STARTED,
        EventType.STATE_CHANGED,  # RECEIVED -> ANALYZING
        EventType.GOAL_ANALYZED,
        EventType.STATE_CHANGED,  # ANALYZING -> PLANNING
        EventType.WORKFLOW_PLANNED,
        EventType.STATE_CHANGED,  # PLANNING -> READY
    ]


async def test_analyzer_and_planner_never_touch_state(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    events_before = len(controller.events(workflow.id))

    await GoalAnalyzer(MockProvider([ANALYSIS])).analyze(GOAL)
    await Planner(MockProvider([PLAN])).plan(GOAL, ANALYSIS)

    state = controller.get_workflow(workflow.id)
    assert state.state is S.RECEIVED
    assert state.analysis is None
    assert state.plan is None
    assert len(controller.events(workflow.id)) == events_before


async def test_analysis_flow_is_deterministic_and_reproducible(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    analysis = await GoalAnalyzer(MockProvider([ANALYSIS])).analyze(GOAL)
    controller.record_analysis(workflow.id, analysis)
    plan = await Planner(MockProvider([PLAN])).plan(GOAL, analysis)
    controller.record_plan(workflow.id, plan)

    twin = controller.create_workflow(GOAL)
    controller.begin_analysis(twin.id)
    controller.record_analysis(twin.id, analysis)
    controller.record_plan(twin.id, plan)

    original = controller.get_workflow(workflow.id)
    replay = controller.get_workflow(twin.id)
    assert replay.task_states == original.task_states
    assert replay.state == original.state
    assert [t.id for t in replay.plan.tasks] == [t.id for t in original.plan.tasks]


def test_begin_analysis_is_legal_only_from_received(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    events_before = len(controller.events(workflow.id))

    with pytest.raises(IllegalTransitionError):
        controller.begin_analysis(workflow.id)

    assert controller.get_workflow(workflow.id).state is S.ANALYZING
    assert len(controller.events(workflow.id)) == events_before


def test_record_analysis_requires_analyzing_state(controller) -> None:
    workflow = controller.create_workflow(GOAL)  # still RECEIVED
    events_before = len(controller.events(workflow.id))

    with pytest.raises(IllegalTransitionError):
        controller.record_analysis(workflow.id, ANALYSIS)

    stored = controller.get_workflow(workflow.id)
    assert stored.state is S.RECEIVED
    assert stored.analysis is None
    assert len(controller.events(workflow.id)) == events_before


def test_record_plan_requires_planning_state(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)  # now PLANNING
    # skip straight past PLANNING is impossible: try recording a plan from
    # a state where the transition table disallows PLANNING -> READY...
    # here state IS PLANNING so this must succeed; the negative case is below.
    controller.record_plan(workflow.id, PLAN)
    assert controller.get_workflow(workflow.id).state is S.READY

    events_before = len(controller.events(workflow.id))
    with pytest.raises(IllegalTransitionError):
        controller.record_plan(workflow.id, PLAN)
    assert controller.get_workflow(workflow.id).state is S.READY
    assert len(controller.events(workflow.id)) == events_before


def test_record_plan_initializes_task_states_and_emits(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, PLAN)

    workflow = controller.get_workflow(workflow.id)
    assert workflow.task_states == {"t1": TaskStatus.PENDING, "t2": TaskStatus.PENDING}

    planned = [
        e for e in controller.events(workflow.id)
        if e.type is EventType.WORKFLOW_PLANNED
    ]
    assert len(planned) == 1
    assert planned[0].payload["tasks"] == 2
    assert planned[0].payload["task_ids"] == ["t1", "t2"]


def test_record_plan_validation_failure_only_in_planning(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)  # PLANNING

    updated = controller.record_plan_validation_failure(
        workflow.id, ["plan has no tasks"]
    )
    assert updated.state is S.PLANNING  # state unchanged: diagnosis only

    failure_events = [
        e for e in controller.events(workflow.id)
        if e.type is EventType.PLAN_VALIDATION_FAILED
    ]
    assert len(failure_events) == 1
    assert failure_events[0].payload["issue_count"] == 1

    controller.record_plan(workflow.id, PLAN)  # recover: PLANNING -> READY

    with pytest.raises(IllegalTransitionError):
        controller.record_plan_validation_failure(workflow.id, ["too late"])
    assert controller.get_workflow(workflow.id).state is S.READY


def test_record_plan_validation_failure_requires_issues(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)

    with pytest.raises(ValueError):
        controller.record_plan_validation_failure(workflow.id, [])
    assert (
        EventType.PLAN_VALIDATION_FAILED
        not in event_types(controller, workflow.id)
    )


async def test_planner_failure_is_reportable_through_controller(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)

    invalid = Plan(tasks=[])
    planner = Planner(MockProvider([invalid, invalid]))
    with pytest.raises(PlanValidationError) as excinfo:
        await planner.plan(GOAL, ANALYSIS)

    controller.record_plan_validation_failure(workflow.id, excinfo.value.issues)

    workflow = controller.get_workflow(workflow.id)
    assert workflow.state is S.PLANNING
    assert workflow.plan is None
    assert any(
        e.type is EventType.PLAN_VALIDATION_FAILED
        for e in controller.events(workflow.id)
    )
