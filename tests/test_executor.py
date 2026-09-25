"""Executor tests: selection, execution paths, failures, and bounds."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.agent.controller import WorkflowController
from app.agent.executor import (
    ExecutionError,
    Executor,
    UnexecutableTasksError,
    build_execution_context,
    select_next_task,
)
from app.agent.state_machine import IllegalTransitionError
from app.config.settings import load_settings
from app.llm.base import LLMError
from app.llm.mock import MockProvider
from app.llm.schemas import TaskResult, ToolDecision
from app.state.events import EventType
from app.state.models import (
    Constraint,
    GoalAnalysis,
    Observation,
    ObservationSource,
    Plan,
    Task,
    TaskStatus,
    WorkflowState,
)
from app.tools.base import Tool
from app.tools.builtin import create_default_registry
from app.tools.registry import ToolRegistry

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


def make_task(id: str, **overrides) -> Task:
    base: dict = {
        "id": id,
        "description": f"step {id}",
        "success_criteria": f"{id} completed",
    }
    base.update(overrides)
    return Task(**base)


def make_plan(*tasks: Task) -> Plan:
    return Plan(tasks=list(tasks))


def ready_workflow(controller: WorkflowController, plan: Plan) -> str:
    """Drive a workflow through analysis/planning to READY."""
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, plan)
    return workflow.id


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(_store(), max_replans=2)


def _store():
    from app.state.store import InMemoryStateStore

    return InMemoryStateStore()


@pytest.fixture
def registry() -> ToolRegistry:
    return create_default_registry(load_settings({}))


def make_executor(
    controller, provider, registry, **kwargs
) -> Executor:
    return Executor(controller, provider, registry, **kwargs)


# ---------------------------------------------------------------------------
# Deterministic task selection (pure)
# ---------------------------------------------------------------------------


def test_selects_first_executable_task() -> None:
    plan = make_plan(make_task("t1"), make_task("t2"))
    selected = select_next_task(plan, {"t1": TaskStatus.PENDING, "t2": TaskStatus.PENDING})
    assert selected is not None and selected.id == "t1"


def test_respects_plan_ordering() -> None:
    plan = make_plan(make_task("t1"), make_task("t2"), make_task("t3"))
    assert select_next_task(plan, {}).id == "t1"


def test_respects_dependencies() -> None:
    plan = make_plan(make_task("t1"), make_task("t2", depends_on=["t1"]))
    states = {"t1": TaskStatus.PENDING, "t2": TaskStatus.PENDING}
    first = select_next_task(plan, states)
    assert first is not None and first.id == "t1"

    states = {"t1": TaskStatus.COMPLETED, "t2": TaskStatus.PENDING}
    second = select_next_task(plan, states)
    assert second is not None and second.id == "t2"


def test_skips_handled_tasks() -> None:
    plan = make_plan(make_task("t1"), make_task("t2"))
    states = {"t1": TaskStatus.FAILED, "t2": TaskStatus.PENDING}
    selected = select_next_task(plan, states)
    assert selected is not None and selected.id == "t2"


def test_blocks_task_with_incomplete_dependency() -> None:
    plan = make_plan(make_task("t1"), make_task("t2", depends_on=["t1"]))
    states = {"t1": TaskStatus.RUNNING, "t2": TaskStatus.PENDING}
    with pytest.raises(UnexecutableTasksError) as excinfo:
        select_next_task(plan, states)
    assert excinfo.value.blocked_task_ids == ["t2"]


def test_detects_impossible_dependency_state() -> None:
    plan = make_plan(make_task("t1"), make_task("t2", depends_on=["t1"]))
    states = {"t1": TaskStatus.FAILED, "t2": TaskStatus.PENDING}
    with pytest.raises(UnexecutableTasksError):
        select_next_task(plan, states)


def test_unknown_dependency_is_blocked_not_skipped() -> None:
    plan = make_plan(make_task("t1", depends_on=["ghost"]))
    with pytest.raises(UnexecutableTasksError) as excinfo:
        select_next_task(plan, {})
    assert excinfo.value.blocked_task_ids == ["t1"]


def test_returns_none_when_every_task_is_done() -> None:
    plan = make_plan(make_task("t1"), make_task("t2"))
    states = {"t1": TaskStatus.COMPLETED, "t2": TaskStatus.COMPLETED}
    assert select_next_task(plan, states) is None


def test_observed_but_not_completed_dependency_still_blocks() -> None:
    plan = make_plan(make_task("t1"), make_task("t2", depends_on=["t1"]))
    states = {"t1": TaskStatus.OBSERVED, "t2": TaskStatus.PENDING}
    with pytest.raises(UnexecutableTasksError):
        select_next_task(plan, states)


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


def test_context_is_bounded_to_five_previous_observations(
    controller, registry
) -> None:
    plan = make_plan(make_task("t1"), make_task("t2"))
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    for i in range(8):
        workflow.observations.append(
            Observation(task_id=f"old{i}", content_summary=f"observation {i}")
        )
    context = build_execution_context(
        workflow, workflow.plan.tasks[1], registry
    )
    assert len(context.previous_observations) == 5
    assert context.previous_observations[-1].summary == "observation 7"


def test_context_lists_registry_tools_as_source_of_truth(
    controller, registry
) -> None:
    plan = make_plan(make_task("t1", requires_tool=True, tool_hint="calculator"))
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    context = build_execution_context(workflow, workflow.plan.tasks[0], registry)
    assert sorted(t.name for t in context.available_tools) == registry.names()


def test_context_exposes_goal_objective_and_constraints(
    controller, registry
) -> None:
    plan = make_plan(make_task("t1", constraint_ids=["c1"]))
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    context = build_execution_context(workflow, workflow.plan.tasks[0], registry)
    assert context.goal == GOAL
    assert context.objective == ANALYSIS.objective
    assert [c.id for c in context.constraints] == ["c1"]


def test_context_has_no_event_log_or_prompt_fields(controller, registry) -> None:
    plan = make_plan(make_task("t1"))
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    dumped = build_execution_context(
        workflow, workflow.plan.tasks[0], registry
    ).model_dump_json()
    assert "chain" not in dumped
    assert "event_count" not in dumped
    assert "final_answer" not in dumped


# ---------------------------------------------------------------------------
# Happy-path execution
# ---------------------------------------------------------------------------

CALC_DECISION = ToolDecision(
    tool_name="calculator",
    arguments={"expression": "2+2"},
    reason_summary="Compute the per-person share.",
)
REASONING_RESULT = TaskResult(
    reason_summary="Itinerary covers 3 days.", data={"days": 3}
)


async def test_executes_tool_task_then_reasoning_task(
    controller, registry
) -> None:
    plan = make_plan(
        make_task("t1", requires_tool=True, tool_hint="calculator"),
        make_task("t2", title="Draft itinerary"),
    )
    workflow_id = ready_workflow(controller, plan)
    plan_before = controller.get_workflow(workflow_id).plan.model_dump()

    provider = MockProvider([CALC_DECISION, REASONING_RESULT])
    report = await make_executor(controller, provider, registry).execute(workflow_id)

    assert report.success is True
    assert report.final_state is S.AWAITING_VALIDATION
    assert report.completed_task_ids == ["t1", "t2"]
    assert report.failed_task_id is None
    assert report.observation_count == 2

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
    # The plan structure is never modified by execution.
    assert workflow.plan.model_dump() == plan_before
    # Exactly one LLM call per task - no hidden retries.
    assert len(provider.calls) == 2


async def test_respects_dependency_order_during_execution(
    controller, registry
) -> None:
    plan = make_plan(
        make_task("t2", depends_on=["t1"]),
        make_task("t1"),
    )
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider([REASONING_RESULT, REASONING_RESULT])
    await make_executor(controller, provider, registry).execute(workflow_id)

    started = [
        e.task_id
        for e in controller.events(workflow_id)
        if e.type is EventType.TASK_STARTED
    ]
    # t1 is listed second but t2 depends on it - t1 must run first.
    assert started == ["t1", "t2"]


async def test_reasoning_task_receives_previous_observations(
    controller, registry
) -> None:
    plan = make_plan(
        make_task("t1", requires_tool=True, tool_hint="calculator"),
        make_task("t2"),
    )
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider([CALC_DECISION, REASONING_RESULT])
    await make_executor(controller, provider, registry).execute(workflow_id)

    second_prompt = provider.calls[1].prompt
    assert "previous_observations" in second_prompt
    assert "calculator" in second_prompt  # t1's observation is visible
    assert "chain_of_thought" not in second_prompt


async def test_tool_decision_prompt_lists_available_tools(
    controller, registry
) -> None:
    plan = make_plan(make_task("t1", requires_tool=True))
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider([CALC_DECISION])
    await make_executor(controller, provider, registry).execute(workflow_id)

    prompt = provider.calls[0].prompt
    assert "available_tools" in prompt
    assert "calculator" in prompt
    assert provider.calls[0].schema_name == "ToolDecision"


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------


async def test_unknown_tool_fails_typed_and_stops(controller, registry) -> None:
    plan = make_plan(
        make_task("t1", requires_tool=True),
        make_task("t2"),
    )
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider(
        [ToolDecision(tool_name="ghost", arguments={}, reason_summary="Guessing.")]
    )
    report = await make_executor(controller, provider, registry).execute(workflow_id)

    assert report.success is False
    assert report.failed_task_id == "t1"
    assert report.failure is not None and report.failure.error_code == "TOOL_NOT_FOUND"

    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.FAILED
    assert workflow.task_states["t1"] is TaskStatus.FAILED
    assert workflow.task_states["t2"] is TaskStatus.PENDING  # never started
    observation = workflow.observations[0]
    assert observation.is_error is True
    assert observation.source is ObservationSource.DECISION
    assert observation.error_code == "TOOL_NOT_FOUND"
    # Execution stopped: t2 never got an LLM call.
    assert len(provider.calls) == 1


async def test_invalid_tool_arguments_fail_before_execution(
    controller, registry
) -> None:
    spy = SpyTool()
    registry.register(spy)
    plan = make_plan(make_task("t1", requires_tool=True))
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider(
        [ToolDecision(tool_name="spy", arguments={"bogus": 1}, reason_summary="Bad args.")]
    )
    report = await make_executor(controller, provider, registry).execute(workflow_id)

    assert report.success is False
    assert report.failure is not None
    assert report.failure.error_code == "MALFORMED_INPUT"
    assert spy.ran is False  # LLM output never reached execution directly
    assert controller.get_workflow(workflow_id).state is S.FAILED


async def test_tool_failure_is_an_observation_then_typed_stop(
    controller, registry
) -> None:
    plan = make_plan(make_task("t1", requires_tool=True, tool_hint="web_search"))
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider(
        [ToolDecision(tool_name="web_search", arguments={"query": "stays"},
                      reason_summary="Search for stays.")]
    )
    report = await make_executor(controller, provider, registry).execute(workflow_id)

    assert report.success is False
    assert report.failure is not None
    assert report.failure.error_code == "UNAVAILABLE"

    workflow = controller.get_workflow(workflow_id)
    observation = workflow.observations[0]
    assert observation.is_error is True
    assert observation.source is ObservationSource.TOOL
    assert observation.tool_name == "web_search"
    assert workflow.task_states["t1"] is TaskStatus.FAILED
    assert workflow.state is S.FAILED
    # No automatic retry beyond the tool primitive: exactly one LLM call.
    assert len(provider.calls) == 1
    event_types = [e.type for e in controller.events(workflow_id)]
    assert EventType.TOOL_FAILED in event_types
    assert EventType.EXECUTION_FAILED in event_types


async def test_llm_error_becomes_typed_failed_observation(
    controller, registry
) -> None:
    plan = make_plan(make_task("t1"))
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider()  # exhausted -> LLMError
    report = await make_executor(controller, provider, registry).execute(workflow_id)

    assert report.success is False
    assert report.failure is not None
    assert report.failure.error_code == "LLM_ERROR"

    workflow = controller.get_workflow(workflow_id)
    observation = workflow.observations[0]
    assert observation.is_error is True
    assert observation.source is ObservationSource.SYSTEM
    assert observation.error_code == "LLM_ERROR"
    assert workflow.state is S.FAILED


async def test_execution_stops_at_first_failure(controller, registry) -> None:
    plan = make_plan(
        make_task("t1", requires_tool=True),
        make_task("t2", requires_tool=True),
        make_task("t3"),
    )
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider(
        [ToolDecision(tool_name="ghost", arguments={}, reason_summary="x")]
    )
    report = await make_executor(controller, provider, registry).execute(workflow_id)

    assert report.completed_task_ids == []
    assert report.failed_task_id == "t1"
    started = [
        e.task_id
        for e in controller.events(workflow_id)
        if e.type is EventType.TASK_STARTED
    ]
    assert started == ["t1"]
    workflow = controller.get_workflow(workflow_id)
    assert workflow.task_states["t2"] is TaskStatus.PENDING
    assert workflow.task_states["t3"] is TaskStatus.PENDING


async def test_no_task_executes_twice(controller, registry) -> None:
    plan = make_plan(
        make_task("t1"),
        make_task("t2", depends_on=["t1"]),
        make_task("t3"),
    )
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider(
        [REASONING_RESULT, REASONING_RESULT, REASONING_RESULT]
    )
    await make_executor(controller, provider, registry).execute(workflow_id)

    started = [
        e.task_id
        for e in controller.events(workflow_id)
        if e.type is EventType.TASK_STARTED
    ]
    assert sorted(started) == ["t1", "t2", "t3"]
    assert len(started) == 3
    completed = [
        e.task_id
        for e in controller.events(workflow_id)
        if e.type is EventType.TASK_COMPLETED
    ]
    assert sorted(completed) == ["t1", "t2", "t3"]
    # Bounded: exactly one LLM call per task, no extra passes.
    assert len(provider.calls) == 3


async def test_dependency_deadlock_records_failure_and_raises(
    controller, registry
) -> None:
    plan = make_plan(make_task("t1"), make_task("t2", depends_on=["t1"]))
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    workflow.task_states["t1"] = TaskStatus.FAILED  # corrupted/blocked state
    controller._store.save_workflow(workflow)

    provider = MockProvider([REASONING_RESULT])
    with pytest.raises(UnexecutableTasksError):
        await make_executor(controller, provider, registry).execute(workflow_id)

    assert controller.get_workflow(workflow_id).state is S.FAILED
    events = controller.events(workflow_id)
    assert any(e.type is EventType.EXECUTION_FAILED for e in events)


# ---------------------------------------------------------------------------
# Bounds / lifecycle guards
# ---------------------------------------------------------------------------


async def test_execute_requires_ready_state(controller, registry) -> None:
    plan = make_plan(make_task("t1"))
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider([REASONING_RESULT])
    executor = make_executor(controller, provider, registry)
    await executor.execute(workflow_id)

    with pytest.raises(IllegalTransitionError):
        await executor.execute(workflow_id)
    assert controller.get_workflow(workflow_id).state is S.AWAITING_VALIDATION


async def test_execute_without_plan_raises_before_mutation(
    controller, registry
) -> None:
    workflow = controller.create_workflow(GOAL)  # RECEIVED, no plan
    provider = MockProvider([REASONING_RESULT])
    with pytest.raises(ExecutionError, match="no plan"):
        await make_executor(controller, provider, registry).execute(workflow.id)
    assert controller.get_workflow(workflow.id).state is S.RECEIVED
    assert controller.events(workflow.id) == [] or len(controller.events(workflow.id)) == 1


async def test_execute_with_empty_plan_raises_before_mutation(
    controller, registry
) -> None:
    plan = make_plan()
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider()
    with pytest.raises(ExecutionError, match="no tasks"):
        await make_executor(controller, provider, registry).execute(workflow_id)
    assert controller.get_workflow(workflow_id).state is S.READY
    assert provider.calls == []


async def test_execution_never_reaches_completed_state(controller, registry) -> None:
    plan = make_plan(make_task("t1"))
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider([REASONING_RESULT])
    report = await make_executor(controller, provider, registry).execute(workflow_id)
    assert report.final_state is not S.COMPLETED
    assert controller.get_workflow(workflow_id).state is S.AWAITING_VALIDATION


async def test_failed_task_is_never_marked_completed(controller, registry) -> None:
    plan = make_plan(make_task("t1", requires_tool=True))
    workflow_id = ready_workflow(controller, plan)
    provider = MockProvider(
        [ToolDecision(tool_name="ghost", arguments={}, reason_summary="x")]
    )
    await make_executor(controller, provider, registry).execute(workflow_id)
    assert controller.get_workflow(workflow_id).task_states["t1"] is not TaskStatus.COMPLETED
    assert controller.get_workflow(workflow_id).task_states["t1"] is TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Test double: a tool that records whether the model reached it directly
# ---------------------------------------------------------------------------


class _SpyInput(BaseModel):
    value: int


class _SpyOutput(BaseModel):
    ran: bool = True


class SpyTool(Tool):
    name = "spy"
    description = "records whether it was executed"
    input_schema = _SpyInput
    output_schema = _SpyOutput

    def __init__(self) -> None:
        self.ran = False

    def execute(self, arguments: BaseModel) -> BaseModel:
        self.ran = True
        return _SpyOutput()
