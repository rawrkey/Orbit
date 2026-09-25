"""Orchestrator tests: state dispatch, clarification, structured failures.

Covers the Controller's clarification/recovery methods and the
WorkflowOrchestrator's dispatch table: fresh runs, clarification rounds,
answers provided upfront, infeasible goals, stage failures as reports
(never exceptions), and terminal-state short circuits. Fully offline.
"""

from __future__ import annotations

import pytest

from app.agent.controller import (
    ControllerError,
    WorkflowController,
)
from app.agent.executor import Executor
from app.agent.goal_analyzer import GoalAnalyzer
from app.agent.planner import Planner
from app.agent.replanner import Replanner
from app.agent.synthesizer import Synthesizer
from app.agent.validator import Validator
from app.agent.workflow import OrchestratorError, WorkflowOrchestrator
from app.config.settings import load_settings
from app.llm.base import LLMError
from app.llm.mock import MockProvider, RoutedMockProvider
from app.llm.schemas import TaskResult, ValidationProposal
from app.state.events import EventType
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    GoalAnalysis,
    Observation,
    ObservationSource,
    Plan,
    SynthesisResult,
    Task,
    TaskStatus,
    Workflow,
    WorkflowState,
)
from app.state.store import InMemoryStateStore, WorkflowNotFoundError
from app.tools.builtin import create_default_registry

S = WorkflowState
E = EventType
GOAL = "Plan a 3-day Mumbai trip for 2 people under 8000 INR"
CLARIFY_GOAL = "Plan a trip."

NO_CONSTRAINT_ANALYSIS = GoalAnalysis(
    objective="Draft a 3-day Mumbai itinerary",
    reason_summary="No hard constraints.",
    constraints=[],
)

CLARIFY_ANALYSIS = GoalAnalysis(
    objective="Plan a trip with the user's dates",
    reason_summary="Travel dates are missing.",
    required_inputs=["travel dates"],
    clarification_required=True,
    reason_valid_placeholder=False if False else "Dates are required.",
)

SYNTH = SynthesisResult(
    final_answer="Itinerary ready for 3 days.",
    key_results=["3-day itinerary drafted"],
    constraints_satisfied=[],
)

SINGLE_PLAN = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Draft itinerary",
            description="Draft the day-by-day itinerary",
            success_criteria="itinerary drafted",
        )
    ],
)

REASONING = TaskResult(
    reason_summary="Itinerary covers 3 days in South Mumbai.",
    data={"days": 3},
)

PASS_PROPOSAL = ValidationProposal(
    reason_summary="Itinerary drafted for all 3 days.",
    constraint_verdicts=[],
    goal_outcome="satisfied",
)


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


@pytest.fixture
def registry():
    return create_default_registry(load_settings({}))


def make_orchestrator(
    controller,
    registry,
    provider,
    *,
    synthesizer: bool = False,
) -> WorkflowOrchestrator:
    return WorkflowOrchestrator(
        controller,
        GoalAnalyzer(provider),
        Planner(provider),
        Executor(controller, provider, registry),
        Validator(provider),
        Replanner(provider),
        synthesizer=Synthesizer(provider) if synthesizer else None,
    )


def event_types(controller, workflow_id: str) -> list[EventType]:
    return [e.type for e in controller.events(workflow_id)]


# ---------------------------------------------------------------------------
# Controller: clarification transitions
# ---------------------------------------------------------------------------


def analyzing(controller: WorkflowController) -> str:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    return workflow.id


def test_require_clarification_stores_analysis_and_questions(controller) -> None:
    workflow_id = analyzing(controller)
    analysis = CLARIFY_ANALYSIS.model_copy(deep=True)
    updated = controller.require_clarification(
        workflow_id, analysis, ["Please provide: travel dates"]
    )
    assert updated.state is S.AWAITING_CLARIFICATION
    assert updated.analysis is not None
    assert updated.analysis.fingerprint() == analysis.fingerprint()
    assert updated.clarification_questions == ["Please provide: travel dates"]
    assert updated.plan is None
    assert event_types(controller, workflow_id) == [
        E.WORKFLOW_STARTED,
        E.STATE_CHANGED,  # RECEIVED -> ANALYZING
        E.GOAL_ANALYZED,
        E.STATE_CHANGED,  # ANALYZING -> AWAITING_CLARIFICATION
        E.CLARIFICATION_REQUESTED,
    ]


def test_require_clarification_illegal_outside_analyzing(controller) -> None:
    workflow_id = analyzing(controller)
    controller.record_analysis(workflow_id, NO_CONSTRAINT_ANALYSIS)
    from app.agent.state_machine import IllegalTransitionError

    with pytest.raises(IllegalTransitionError):
        controller.require_clarification(
            workflow_id, NO_CONSTRAINT_ANALYSIS, ["q1"]
        )


def test_require_clarification_requires_a_question(controller) -> None:
    workflow_id = analyzing(controller)
    with pytest.raises(ControllerError, match="at least"):
        controller.require_clarification(workflow_id, NO_CONSTRAINT_ANALYSIS, [])
    assert controller.get_workflow(workflow_id).state is S.ANALYZING
    with pytest.raises(ControllerError, match="at least"):
        controller.require_clarification(
            workflow_id, NO_CONSTRAINT_ANALYSIS, ["   "]
        )


def test_record_clarification_moves_to_planning(controller) -> None:
    workflow_id = analyzing(controller)
    controller.require_clarification(
        workflow_id, CLARIFY_ANALYSIS, ["Please provide: travel dates"]
    )
    updated = controller.record_clarification(
        workflow_id, {"Please provide: travel dates": "Dec 1-4"}
    )
    assert updated.state is S.PLANNING
    assert updated.clarifications == {"Please provide: travel dates": "Dec 1-4"}
    assert event_types(controller, workflow_id)[-2:] == [
        E.CLARIFICATION_ANSWERED,
        E.STATE_CHANGED,  # AWAITING_CLARIFICATION -> PLANNING
    ]


def test_record_clarification_rejects_missing_answers(controller) -> None:
    workflow_id = analyzing(controller)
    controller.require_clarification(
        workflow_id, CLARIFY_ANALYSIS, ["Please provide: travel dates"]
    )
    with pytest.raises(ControllerError, match="missing answers"):
        controller.record_clarification(workflow_id, {})
    with pytest.raises(ControllerError, match="missing answers"):
        controller.record_clarification(
            workflow_id, {"Please provide: travel dates": "  "}
        )
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.AWAITING_CLARIFICATION
    assert workflow.clarifications == {}


def test_record_clarification_illegal_outside_awaiting(controller) -> None:
    workflow_id = analyzing(controller)
    with pytest.raises(ControllerError, match="not awaiting"):
        controller.record_clarification(workflow_id, {"q": "a"})


# ---------------------------------------------------------------------------
# Controller: interrupted-pass recovery
# ---------------------------------------------------------------------------


def test_recover_interrupted_tasks_resets_running_to_pending(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, NO_CONSTRAINT_ANALYSIS)
    controller.record_plan(workflow.id, SINGLE_PLAN)
    controller.begin_execution(workflow.id)
    controller.begin_task(workflow.id, "t1")  # left RUNNING by the "crash"
    events_before = len(controller.events(workflow.id))

    updated = controller.recover_interrupted_tasks(workflow.id)
    assert updated.state is S.EXECUTING  # no transition
    assert updated.task_states["t1"] is TaskStatus.PENDING
    # repair only: no events, no task-loss
    assert len(controller.events(workflow.id)) == events_before


def test_recover_interrupted_tasks_requires_executing(controller) -> None:
    workflow_id = analyzing(controller)
    with pytest.raises(ControllerError, match="expects EXECUTING"):
        controller.recover_interrupted_tasks(workflow_id)


def test_recover_interrupted_tasks_is_a_noop_without_running_tasks(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, NO_CONSTRAINT_ANALYSIS)
    controller.record_plan(workflow.id, SINGLE_PLAN)
    controller.begin_execution(workflow.id)
    updated = controller.recover_interrupted_tasks(workflow.id)
    assert updated.task_states == {"t1": TaskStatus.PENDING}


# ---------------------------------------------------------------------------
# Orchestrator: happy paths
# ---------------------------------------------------------------------------


async def test_fresh_goal_runs_to_completion(controller, registry) -> None:
    provider = RoutedMockProvider(
        GoalAnalysis=[NO_CONSTRAINT_ANALYSIS],
        Plan=[SINGLE_PLAN],
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
    )
    orchestrator = make_orchestrator(controller, registry, provider)

    workflow = controller.create_workflow(GOAL)
    report = await orchestrator.run(workflow.id)

    assert report.success is True
    assert report.termination == "validated"
    assert report.final_state is S.COMPLETED
    assert (report.executions, report.validations, report.replans) == (1, 1, 0)
    types = event_types(controller, workflow.id)
    assert E.GOAL_ANALYZED in types
    assert E.WORKFLOW_PLANNED in types
    assert E.VALIDATION_PASSED in types
    assert E.WORKFLOW_COMPLETED in types


async def test_with_synthesizer_completion_includes_synthesis(
    controller, registry
) -> None:
    provider = RoutedMockProvider(
        GoalAnalysis=[NO_CONSTRAINT_ANALYSIS],
        Plan=[SINGLE_PLAN],
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
        SynthesisResult=[SYNTH],
    )
    orchestrator = make_orchestrator(
        controller, registry, provider, synthesizer=True
    )
    workflow = controller.create_workflow(GOAL)
    report = await orchestrator.run(workflow.id)

    assert report.termination == "validated"
    assert report.success is True
    workflow = controller.get_workflow(workflow.id)
    assert workflow.synthesis is not None
    assert workflow.final_answer == SYNTH.final_answer
    types = event_types(controller, workflow.id)
    assert E.SYNTHESIS_COMPLETED in types
    assert types[-1] is E.WORKFLOW_COMPLETED


async def test_entry_at_ready_is_dispatched_to_the_loop(
    controller, registry
) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, NO_CONSTRAINT_ANALYSIS)
    controller.record_plan(workflow.id, SINGLE_PLAN)  # READY, never ran
    provider = RoutedMockProvider(
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
    )
    orchestrator = make_orchestrator(controller, registry, provider)
    report = await orchestrator.run(workflow.id)
    assert report.termination == "validated"
    assert report.success is True


# ---------------------------------------------------------------------------
# Orchestrator: clarification rounds
# ---------------------------------------------------------------------------


async def test_clarification_requested_then_answered(controller, registry) -> None:
    provider = RoutedMockProvider(
        GoalAnalysis=[CLARIFY_ANALYSIS],
        Plan=[SINGLE_PLAN],
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
    )
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(CLARIFY_GOAL)

    first = await orchestrator.run(workflow.id)
    assert first.termination == "awaiting_clarification"
    assert first.success is False
    state = controller.get_workflow(workflow.id)
    assert state.state is S.AWAITING_CLARIFICATION
    assert state.clarification_questions == ["Please provide: travel dates"]
    assert state.plan is None
    assert E.CLARIFICATION_REQUESTED in event_types(controller, workflow.id)

    second = await orchestrator.run(
        workflow.id,
        answers={"Please provide: travel dates": "Dec 1-4"},
    )
    assert second.termination == "validated"
    assert second.success is True
    final = controller.get_workflow(workflow.id)
    assert final.state is S.COMPLETED
    assert final.clarifications == {"Please provide: travel dates": "Dec 1-4"}
    types = event_types(controller, workflow.id)
    assert E.CLARIFICATION_ANSWERED in types
    # The planner saw the answer as binding context.
    plan_calls = [c for c in provider.calls if c.schema_name == "Plan"]
    assert len(plan_calls) == 1
    assert "Dec 1-4" in plan_calls[0].prompt
    assert "binding" in plan_calls[0].prompt


async def test_answers_upfront_skip_the_clarification_round(
    controller, registry
) -> None:
    provider = RoutedMockProvider(
        GoalAnalysis=[CLARIFY_ANALYSIS],
        Plan=[SINGLE_PLAN],
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
    )
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(CLARIFY_GOAL)

    report = await orchestrator.run(
        workflow.id,
        answers={"Please provide: travel dates": "Dec 1-4"},
    )
    assert report.termination == "validated"
    types = event_types(controller, workflow.id)
    assert E.CLARIFICATION_REQUESTED not in types
    assert E.CLARIFICATION_ANSWERED not in types
    plan_calls = [c for c in provider.calls if c.schema_name == "Plan"]
    assert "Dec 1-4" in plan_calls[0].prompt


async def test_partial_answers_are_rejected_by_the_controller(
    controller, registry
) -> None:
    provider = RoutedMockProvider(GoalAnalysis=[CLARIFY_ANALYSIS])
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(CLARIFY_GOAL)
    await orchestrator.run(workflow.id)  # -> AWAITING_CLARIFICATION

    with pytest.raises(ControllerError, match="missing answers"):
        await orchestrator.run(workflow.id, answers={"wrong key": "Dec 1-4"})
    assert (
        controller.get_workflow(workflow.id).state
        is S.AWAITING_CLARIFICATION
    )


async def test_awaiting_without_answers_reports_awaiting(controller, registry) -> None:
    provider = RoutedMockProvider(GoalAnalysis=[CLARIFY_ANALYSIS])
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(CLARIFY_GOAL)
    await orchestrator.run(workflow.id)
    report = await orchestrator.run(workflow.id)
    assert report.termination == "awaiting_clarification"
    assert report.final_state is S.AWAITING_CLARIFICATION


# ---------------------------------------------------------------------------
# Orchestrator: structured failures (reports, not exceptions)
# ---------------------------------------------------------------------------


async def test_analysis_llm_error_fails_the_workflow(controller, registry) -> None:
    provider = RoutedMockProvider()  # no GoalAnalysis scripted
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(GOAL)

    report = await orchestrator.run(workflow.id)

    assert report.termination == "analysis_error"
    assert report.success is False
    assert report.final_state is S.FAILED
    assert report.failure is not None
    assert report.failure.stage is S.ANALYZING
    assert report.failure.error_code == "LLM_ERROR"
    assert E.WORKFLOW_FAILED in event_types(controller, workflow.id)


async def test_infeasible_goal_fails_fast(controller, registry) -> None:
    infeasible = NO_CONSTRAINT_ANALYSIS.model_copy(
        update={"is_feasible": False, "reason_summary": "Impossible goal."}
    )
    provider = RoutedMockProvider(GoalAnalysis=[infeasible])
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(GOAL)

    report = await orchestrator.run(workflow.id)

    assert report.termination == "infeasible"
    assert report.failure is not None
    assert report.failure.error_code == "GOAL_INFEASIBLE"
    assert report.final_state is S.FAILED


async def test_planning_llm_error_fails_the_workflow(controller, registry) -> None:
    provider = RoutedMockProvider(GoalAnalysis=[NO_CONSTRAINT_ANALYSIS])
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(GOAL)

    report = await orchestrator.run(workflow.id)

    assert report.termination == "planning_error"
    assert report.failure is not None
    assert report.failure.stage is S.PLANNING
    assert report.failure.error_code == "LLM_ERROR"
    assert report.final_state is S.FAILED


async def test_invalid_plan_becomes_plan_invalid_failure(controller, registry) -> None:
    broken = Plan(
        tasks=[
            Task(
                id="t1",
                description="Step",
                success_criteria="done",
                depends_on=["ghost"],
            )
        ]
    )
    provider = RoutedMockProvider(
        GoalAnalysis=[NO_CONSTRAINT_ANALYSIS],
        Plan=[broken, broken],  # repair attempt fails too
    )
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(GOAL)

    report = await orchestrator.run(workflow.id)

    assert report.termination == "planning_error"
    assert report.failure is not None
    assert report.failure.error_code == "PLAN_INVALID"
    types = event_types(controller, workflow.id)
    assert E.PLAN_VALIDATION_FAILED in types
    assert E.WORKFLOW_FAILED in types
    assert controller.get_workflow(workflow.id).plan is None


# ---------------------------------------------------------------------------
# Orchestrator: terminal short circuits and dispatch errors
# ---------------------------------------------------------------------------


async def test_terminal_workflows_short_circuit(controller, registry) -> None:
    provider = RoutedMockProvider()
    orchestrator = make_orchestrator(controller, registry, provider)
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, NO_CONSTRAINT_ANALYSIS)
    controller.fail(
        workflow.id,
        __import__("app.state.models", fromlist=["FailureContext"]).FailureContext(
            stage=S.PLANNING, error_code="X", detail="stopped"
        ),
    )
    assert controller.get_workflow(workflow.id).state is S.FAILED

    report = await orchestrator.run(workflow.id)
    assert report.termination == "already_terminal"
    assert report.success is False
    assert provider.calls == []  # no LLM calls on a terminal workflow


async def test_unknown_workflow_raises(controller, registry) -> None:
    provider = RoutedMockProvider()
    orchestrator = make_orchestrator(controller, registry, provider)
    with pytest.raises(WorkflowNotFoundError):
        await orchestrator.run("missing-id")


async def test_planning_without_analysis_is_a_programming_error(
    controller, registry
) -> None:
    workflow = controller.create_workflow(GOAL)
    workflow.state = S.PLANNING  # crafted invalid shape
    controller._store.save_workflow(workflow)
    provider = RoutedMockProvider()
    orchestrator = make_orchestrator(controller, registry, provider)
    with pytest.raises(OrchestratorError, match="without a stored analysis"):
        await orchestrator.run(workflow.id)


async def test_awaiting_with_questions_dispatches_back_to_status(
    controller, registry
) -> None:
    workflow = controller.create_workflow(GOAL)
    provider = RoutedMockProvider()
    orchestrator = make_orchestrator(controller, registry, provider)
    # Craft an awaiting shape the LOOP would refuse (AdaptiveLoopError);
    # the orchestrator intercepts it and reports status instead of crashing.
    workflow.clarification_questions = ["?"]
    workflow.state = S.AWAITING_CLARIFICATION
    controller._store.save_workflow(workflow)
    report = await orchestrator.run(workflow.id)
    assert report.termination == "awaiting_clarification"
    assert provider.calls == []
