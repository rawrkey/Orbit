"""End-to-end integration: full lifecycle, clarification, and the demo.

Each test wires real stages (analyzer -> planner -> executor -> validator ->
replanner -> synthesizer) against a scripted offline provider and asserts
the observable contract: state progression, preserved work across replans,
PASS-only completion with synthesis, and an intact append-only event log.
No network, no API key, no fabricated evidence.
"""

from __future__ import annotations

import pytest

from app.agent.controller import WorkflowController
from app.agent.executor import Executor
from app.agent.goal_analyzer import GoalAnalyzer
from app.agent.planner import Planner
from app.agent.replanner import Replanner
from app.agent.synthesizer import Synthesizer
from app.agent.validator import Validator
from app.agent.workflow import WorkflowOrchestrator
from app.config.settings import load_settings
from app.demo import DEMO_GOAL, DEMO_SYNTHESIS, create_demo_provider
from app.llm.mock import RoutedMockProvider
from app.llm.schemas import TaskResult, ValidationProposal
from app.state.events import EventType
from app.state.models import (
    GoalAnalysis,
    Plan,
    SynthesisResult,
    Task,
    TaskStatus,
    WorkflowState,
)
from app.state.store import InMemoryStateStore
from app.tools.builtin import create_default_registry

S = WorkflowState
E = EventType
ANSWER_KEY = "Please provide: travel dates"

CLARIFY_ANALYSIS = GoalAnalysis(
    objective="Plan the trip once dates are known",
    reason_summary="Travel dates are missing.",
    required_inputs=["travel dates"],
    clarification_required=True,
)
CLARIFY_PLAN = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Draft itinerary",
            description="Draft the itinerary for the given dates",
            success_criteria="itinerary drafted",
        )
    ],
)
SYNTH = SynthesisResult(
    final_answer="Itinerary ready for Dec 1-4.",
    key_results=["3-day itinerary"],
    constraints_satisfied=[],
)


def build_orchestrator(controller, provider, *, synthesizer=True):
    registry = create_default_registry(load_settings({}))
    return WorkflowOrchestrator(
        controller,
        GoalAnalyzer(provider),
        Planner(provider),
        Executor(controller, provider, registry),
        Validator(provider),
        Replanner(provider),
        synthesizer=Synthesizer(provider) if synthesizer else None,
    )


def visited_states(controller, workflow_id: str) -> list[str]:
    """States observed in the log, in order (workflow state field per event)."""
    return [event.state.value for event in controller.events(workflow_id)]


# ---------------------------------------------------------------------------
# Item 25: full lifecycle - analyze, plan, execute, validate, replan,
# execute, validate, synthesize, complete
# ---------------------------------------------------------------------------


async def test_full_lifecycle_with_replan_preserves_work() -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=3)
    provider = create_demo_provider()
    orchestrator = build_orchestrator(controller, provider)

    workflow = controller.create_workflow(DEMO_GOAL)
    report = await orchestrator.run(workflow.id)

    # Report contract
    assert report.success is True
    assert report.termination == "validated"
    assert (report.executions, report.validations, report.replans) == (2, 2, 1)

    # Terminal state + PASS-gated synthesis
    final = controller.get_workflow(workflow.id)
    assert final.state is S.COMPLETED
    assert final.replan_count == 1
    assert final.plan.version == 2
    assert final.validation is not None
    assert final.validation.status.value == "PASS"
    assert final.synthesis is not None
    assert final.final_answer == DEMO_SYNTHESIS.final_answer
    assert "7,500" in final.final_answer

    # Work preserved across the replan: t2 survived, both tasks completed
    assert final.task_states["t2"] is TaskStatus.COMPLETED
    assert final.task_states["t1"] is TaskStatus.COMPLETED
    assert len(final.observations) == 3  # 9000-result, summary, 7500-result
    tool_results = [
        o.raw.get("result")
        for o in final.observations
        if o.source.value == "tool"
    ]
    assert tool_results == [9000, 7500]  # failing run kept, then the fix

    # Event log: intact, ordered, tells the story
    events = controller.events(workflow.id)
    assert [e.seq for e in events] == list(range(len(events)))
    types = [e.type for e in events]
    for expected in (
        E.GOAL_ANALYZED,
        E.WORKFLOW_PLANNED,
        E.VALIDATION_FAILED,
        E.REPLAN_ACCEPTED,
        E.VALIDATION_PASSED,
        E.SYNTHESIS_COMPLETED,
        E.WORKFLOW_COMPLETED,
    ):
        assert expected in types, expected
    assert types.index(E.VALIDATION_FAILED) < types.index(E.REPLAN_ACCEPTED)
    assert types.index(E.REPLAN_ACCEPTED) < types.index(E.VALIDATION_PASSED)
    assert types[-1] is E.WORKFLOW_COMPLETED

    # Bounded: every visited state was legal along the way
    visited = visited_states(controller, workflow.id)
    assert visited[0] == S.RECEIVED.value
    assert visited[-1] == S.COMPLETED.value
    assert S.REPLANNING.value in visited
    assert len(visited) < 50

    # Offline: only the scripted provider was ever called
    assert provider.model_id.startswith("mock/")


async def test_completed_lifecycle_rerun_makes_no_calls() -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=3)
    provider = create_demo_provider()
    orchestrator = build_orchestrator(controller, provider)
    workflow = controller.create_workflow(DEMO_GOAL)
    await orchestrator.run(workflow.id)

    calls_before = len(provider.calls)
    rerun = await orchestrator.run(workflow.id)
    assert rerun.termination == "already_terminal"
    assert rerun.success is True
    assert len(provider.calls) == calls_before


# ---------------------------------------------------------------------------
# Item 26: clarification happens before execution
# ---------------------------------------------------------------------------


async def test_clarification_blocks_execution_until_answered() -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=3)
    provider = RoutedMockProvider(
        GoalAnalysis=[CLARIFY_ANALYSIS],
        Plan=[CLARIFY_PLAN],
        TaskResult=[TaskResult(reason_summary="Itinerary drafted.", data={})],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="Itinerary complete.",
                constraint_verdicts=[],
                goal_outcome="satisfied",
            )
        ],
        SynthesisResult=[SYNTH],
    )
    orchestrator = build_orchestrator(controller, provider)
    workflow = controller.create_workflow("Plan a trip without dates.")

    # Round 1: analysis demands clarification - execution must NOT start
    first = await orchestrator.run(workflow.id)
    assert first.termination == "awaiting_clarification"
    midway = controller.get_workflow(workflow.id)
    assert midway.state is S.AWAITING_CLARIFICATION
    assert midway.plan is None
    assert midway.observations == []
    assert midway.clarification_questions == [ANSWER_KEY]
    types = [e.type for e in controller.events(workflow.id)]
    assert E.CLARIFICATION_REQUESTED in types
    assert E.EXECUTION_STARTED not in types
    # exactly one LLM call so far: the analysis
    assert len(provider.calls) == 1

    # Round 2: answer supplied - plan, execute, validate, synthesize
    second = await orchestrator.run(
        workflow.id, answers={ANSWER_KEY: "Dec 1-4"}
    )
    assert second.termination == "validated"
    final = controller.get_workflow(workflow.id)
    assert final.state is S.COMPLETED
    assert final.clarifications == {ANSWER_KEY: "Dec 1-4"}
    assert final.final_answer == SYNTH.final_answer

    types = [e.type for e in controller.events(workflow.id)]
    assert E.CLARIFICATION_ANSWERED in types
    assert E.EXECUTION_STARTED in types
    assert E.WORKFLOW_COMPLETED in types
    assert types.index(E.CLARIFICATION_REQUESTED) < types.index(
        E.EXECUTION_STARTED
    )

    # The planner saw the answer as binding context
    plan_calls = [c for c in provider.calls if c.schema_name == "Plan"]
    assert len(plan_calls) == 1
    assert "Dec 1-4" in plan_calls[0].prompt


# ---------------------------------------------------------------------------
# Item 30: the offline demo
# ---------------------------------------------------------------------------


async def test_demo_scenario_end_to_end() -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=3)
    provider = create_demo_provider()
    orchestrator = build_orchestrator(controller, provider)

    workflow = controller.create_workflow(DEMO_GOAL)
    report = await orchestrator.run(workflow.id)

    assert report.success is True
    assert report.termination == "validated"
    assert report.replans == 1
    final = controller.get_workflow(workflow.id)
    assert final.state is S.COMPLETED
    assert final.synthesis.final_answer == DEMO_SYNTHESIS.final_answer
    # constraint claims cross-check against the PASS validation
    assert final.synthesis.constraints_satisfied == ["c1"]
    assert final.validation.satisfied_constraints == ["c1"]
    # the demo is fully offline and keyless
    assert provider.model_id.startswith("mock/")

