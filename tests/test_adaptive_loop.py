"""Phase 5 adaptive loop tests: execute -> validate -> replan -> execute.

Covers the critical adaptive path with an EXACT event sequence (goal ->
FAIL -> replan -> re-execute -> PASS -> COMPLETED), plus every terminal
failure mode: execution failure without replan, budget exhaustion, invalid
replan proposals, LLM errors during validation/replanning, and the
defensive pass bound. Fully offline; no API key.
"""

from __future__ import annotations

import pytest

from app.agent.adaptive_loop import (
    AdaptiveLoop,
    AdaptiveLoopError,
    AdaptiveRunReport,
)
from app.agent.controller import WorkflowController
from app.agent.executor import Executor
from app.agent.replanner import Replanner
from app.agent.validator import Validator
from app.config.settings import load_settings
from app.llm.base import LLMError, T
from app.llm.mock import LLMCallRecord, MockProvider
from app.llm.schemas import TaskResult, ToolDecision, ValidationProposal
from app.state.events import EventType
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    EvidenceRef,
    GoalAnalysis,
    ObservationSource,
    Plan,
    Task,
    TaskStatus,
    ValidationStatus,
    WorkflowState,
)
from app.state.store import InMemoryStateStore, WorkflowNotFoundError
from app.tools.builtin import create_default_registry

S = WorkflowState
E = EventType
GOAL = "Plan a 3-day Mumbai trip for 2 people under 8000 INR"

ANALYSIS = GoalAnalysis(
    objective="Plan a 3-day Mumbai trip under 8000 INR",
    reason_summary="Duration and budget are explicit.",
    constraints=[
        Constraint(
            id="c1",
            name="budget",
            type=ConstraintType.BUDGET,
            description="Total stay under 8000 INR",
            rule=ConstraintRule(
                operator=ConstraintOperator.LTE, value=8000.0, unit="INR"
            ),
        )
    ],
)

NO_CONSTRAINT_ANALYSIS = GoalAnalysis(
    objective="Draft a 3-day Mumbai itinerary",
    reason_summary="No hard constraints.",
    constraints=[],
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
        PLAN.tasks[1],  # unchanged -> preserved as completed work
    ],
)

SINGLE_PLAN = Plan(
    version=1,
    tasks=[
        Task(
            id="t2",
            title="Draft itinerary",
            description="Draft the day-by-day itinerary",
            success_criteria="itinerary drafted",
        )
    ],
)

# A8: plan A is reasoning-only (no numeric "result" field) -> INSUFFICIENT,
# then the replan adds a tool task that gathers the missing evidence.
ESTIMATE_PLAN = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Estimate stay cost",
            description="Estimate the cheapest 3-day stay cost from memory",
            constraint_ids=["c1"],
            success_criteria="cost estimated",
        )
    ],
)

TOOL_PLAN = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Compute stay cost",
            description="Compute the cheapest 3-day stay cost with the calculator",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="cost computed",
        )
    ],
)

# A10: each replan proposal must differ from its predecessor.
PLAN_V3 = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Find hostel total",
            description="Recompute the total using hostel options",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="hostel total computed",
        ),
        PLAN.tasks[1],
    ],
)


# ---------------------------------------------------------------------------
# Scripted, schema-routed provider (call order across stages stays stable)
# ---------------------------------------------------------------------------


class RoutedProvider:
    """Deterministic LLM provider that dispatches by requested schema."""

    def __init__(self, **queues: list) -> None:
        self._queues: dict[str, list] = {
            name: list(items) for name, items in queues.items()
        }
        self.calls: list[LLMCallRecord] = []

    @property
    def model_id(self) -> str:
        return "mock/routed"

    async def complete(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
    ) -> T:
        self.calls.append(
            LLMCallRecord(
                prompt=prompt,
                schema_name=schema.__name__,
                system=system,
                temperature=temperature,
                timeout_s=timeout_s,
            )
        )
        queue = self._queues.get(schema.__name__)
        if not queue:
            raise LLMError(f"no scripted {schema.__name__} responses left")
        return queue.pop(0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def calc_decision(expression: str) -> ToolDecision:
    return ToolDecision(
        tool_name="calculator",
        arguments={"expression": expression},
        reason_summary=f"Evaluate {expression}.",
    )


REASONING = TaskResult(
    reason_summary="Itinerary covers 3 days in South Mumbai.",
    data={"days": 3},
)


def claim(*, evidence_task: str = "t1") -> ValidationProposal:
    """A proposal CLAIMING the budget is satisfied (Python decides)."""
    return ValidationProposal(
        reason_summary="Trip total observed from the calculator.",
        constraint_verdicts=[
            {
                "constraint_id": "c1",
                "outcome": "satisfied",
                "evidence": [{"task_id": evidence_task, "note": "total"}],
                "evidence_field": "result",
            }
        ],
        goal_outcome="satisfied",
    )


def pass_claim() -> ValidationProposal:
    return ValidationProposal(
        reason_summary="Trip total 7500 INR respects the 8000 INR budget.",
        constraint_verdicts=[
            {
                "constraint_id": "c1",
                "outcome": "satisfied",
                "evidence": [{"task_id": "t1", "note": "total 7500"}],
                "evidence_field": "result",
            }
        ],
        goal_outcome="satisfied",
    )


def blind_claim() -> ValidationProposal:
    """No constraint verdicts -> c1 unresolved -> INSUFFICIENT, never PASS."""
    return ValidationProposal(
        reason_summary="Insufficient price coverage.",
        constraint_verdicts=[],
        goal_outcome="insufficient",
    )


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


@pytest.fixture
def registry():
    return create_default_registry(load_settings({}))


def ready(
    controller: WorkflowController,
    *,
    plan: Plan = PLAN,
    analysis: GoalAnalysis = ANALYSIS,
) -> str:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, analysis)
    controller.record_plan(workflow.id, plan)
    return workflow.id


def make_loop(
    controller, registry, provider: RoutedProvider, **kwargs
) -> AdaptiveLoop:
    return AdaptiveLoop(
        controller,
        Executor(controller, provider, registry),
        Validator(provider),
        Replanner(provider),
        **kwargs,
    )


def event_types(controller, workflow_id: str) -> list[EventType]:
    return [e.type for e in controller.events(workflow_id)]


def count(controller, workflow_id: str, event_type: EventType) -> int:
    return sum(1 for e in controller.events(workflow_id) if e.type is event_type)


# ---------------------------------------------------------------------------
# Critical path: FAIL -> replan -> PASS, exact event sequence
# ---------------------------------------------------------------------------

PRE_EVENTS = [
    E.WORKFLOW_STARTED,
    E.STATE_CHANGED,  # RECEIVED -> ANALYZING
    E.GOAL_ANALYZED,
    E.STATE_CHANGED,  # ANALYZING -> PLANNING
    E.WORKFLOW_PLANNED,
    E.STATE_CHANGED,  # PLANNING -> READY
]

EXEC_PASS_1_EVENTS = [
    E.STATE_CHANGED,  # READY -> EXECUTING
    E.EXECUTION_STARTED,
    E.TASK_STARTED,  # t1 (tool)
    E.TOOL_DECISION_PROPOSED,
    E.TOOL_CALLED,
    E.TOOL_COMPLETED,
    E.OBSERVATION_CREATED,
    E.TASK_COMPLETED,
    E.TASK_STARTED,  # t2 (reasoning)
    E.OBSERVATION_CREATED,
    E.TASK_COMPLETED,
    E.STATE_CHANGED,  # EXECUTING -> AWAITING_VALIDATION
    E.EXECUTION_FINISHED,
]

VALIDATION_FAILED_EVENTS = [
    E.STATE_CHANGED,  # AWAITING_VALIDATION -> VALIDATING
    E.VALIDATION_STARTED,
    E.VALIDATION_FAILED,
]

REPLAN_EVENTS = [
    E.STATE_CHANGED,  # VALIDATING -> REPLANNING
    E.REPLANNING_STARTED,
    E.WORKFLOW_REPLANNED,
    E.REPLAN_PROPOSED,
    E.REPLAN_ACCEPTED,
    E.STATE_CHANGED,  # REPLANNING -> PLANNING
    E.WORKFLOW_PLANNED,
    E.STATE_CHANGED,  # PLANNING -> READY
]

EXEC_PASS_2_EVENTS = [
    E.STATE_CHANGED,  # READY -> EXECUTING
    E.EXECUTION_STARTED,
    E.TASK_STARTED,  # t1 again (t2 is preserved COMPLETED)
    E.TOOL_DECISION_PROPOSED,
    E.TOOL_CALLED,
    E.TOOL_COMPLETED,
    E.OBSERVATION_CREATED,
    E.TASK_COMPLETED,
    E.STATE_CHANGED,  # EXECUTING -> AWAITING_VALIDATION
    E.EXECUTION_FINISHED,
]

VALIDATION_PASSED_EVENTS = [
    E.STATE_CHANGED,  # AWAITING_VALIDATION -> VALIDATING
    E.VALIDATION_STARTED,
    E.VALIDATION_PASSED,
]

COMPLETION_EVENTS = [
    E.STATE_CHANGED,  # VALIDATING -> COMPLETED
    E.WORKFLOW_COMPLETED,
]

CRITICAL_PATH_EVENTS = (
    PRE_EVENTS
    + EXEC_PASS_1_EVENTS
    + VALIDATION_FAILED_EVENTS
    + REPLAN_EVENTS
    + EXEC_PASS_2_EVENTS
    + VALIDATION_PASSED_EVENTS
    + COMPLETION_EVENTS
)


async def test_critical_path_fail_replan_pass_exact_events(
    controller, registry
) -> None:
    """total=9000 -> FAIL -> replan -> 7500 -> PASS -> COMPLETED."""
    workflow_id = ready(controller)
    provider = RoutedProvider(
        ToolDecision=[calc_decision("9000+0"), calc_decision("7500+0")],
        TaskResult=[REASONING],
        ValidationProposal=[claim(), pass_claim()],
        Plan=[PLAN_V2],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert isinstance(report, AdaptiveRunReport)
    assert report.success is True
    assert report.final_state is S.COMPLETED
    assert report.termination == "validated"
    assert (report.executions, report.validations, report.replans) == (2, 2, 1)
    assert report.failure is None

    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.COMPLETED
    assert workflow.validation is not None
    assert workflow.validation.status.value == "PASS"
    assert workflow.validation.violated_constraints == []
    assert workflow.replan_count == 1
    assert workflow.plan is not None and workflow.plan.version == 2
    assert [revision.plan.version for revision in workflow.plan_history] == [1]
    assert workflow.task_states == {
        "t1": TaskStatus.COMPLETED,
        "t2": TaskStatus.COMPLETED,
    }
    # t2 was preserved across the replan: it ran exactly once.
    t2_observations = [o for o in workflow.observations if o.task_id == "t2"]
    t1_observations = [o for o in workflow.observations if o.task_id == "t1"]
    assert len(t2_observations) == 1
    assert len(t1_observations) == 2  # original (9000) + replanned (7500)
    assert t1_observations[0].raw is not None
    assert t1_observations[0].raw["result"] == 9000.0
    assert t1_observations[1].raw is not None
    assert t1_observations[1].raw["result"] == 7500.0

    assert event_types(controller, workflow_id) == CRITICAL_PATH_EVENTS


async def test_pass_on_first_attempt_never_replans(controller, registry) -> None:
    workflow_id = ready(controller, plan=SINGLE_PLAN, analysis=NO_CONSTRAINT_ANALYSIS)
    provider = RoutedProvider(
        TaskResult=[REASONING],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="Itinerary drafted for all 3 days.",
                constraint_verdicts=[],
                goal_outcome="satisfied",
            )
        ],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.success is True
    assert report.termination == "validated"
    assert (report.executions, report.validations, report.replans) == (1, 1, 0)
    assert event_types(controller, workflow_id) == (
        PRE_EVENTS
        + [
            E.STATE_CHANGED,
            E.EXECUTION_STARTED,
            E.TASK_STARTED,
            E.OBSERVATION_CREATED,
            E.TASK_COMPLETED,
            E.STATE_CHANGED,
            E.EXECUTION_FINISHED,
        ]
        + VALIDATION_PASSED_EVENTS
        + COMPLETION_EVENTS
    )
    assert count(controller, workflow_id, E.REPLANNING_STARTED) == 0
    assert count(controller, workflow_id, E.WORKFLOW_FAILED) == 0


async def test_execution_failure_terminates_without_replan(
    controller, registry
) -> None:
    plan = Plan(tasks=[PLAN.tasks[0]])
    workflow_id = ready(controller, plan=plan)
    provider = RoutedProvider(
        ToolDecision=[
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "stays"},
                reason_summary="Search for stays.",
            )
        ],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.success is False
    assert report.termination == "execution_failed"
    assert (report.executions, report.validations, report.replans) == (1, 0, 0)
    assert report.final_state is S.FAILED
    assert report.failure is not None
    assert report.failure.error_code == "UNAVAILABLE"

    types = event_types(controller, workflow_id)
    assert E.VALIDATION_STARTED not in types  # no validation after failure
    assert E.REPLANNING_STARTED not in types  # and no replan
    assert E.WORKFLOW_FAILED in types


async def test_insufficient_evidence_triggers_replan(controller, registry) -> None:
    workflow_id = ready(controller)
    provider = RoutedProvider(
        ToolDecision=[calc_decision("9000+0"), calc_decision("7500+0")],
        TaskResult=[REASONING],
        ValidationProposal=[blind_claim(), pass_claim()],
        Plan=[PLAN_V2],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.success is True
    assert report.termination == "validated"
    assert (report.executions, report.validations, report.replans) == (2, 2, 1)
    workflow = controller.get_workflow(workflow_id)
    assert workflow.validation is not None
    # First verdict is kept out of state; the final stored one is PASS.
    assert workflow.validation.status.value == "PASS"
    assert count(controller, workflow_id, E.VALIDATION_FAILED) == 1
    assert count(controller, workflow_id, E.REPLANNING_STARTED) == 1
    assert count(controller, workflow_id, E.VALIDATION_PASSED) == 1


async def test_replan_budget_exhausted_fails_with_evidence(
    controller, registry
) -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=1)
    workflow_id = ready(controller)
    provider = RoutedProvider(
        ToolDecision=[calc_decision("9000+0"), calc_decision("9000+0")],
        TaskResult=[REASONING],
        ValidationProposal=[claim(), claim()],
        Plan=[PLAN_V2],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.success is False
    assert report.termination == "replan_exhausted"
    assert (report.executions, report.validations, report.replans) == (2, 2, 1)
    assert report.final_state is S.FAILED
    assert report.failure is not None
    assert report.failure.error_code == "REPLAN_LIMIT_EXCEEDED"
    assert "replanning attempt" in report.failure.detail

    workflow = controller.get_workflow(workflow_id)
    assert workflow.replan_count == 1
    assert workflow.validation is not None
    assert workflow.validation.status.value == "FAIL"  # evidence kept
    assert count(controller, workflow_id, E.REPLANNING_EXHAUSTED) == 1
    assert count(controller, workflow_id, E.REPLANNING_STARTED) == 1  # only one
    assert count(controller, workflow_id, E.WORKFLOW_FAILED) == 1
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.FAILED


async def test_invalid_replan_terminates_with_rejected_event(
    controller, registry
) -> None:
    workflow_id = ready(controller)
    provider = RoutedProvider(
        ToolDecision=[calc_decision("9000+0")],
        TaskResult=[REASONING],
        ValidationProposal=[claim()],
        Plan=[PLAN, PLAN],  # identical proposals -> rejected after repair
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.success is False
    assert report.termination == "replan_invalid"
    assert (report.executions, report.validations, report.replans) == (1, 1, 0)
    assert report.final_state is S.FAILED
    assert report.failure is not None
    assert report.failure.error_code == "REPLAN_INVALID"
    assert report.failure.stage is S.REPLANNING

    rejected = [
        e
        for e in controller.events(workflow_id)
        if e.type is E.REPLAN_REJECTED
    ]
    assert len(rejected) == 1
    assert rejected[0].payload["issue_count"] >= 1
    assert count(controller, workflow_id, E.REPLANNING_EXHAUSTED) == 0
    workflow = controller.get_workflow(workflow_id)
    assert workflow.plan is not None and workflow.plan.version == 1  # untouched
    assert workflow.plan_history == []


async def test_validation_llm_error_terminates_without_replan(
    controller, registry
) -> None:
    workflow_id = ready(controller, plan=SINGLE_PLAN)
    provider = RoutedProvider(TaskResult=[REASONING])  # no ValidationProposal
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.termination == "validation_error"
    assert report.final_state is S.FAILED
    assert report.failure is not None
    assert report.failure.stage is S.VALIDATING
    assert report.failure.error_code == "LLM_ERROR"
    assert (report.validations, report.replans) == (0, 0)
    assert E.REPLANNING_STARTED not in event_types(controller, workflow_id)


async def test_replanner_llm_error_terminates(controller, registry) -> None:
    workflow_id = ready(controller)
    provider = RoutedProvider(
        ToolDecision=[calc_decision("9000+0")],
        TaskResult=[REASONING],
        ValidationProposal=[claim()],
        # no Plan scripted -> replanner LLM call raises
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.termination == "replan_error"
    assert report.final_state is S.FAILED
    assert report.failure is not None
    assert report.failure.stage is S.REPLANNING
    assert report.failure.error_code == "LLM_ERROR"
    assert count(controller, workflow_id, E.REPLAN_REJECTED) == 0
    assert count(controller, workflow_id, E.REPLANNING_STARTED) == 1


async def test_defensive_pass_bound_raises(controller, registry) -> None:
    workflow_id = ready(controller)
    provider = RoutedProvider()
    loop = make_loop(controller, registry, provider, max_passes=0)
    with pytest.raises(AdaptiveLoopError, match="bound of 0"):
        await loop.run(workflow_id)
    assert provider.calls == []  # bound checked before any LLM call


async def test_unknown_workflow_raises(controller, registry) -> None:
    provider = RoutedProvider()
    loop = make_loop(controller, registry, provider)
    with pytest.raises(WorkflowNotFoundError):
        await loop.run("does-not-exist")


async def test_report_is_structured_data() -> None:
    report = AdaptiveRunReport(
        workflow_id="wf",
        success=False,
        final_state=S.FAILED,
        executions=1,
        validations=0,
        replans=0,
        termination="execution_failed",
    )
    assert report.model_dump()["termination"] == "execution_failed"
    assert report.failure is None


# ---------------------------------------------------------------------------
# A6/A8/A10: exhaustion keeps history, INSUFFICIENT routes to evidence
# gathering, and the execution bound is N+1
# ---------------------------------------------------------------------------


async def test_exhaustion_preserves_history_evidence_and_violations(
    controller, registry
) -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=1)
    workflow_id = ready(controller)
    provider = RoutedProvider(
        ToolDecision=[calc_decision("9000+0"), calc_decision("9000+0")],
        TaskResult=[REASONING],
        ValidationProposal=[claim(), claim()],
        Plan=[PLAN_V2],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.termination == "replan_exhausted"
    assert (report.executions, report.validations, report.replans) == (2, 2, 1)

    workflow = controller.get_workflow(workflow_id)
    # Append-only plan history survives the failure.
    assert len(workflow.plan_history) == 1
    revision = workflow.plan_history[0]
    assert revision.validation is not None
    assert revision.validation.status is ValidationStatus.FAIL
    assert revision.validation.violated_constraints == ["c1"]
    assert revision.new_plan_version == 2
    assert workflow.plan is not None and workflow.plan.version == 2
    # Evidence survives: t1 ran twice (9000, 9000), t2 once (preserved).
    assert len(workflow.observations) == 3
    # The final stored validation still shows what failed.
    assert workflow.validation is not None
    assert workflow.validation.status is ValidationStatus.FAIL
    assert workflow.validation.violated_constraints == ["c1"]
    assert workflow.replan_count == 1


async def test_insufficient_routes_to_evidence_gathering_replan(
    controller, registry
) -> None:
    """Reasoning-only plan -> INSUFFICIENT -> evidence-gathering replan -> PASS."""
    workflow_id = ready(controller, plan=ESTIMATE_PLAN)
    provider = RoutedProvider(
        TaskResult=[
            REASONING,  # no "result" field -> evidence missing, not violated
            REASONING,
        ],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="Cost only estimated, never computed.",
                constraint_verdicts=[
                    {
                        "constraint_id": "c1",
                        "outcome": "satisfied",
                        "evidence": [{"task_id": "t1", "note": "estimate"}],
                        "evidence_field": "result",
                    }
                ],
                goal_outcome="insufficient",
            ),
            pass_claim(),
        ],
        Plan=[TOOL_PLAN],
        ToolDecision=[calc_decision("7500+0")],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    assert report.success is True
    assert report.termination == "validated"
    assert (report.executions, report.validations, report.replans) == (2, 2, 1)
    assert count(controller, workflow_id, E.VALIDATION_FAILED) == 1
    workflow = controller.get_workflow(workflow_id)
    assert workflow.validation is not None
    assert workflow.validation.status is ValidationStatus.PASS

    # The replanner was invoked in evidence_gathering mode.
    plan_calls = [c for c in provider.calls if c.schema_name == "Plan"]
    assert len(plan_calls) == 1
    assert "MODE: evidence_gathering" in plan_calls[0].prompt
    assert '"replan_mode": "evidence_gathering"' in plan_calls[0].prompt


async def test_execution_bound_is_n_plus_one(controller, registry) -> None:
    controller = WorkflowController(InMemoryStateStore(), max_replans=2)
    workflow_id = ready(controller)
    provider = RoutedProvider(
        ToolDecision=[
            calc_decision("9000+0"),
            calc_decision("9000+0"),
            calc_decision("9000+0"),
        ],
        TaskResult=[REASONING] * 3,
        ValidationProposal=[claim(), claim(), claim()],
        Plan=[PLAN_V2, PLAN_V3],
    )
    loop = make_loop(controller, registry, provider)

    report = await loop.run(workflow_id)

    # Exactly N+1 execution passes, then terminal failure - never a loop.
    assert report.termination == "replan_exhausted"
    assert (report.executions, report.validations, report.replans) == (3, 3, 2)
    assert report.executions == 2 + 1  # max_replans + 1
    assert report.final_state is S.FAILED
    assert count(controller, workflow_id, E.REPLANNING_STARTED) == 2
    assert count(controller, workflow_id, E.WORKFLOW_FAILED) == 1
    workflow = controller.get_workflow(workflow_id)
    assert workflow.replan_count == 2
    assert workflow.state is S.FAILED
