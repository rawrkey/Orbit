"""Phase 5 Replanner tests: bounded, validated, targeted replan proposals.

The Replanner returns a *proposed* Plan; it never mutates workflow state.
Every proposal must pass the Planner's deterministic structural validation
PLUS the replan-specific identical-plan rejection, with one bounded repair
attempt. No API key, no network.
"""

from __future__ import annotations

import pytest

from app.agent.planner import PlanValidationError, validate_plan
from app.agent.replanner import (
    ReplanValidationError,
    Replanner,
    ReplannerError,
)
from app.llm.base import LLMError, LLMOutputInvalidError
from app.llm.mock import MockProvider
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    EvidenceRef,
    GoalAnalysis,
    Plan,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
)

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

PLAN_V1 = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Compute trip total",
            description="Compute the trip total from published prices",
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
            title="Compute cheaper trip total",
            description="Recompute the total using cheaper stay options",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="revised total computed",
        ),
        PLAN_V1.tasks[1],  # unchanged -> reusable as completed work
    ],
)

FAILED_VALIDATION = ValidationResult(
    status=ValidationStatus.FAIL,
    reason_summary="Trip total 9000 INR exceeds the 8000 INR budget.",
    satisfied_constraints=[],
    violated_constraints=["c1"],
    evidence=[EvidenceRef(task_id="t1", note="total 9000")],
    replanning_required=True,
)


def make_workflow(
    *,
    task_states: dict[str, TaskStatus] | None = None,
    plan: Plan = PLAN_V1,
) -> Workflow:
    return Workflow(
        id="wf1",
        goal=GOAL,
        analysis=ANALYSIS,
        plan=plan,
        task_states=task_states
        or {"t1": TaskStatus.COMPLETED, "t2": TaskStatus.COMPLETED},
        replan_count=1,
    )


async def propose(
    responses, workflow: Workflow | None = None
) -> Plan:
    wf = workflow or make_workflow()
    return await Replanner(MockProvider(responses)).propose(
        wf, FAILED_VALIDATION
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_valid_proposal_returned_unchanged() -> None:
    result = await propose([PLAN_V2])
    assert result is PLAN_V2 or result == PLAN_V2
    assert result.tasks[0].description.startswith("Recompute")


async def test_workflow_state_is_never_mutated() -> None:
    workflow = make_workflow()
    before = workflow.model_copy(deep=True)
    await Replanner(MockProvider([PLAN_V2])).propose(workflow, FAILED_VALIDATION)
    assert workflow == before


# ---------------------------------------------------------------------------
# Bounded repair loop
# ---------------------------------------------------------------------------


async def test_identical_proposal_is_repaired_on_second_attempt() -> None:
    provider = MockProvider([PLAN_V1, PLAN_V2])
    result = await Replanner(provider).propose(make_workflow(), FAILED_VALIDATION)
    assert result == PLAN_V2
    assert len(provider.calls) == 2
    assert "previous proposal failed validation" in provider.calls[1].prompt
    assert "identical" in provider.calls[1].prompt


async def test_identical_twice_raises_replan_validation_error() -> None:
    provider = MockProvider([PLAN_V1, PLAN_V1])
    with pytest.raises(ReplanValidationError, match="identical"):
        await Replanner(provider).propose(make_workflow(), FAILED_VALIDATION)
    assert len(provider.calls) == 2  # exactly one repair attempt


async def test_structural_issues_are_fed_back_for_repair() -> None:
    broken = Plan(
        tasks=[
            Task(
                id="t1",
                description="Compute total",
                requires_tool=True,
                tool_hint="calculator",
                constraint_ids=["c1"],
                success_criteria="total computed",
                depends_on=["ghost"],
            ),
            PLAN_V1.tasks[1],
        ]
    )
    provider = MockProvider([broken, PLAN_V2])
    result = await Replanner(provider).propose(make_workflow(), FAILED_VALIDATION)
    assert result == PLAN_V2
    prompt = provider.calls[1].prompt
    assert "previous proposal failed validation" in prompt
    assert "unknown task" in prompt


async def test_oversized_proposal_is_rejected_then_repaired() -> None:
    oversized = Plan(
        tasks=[
            Task(
                id=f"t{i}",
                description=f"Step {i}",
                success_criteria=f"step {i} done",
            )
            for i in range(1, 9)  # 8 > DEFAULT_MAX_PLAN_TASKS (7)
        ]
    )
    provider = MockProvider([oversized, PLAN_V2])
    result = await Replanner(provider).propose(make_workflow(), FAILED_VALIDATION)
    assert result == PLAN_V2
    assert "limit is 7" in provider.calls[1].prompt


async def test_replan_error_is_a_plan_validation_error() -> None:
    with pytest.raises(PlanValidationError):
        await propose([PLAN_V1, PLAN_V1])


# ---------------------------------------------------------------------------
# Contracts and guards
# ---------------------------------------------------------------------------


async def test_requires_analysis() -> None:
    workflow = Workflow(id="wf", goal=GOAL, plan=PLAN_V1)
    provider = MockProvider([PLAN_V2])
    with pytest.raises(ReplannerError, match="no goal analysis"):
        await Replanner(provider).propose(workflow, FAILED_VALIDATION)
    assert provider.calls == []


async def test_requires_plan() -> None:
    workflow = Workflow(id="wf", goal=GOAL, analysis=ANALYSIS)
    provider = MockProvider([PLAN_V2])
    with pytest.raises(ReplannerError, match="no plan recorded"):
        await Replanner(provider).propose(workflow, FAILED_VALIDATION)
    assert provider.calls == []


async def test_llm_errors_propagate_without_repair() -> None:
    provider = MockProvider([])  # exhausted -> LLMError
    with pytest.raises(LLMError, match="no scripted"):
        await Replanner(provider).propose(make_workflow(), FAILED_VALIDATION)
    assert len(provider.calls) == 1


async def test_malformed_proposal_raises_output_invalid() -> None:
    provider = MockProvider([{"tasks": "not-a-list"}])
    with pytest.raises(LLMOutputInvalidError):
        await Replanner(provider).propose(make_workflow(), FAILED_VALIDATION)


async def test_prompt_carries_the_failure_and_reuse_instructions() -> None:
    provider = MockProvider([PLAN_V2])
    await Replanner(provider).propose(make_workflow(), FAILED_VALIDATION)
    prompt = provider.calls[0].prompt
    system = provider.calls[0].system or ""
    assert "FAIL" in prompt
    assert '"c1"' in prompt  # violated constraint id
    assert '"t1"' in prompt and '"t2"' in prompt
    assert "EXACT same id" in system  # completed-work reuse instruction
    assert "TARGETED" in system
    assert "9000" in prompt  # validation reason reaches the model
    assert len(prompt) < 50_000


async def test_rejected_proposal_passes_planner_validation_rules() -> None:
    """The proposal the Replanner accepts is structurally valid by design."""
    result = await propose([PLAN_V2])
    validate_plan(result, analysis=ANALYSIS)  # raises if invalid


async def test_invalid_constructor_bounds() -> None:
    provider = MockProvider([])
    with pytest.raises(ValueError, match="max_tasks"):
        Replanner(provider, max_tasks=0)
    with pytest.raises(ValueError, match="repair_attempts"):
        Replanner(provider, repair_attempts=-1)
