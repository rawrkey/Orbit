"""Phase 5 Validator tests: deterministic-first goal validation.

The LLM proposes; Python disposes. These tests prove that numeric rules
are recomputed independently of the LLM's claims, that fabricated evidence
degrades to INSUFFICIENT (never PASS), and that the status composition is
Python-owned. No API key, no network.
"""

from __future__ import annotations

import pytest

from app.agent.validator import (
    InconsistentValidationError,
    Validator,
    ValidatorError,
    evaluate_rule,
    extract_number,
    validate_result_consistency,
)
from app.llm.base import LLMError, LLMOutputInvalidError
from app.llm.mock import MockProvider
from app.llm.schemas import ConstraintVerdict, ValidationProposal
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    EvidenceRef,
    GoalAnalysis,
    Observation,
    ObservationSource,
    Plan,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
)

GOAL = "Plan a 3-day Mumbai trip for 2 people under 8000 INR"


def budget_constraint() -> Constraint:
    return Constraint(
        id="c1",
        name="budget",
        type=ConstraintType.BUDGET,
        description="Total stay under 8000 INR",
        rule=ConstraintRule(
            operator=ConstraintOperator.LTE, value=8000.0, unit="INR"
        ),
    )


def make_analysis(constraints: list[Constraint] | None = None) -> GoalAnalysis:
    return GoalAnalysis(
        objective="Plan a 3-day Mumbai trip under 8000 INR",
        reason_summary="Duration and budget are explicit.",
        constraints=[budget_constraint()] if constraints is None else constraints,
    )


PLAN = Plan(
    tasks=[
        Task(
            id="t1",
            description="Compute the trip total",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="total computed",
        ),
        Task(
            id="t2",
            description="Draft the itinerary",
            success_criteria="itinerary drafted",
        ),
    ]
)


def make_workflow(
    observations: list[Observation] | None = None,
    *,
    constraints: list[Constraint] | None = None,
) -> Workflow:
    return Workflow(
        id="wf1",
        goal=GOAL,
        analysis=make_analysis(constraints),
        plan=PLAN,
        task_states={"t1": TaskStatus.COMPLETED, "t2": TaskStatus.COMPLETED},
        observations=list(observations or []),
    )


def tool_observation(result: float, *, task_id: str = "t1") -> Observation:
    return Observation(
        task_id=task_id,
        content_summary=f"Trip total is {result} INR.",
        raw={"expression": "9000+0", "result": result},
        source=ObservationSource.TOOL,
        tool_name="calculator",
    )


def rule_verdict(
    *,
    outcome: str = "satisfied",
    task_id: str = "t1",
    evidence_field: str | None = "result",
    constraint_id: str = "c1",
) -> ConstraintVerdict:
    return ConstraintVerdict(
        constraint_id=constraint_id,
        outcome=outcome,
        evidence=[EvidenceRef(task_id=task_id, note="total observed")],
        evidence_field=evidence_field,
    )


def proposal(
    verdicts: list[ConstraintVerdict] | None = None,
    *,
    goal_outcome: str = "satisfied",
    summary: str = "Evidence reviewed against the goal.",
) -> ValidationProposal:
    return ValidationProposal(
        reason_summary=summary,
        constraint_verdicts=list(verdicts or []),
        goal_outcome=goal_outcome,
    )


async def validate(
    responses: list[ValidationProposal], workflow: Workflow | None = None
) -> ValidationResult:
    wf = workflow or make_workflow([tool_observation(9000.0)])
    return await Validator(MockProvider(responses)).validate(wf)


# ---------------------------------------------------------------------------
# Deterministic primitives
# ---------------------------------------------------------------------------


def test_evaluate_rule_applies_every_operator() -> None:
    rule = lambda op, value: ConstraintRule(operator=op, value=value)  # noqa: E731
    assert evaluate_rule(8000.0, rule(ConstraintOperator.LTE, 8000.0)) is True
    assert evaluate_rule(8000.1, rule(ConstraintOperator.LTE, 8000.0)) is False
    assert evaluate_rule(7999.9, rule(ConstraintOperator.LTE, 8000.0)) is True
    assert evaluate_rule(8000.0, rule(ConstraintOperator.GTE, 8000.0)) is True
    assert evaluate_rule(7999.9, rule(ConstraintOperator.GTE, 8000.0)) is False
    assert evaluate_rule(42.0, rule(ConstraintOperator.EQ, 42.0)) is True
    assert evaluate_rule(42.000000001, rule(ConstraintOperator.EQ, 42.0)) is True
    assert evaluate_rule(43.0, rule(ConstraintOperator.EQ, 42.0)) is False
    assert evaluate_rule(5.0, rule(ConstraintOperator.MAX, 5.0)) is True
    assert evaluate_rule(5.1, rule(ConstraintOperator.MAX, 5.0)) is False
    assert evaluate_rule(5.0, rule(ConstraintOperator.MIN, 5.0)) is True
    assert evaluate_rule(4.9, rule(ConstraintOperator.MIN, 5.0)) is False


def test_extract_number_walks_paths_and_rejects_garbage() -> None:
    raw = {"result": 9000.0, "nested": {"values": [1, 2, {"v": 3.5}]}, "flag": True}
    assert extract_number(raw, "result") == 9000.0
    assert extract_number(raw, "nested.values.2.v") == 3.5
    assert extract_number(raw, "missing") is None
    assert extract_number(raw, "nested.values.9") is None
    assert extract_number(raw, "nested.values.x") is None
    assert extract_number(raw, "flag") is None  # bools are not numbers
    assert extract_number(raw, "nested") is None  # objects are not numbers
    assert extract_number(raw, "") is None
    assert extract_number(None, "result") is None


# ---------------------------------------------------------------------------
# Deterministic override (the core guarantee)
# ---------------------------------------------------------------------------


async def test_deterministic_violation_overrides_llm_claim() -> None:
    """The LLM claims satisfied; Python recomputes and FAILs anyway."""
    result = await validate(
        [proposal([rule_verdict(outcome="satisfied")], goal_outcome="satisfied")]
    )
    assert result.status is ValidationStatus.FAIL
    assert result.violated_constraints == ["c1"]
    assert result.satisfied_constraints == []
    assert result.replanning_required is True


async def test_rule_pass_yields_satisfied_and_pass() -> None:
    workflow = make_workflow([tool_observation(7500.0)])
    result = await validate(
        [proposal([rule_verdict()], goal_outcome="satisfied")], workflow
    )
    assert result.status is ValidationStatus.PASS
    assert result.satisfied_constraints == ["c1"]
    assert result.violated_constraints == []
    assert result.replanning_required is False


async def test_claimed_violation_with_rule_pass_is_satisfied() -> None:
    """Even a pessimistic LLM claim cannot FAIL a rule that passes."""
    workflow = make_workflow([tool_observation(7500.0)])
    result = await validate(
        [proposal([rule_verdict(outcome="violated")], goal_outcome="not_satisfied")],
        workflow,
    )
    assert result.satisfied_constraints == ["c1"]
    # goal_outcome not_satisfied still fails the goal (composition order).
    assert result.status is ValidationStatus.FAIL


async def test_missing_evidence_field_yields_insufficient_not_pass() -> None:
    workflow = make_workflow([tool_observation(7500.0)])
    result = await validate(
        [proposal([rule_verdict(evidence_field=None)])], workflow
    )
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.unresolved_constraints == ["c1"]


async def test_latest_observation_supersedes_earlier_for_same_task() -> None:
    """After a replan a task has several observations; only the latest counts."""
    workflow = make_workflow(
        [tool_observation(9000.0), tool_observation(7500.0)]
    )
    result = await validate([proposal([rule_verdict()])], workflow)
    assert result.status is ValidationStatus.PASS  # 7500, not the stale 9000


async def test_unextractable_value_yields_insufficient() -> None:
    """Evidence exists but the field is absent - never guess a number."""
    workflow = make_workflow(
        [Observation(task_id="t1", content_summary="done", raw={"other": 1})]
    )
    result = await validate([proposal([rule_verdict()])], workflow)
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.unresolved_constraints == ["c1"]


# ---------------------------------------------------------------------------
# Evidence fabrication and coverage
# ---------------------------------------------------------------------------


async def test_fabricated_evidence_degrades_to_insufficient() -> None:
    """Evidence pointing at a task with no observation is dropped."""
    workflow = make_workflow([tool_observation(7500.0)])
    result = await validate(
        [proposal([rule_verdict(task_id="ghost")])], workflow
    )
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.unresolved_constraints == ["c1"]
    assert all(ref.task_id != "ghost" for ref in result.evidence)


async def test_missing_verdict_becomes_unresolved() -> None:
    result = await validate([proposal([], goal_outcome="satisfied")])
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.unresolved_constraints == ["c1"]


async def test_semantic_constraint_uses_llm_but_requires_real_evidence() -> None:
    constraints = [
        Constraint(
            id="c2",
            type=ConstraintType.PREFERENCE,
            description="Itinerary should be family friendly",
        )
    ]
    workflow = make_workflow(
        [
            Observation(
                task_id="t2",
                content_summary="Itinerary covers parks and museums.",
                source=ObservationSource.REASONING,
            )
        ],
        constraints=constraints,
    )
    verdict = ConstraintVerdict(
        constraint_id="c2",
        outcome="satisfied",
        evidence=[EvidenceRef(task_id="t2", note="itinerary reviewed")],
    )
    result = await validate([proposal([verdict], goal_outcome="satisfied")], workflow)
    assert result.status is ValidationStatus.PASS
    assert result.satisfied_constraints == ["c2"]


async def test_semantic_claim_without_evidence_is_insufficient() -> None:
    constraints = [
        Constraint(
            id="c2",
            type=ConstraintType.PREFERENCE,
            description="Itinerary should be family friendly",
        )
    ]
    workflow = make_workflow([], constraints=constraints)
    verdict = ConstraintVerdict(
        constraint_id="c2", outcome="satisfied", evidence=[]
    )
    result = await validate([proposal([verdict], goal_outcome="satisfied")], workflow)
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.unresolved_constraints == ["c2"]


# ---------------------------------------------------------------------------
# Status composition
# ---------------------------------------------------------------------------


async def test_violation_wins_over_unresolved() -> None:
    constraints = [
        budget_constraint(),
        Constraint(
            id="c2", type=ConstraintType.COUNT, description="At least 3 days"
        ),
    ]
    workflow = make_workflow(
        [tool_observation(9000.0)], constraints=constraints
    )
    result = await validate(
        [proposal([rule_verdict(), rule_verdict(constraint_id="c2", outcome="insufficient")])],
        workflow,
    )
    assert result.status is ValidationStatus.FAIL
    assert result.violated_constraints == ["c1"]
    assert result.unresolved_constraints == ["c2"]


async def test_goal_not_satisfied_fails_without_constraints() -> None:
    workflow = make_workflow([], constraints=[])
    result = await validate([proposal([], goal_outcome="not_satisfied")], workflow)
    assert result.status is ValidationStatus.FAIL
    assert result.violated_constraints == []


async def test_no_evidence_at_all_is_never_pass() -> None:
    workflow = make_workflow([], constraints=[])
    result = await validate([proposal([], goal_outcome="satisfied")], workflow)
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.replanning_required is True


async def test_insufficient_evidence_for_goal_is_not_pass() -> None:
    workflow = make_workflow([], constraints=[])
    result = await validate([proposal([], goal_outcome="insufficient")], workflow)
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.replanning_required is True


# ---------------------------------------------------------------------------
# Consistency checker
# ---------------------------------------------------------------------------


def test_consistency_rejects_pass_with_violated_constraints() -> None:
    result = ValidationResult(
        status=ValidationStatus.PASS,
        reason_summary="All good.",
        violated_constraints=["c1"],
    )
    with pytest.raises(InconsistentValidationError, match="violated"):
        validate_result_consistency(result, make_analysis())


def test_consistency_rejects_unknown_and_overlapping_ids() -> None:
    result = ValidationResult(
        status=ValidationStatus.FAIL,
        reason_summary="bad",
        satisfied_constraints=["c1", "ghost"],
        violated_constraints=["c1"],
    )
    with pytest.raises(InconsistentValidationError) as exc:
        validate_result_consistency(result, make_analysis())
    assert "unknown constraint ids" in str(exc.value)
    assert "both satisfied and violated" in str(exc.value)


def test_consistency_rejects_fail_without_reason() -> None:
    result = ValidationResult(
        status=ValidationStatus.FAIL, reason_summary="bad", violated_constraints=[]
    )
    with pytest.raises(InconsistentValidationError, match="FAIL requires"):
        validate_result_consistency(result, make_analysis(), goal_outcome="satisfied")


def test_consistency_accepts_goal_level_fail() -> None:
    result = ValidationResult(
        status=ValidationStatus.FAIL, reason_summary="goal not met"
    )
    validate_result_consistency(result, make_analysis(), goal_outcome="not_satisfied")


# ---------------------------------------------------------------------------
# Contracts and guards
# ---------------------------------------------------------------------------


async def test_validator_requires_analysis() -> None:
    workflow = Workflow(id="wf", goal=GOAL)
    with pytest.raises(ValidatorError, match="no goal analysis"):
        await Validator(MockProvider([])).validate(workflow)


async def test_validator_requires_plan() -> None:
    workflow = Workflow(id="wf", goal=GOAL, analysis=make_analysis())
    with pytest.raises(ValidatorError, match="no plan recorded"):
        await Validator(MockProvider([])).validate(workflow)


async def test_llm_errors_propagate_untouched() -> None:
    with pytest.raises(LLMError, match="no scripted"):
        await validate([])


async def test_malformed_proposal_raises_output_invalid() -> None:
    workflow = make_workflow([tool_observation(7500.0)])
    provider = MockProvider([{"goal_outcome": "satisfied"}])  # missing fields
    with pytest.raises(LLMOutputInvalidError):
        await Validator(provider).validate(workflow)


async def test_validator_does_not_mutate_the_workflow() -> None:
    workflow = make_workflow([tool_observation(9000.0)])
    before = workflow.model_copy(deep=True)
    await validate([proposal([rule_verdict()])], workflow)
    assert workflow == before


async def test_prompt_is_bounded_and_carries_the_evidence() -> None:
    provider = MockProvider([proposal([rule_verdict()])])
    workflow = make_workflow([tool_observation(9000.0)])
    await Validator(provider).validate(workflow)
    prompt = provider.calls[0].prompt
    assert GOAL in prompt
    assert '"c1"' in prompt
    assert '"result"' in prompt or "9000" in prompt
    assert len(prompt) < 50_000


async def test_zero_temperature_and_no_chain_of_thought_in_result() -> None:
    provider = MockProvider([proposal([rule_verdict()])])
    workflow = make_workflow([tool_observation(7500.0)])
    result = await Validator(provider, temperature=0.0).validate(workflow)
    assert provider.calls[0].temperature == 0.0
    assert "chain" not in result.reason_summary.lower()
