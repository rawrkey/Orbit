"""Goal Analyzer tests: strict single call, deterministic consistency checks."""

from __future__ import annotations

import pytest

from app.agent.goal_analyzer import (
    GoalAnalysisError,
    GoalAnalyzer,
    InvalidGoalError,
    validate_analysis,
)
from app.llm.base import LLMOutputInvalidError
from app.llm.mock import MockProvider
from app.state.models import (
    Constraint,
    ConstraintRule,
    ConstraintOperator,
    ConstraintType,
    GoalAnalysis,
)


def make_analysis(**overrides) -> GoalAnalysis:
    base: dict = {
        "objective": "Plan a 3-day Mumbai trip under 8000 INR",
        "reason_summary": "Duration and budget are explicit.",
        "is_feasible": True,
    }
    base.update(overrides)
    return GoalAnalysis(**base)


def make_budget_constraint(id: str = "c1") -> Constraint:
    return Constraint(
        id=id,
        name="budget",
        type=ConstraintType.BUDGET,
        description="Total under 8000 INR",
        rule=ConstraintRule(
            operator=ConstraintOperator.LTE, value=8000, unit="INR"
        ),
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


async def test_blank_goal_rejected_without_any_llm_call() -> None:
    provider = MockProvider()
    analyzer = GoalAnalyzer(provider)

    with pytest.raises(InvalidGoalError):
        await analyzer.analyze("    ")

    assert provider.calls == []


async def test_analyze_returns_validated_analysis_with_one_call() -> None:
    provider = MockProvider([make_analysis()])
    analyzer = GoalAnalyzer(provider)

    result = await analyzer.analyze("Plan a 3-day Mumbai trip under 8000 INR")

    assert result.objective.startswith("Plan a 3-day Mumbai trip")
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call.schema_name == "GoalAnalysis"
    assert call.system is not None and "JSON object" in call.system
    assert "Plan a 3-day Mumbai trip under 8000 INR" in call.prompt


async def test_analyze_rejects_missing_required_field() -> None:
    provider = MockProvider([{"objective": "only an objective"}])
    analyzer = GoalAnalyzer(provider)

    with pytest.raises(LLMOutputInvalidError):
        await analyzer.analyze("Plan a trip")

    assert len(provider.calls) == 1  # no silent retries inside the analyzer


# ---------------------------------------------------------------------------
# Deterministic consistency checks (validate_analysis)
# ---------------------------------------------------------------------------


def test_consistent_analysis_passes_validation() -> None:
    validate_analysis(
        make_analysis(
            constraints=[make_budget_constraint()],
            required_inputs=["travel dates"],
            clarification_required=True,
            optional_inputs=["preferred neighborhood"],
        )
    )


def test_optional_inputs_never_require_clarification() -> None:
    validate_analysis(
        make_analysis(
            required_inputs=[],
            clarification_required=False,
            optional_inputs=["preferred neighborhood"],
        )
    )


def test_clarification_flag_without_required_inputs_is_rejected() -> None:
    analysis = make_analysis(
        required_inputs=[], clarification_required=True
    )
    with pytest.raises(GoalAnalysisError) as excinfo:
        validate_analysis(analysis)
    assert "clarification_required" in excinfo.value.issues[0]


def test_required_inputs_without_clarification_flag_is_rejected() -> None:
    analysis = make_analysis(
        required_inputs=["travel dates"], clarification_required=False
    )
    with pytest.raises(GoalAnalysisError) as excinfo:
        validate_analysis(analysis)
    assert any("required_inputs" in issue for issue in excinfo.value.issues)


def test_duplicate_constraint_ids_are_rejected() -> None:
    analysis = make_analysis(
        constraints=[make_budget_constraint("c1"), make_budget_constraint("c1")]
    )
    with pytest.raises(GoalAnalysisError) as excinfo:
        validate_analysis(analysis)
    assert any("duplicate constraint id" in issue for issue in excinfo.value.issues)


async def test_inconsistent_analysis_fails_after_exactly_one_call() -> None:
    provider = MockProvider(
        [make_analysis(required_inputs=[], clarification_required=True)]
    )
    analyzer = GoalAnalyzer(provider)

    with pytest.raises(GoalAnalysisError):
        await analyzer.analyze("Plan a trip")

    assert len(provider.calls) == 1  # the analyzer never retries itself


def test_multiple_issues_are_all_reported() -> None:
    analysis = make_analysis(
        constraints=[make_budget_constraint("c1"), make_budget_constraint("c1")],
        clarification_required=True,
        required_inputs=[],
    )
    with pytest.raises(GoalAnalysisError) as excinfo:
        validate_analysis(analysis)
    assert len(excinfo.value.issues) >= 2


def test_blank_constraint_description_is_rejected() -> None:
    constraint = make_budget_constraint()
    constraint.description = "   "
    with pytest.raises(GoalAnalysisError):
        validate_analysis(make_analysis(constraints=[constraint]))
