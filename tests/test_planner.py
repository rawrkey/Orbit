"""Planner tests: deterministic structural validation + bounded repair."""

from __future__ import annotations

import pytest

from app.agent.planner import (
    DEFAULT_MAX_PLAN_TASKS,
    PlanValidationError,
    Planner,
    validate_plan,
)
from app.llm.base import LLMOutputInvalidError
from app.llm.mock import MockProvider
from app.state.models import (
    Constraint,
    ConstraintType,
    GoalAnalysis,
    Plan,
    Task,
    TaskStatus,
)


def make_task(id: str = "t1", **overrides) -> Task:
    base: dict = {
        "id": id,
        "description": f"step {id}",
        "success_criteria": f"{id} completed",
    }
    base.update(overrides)
    return Task(**base)


def make_plan(*tasks: Task) -> Plan:
    return Plan(tasks=list(tasks))


def make_analysis(**overrides) -> GoalAnalysis:
    base: dict = {
        "objective": "Plan a 3-day Mumbai trip",
        "reason_summary": "Duration is explicit.",
    }
    base.update(overrides)
    return GoalAnalysis(**base)


ANALYSIS = make_analysis(
    constraints=[
        Constraint(
            id="c1",
            type=ConstraintType.BUDGET,
            description="Total under 8000 INR",
        )
    ]
)


# ---------------------------------------------------------------------------
# validate_plan - every rule
# ---------------------------------------------------------------------------


def test_default_max_plan_tasks_is_seven() -> None:
    assert DEFAULT_MAX_PLAN_TASKS == 7


def test_valid_plan_passes() -> None:
    validate_plan(
        make_plan(
            make_task("t1", requires_tool=True, tool_hint="web_search"),
            make_task("t2", depends_on=["t1"]),
        ),
        analysis=ANALYSIS,
    )


def test_empty_plan_is_rejected() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(make_plan())
    assert any("no tasks" in issue for issue in excinfo.value.issues)


def test_task_limit_is_enforced() -> None:
    tasks = [make_task(f"t{i}") for i in range(3)]
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(make_plan(*tasks), max_tasks=2)
    assert any("limit is 2" in issue for issue in excinfo.value.issues)


def test_duplicate_task_ids_are_rejected() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(make_plan(make_task("t1"), make_task("t1")))
    assert any("duplicate task id" in issue for issue in excinfo.value.issues)


def test_unknown_dependency_is_rejected() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(make_plan(make_task("t1", depends_on=["t99"])))
    assert any("unknown task" in issue for issue in excinfo.value.issues)


def test_dependency_must_be_listed_earlier() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(
            make_plan(
                make_task("t1", depends_on=["t2"]),
                make_task("t2"),
            )
        )
    assert any("not listed earlier" in issue for issue in excinfo.value.issues)


def test_self_dependency_is_rejected() -> None:
    with pytest.raises(PlanValidationError):
        validate_plan(make_plan(make_task("t1", depends_on=["t1"])))


def test_dependency_cycle_is_rejected() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(
            make_plan(
                make_task("t1", depends_on=["t2"]),
                make_task("t2", depends_on=["t1"]),
            )
        )
    assert excinfo.value.issues


def test_tasks_must_start_pending() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(make_plan(make_task("t1", status=TaskStatus.RUNNING)))
    assert any("status pending" in issue for issue in excinfo.value.issues)


def test_requires_tool_without_tool_hint_is_rejected() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(make_plan(make_task("t1", requires_tool=True)))
    assert any("no tool_hint" in issue for issue in excinfo.value.issues)


def test_tool_hint_without_requires_tool_is_rejected() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(
            make_plan(make_task("t1", requires_tool=False, tool_hint="file"))
        )
    assert any("requires_tool is false" in issue for issue in excinfo.value.issues)


def test_unknown_constraint_reference_is_rejected() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(
            make_plan(make_task("t1", constraint_ids=["c99"])),
            analysis=ANALYSIS,
        )
    assert any(
        "unknown constraint" in issue for issue in excinfo.value.issues
    )


def test_known_constraint_reference_is_accepted() -> None:
    validate_plan(
        make_plan(make_task("t1", constraint_ids=["c1"])),
        analysis=ANALYSIS,
    )


def test_all_issues_are_reported_together() -> None:
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(
            make_plan(
                make_task("t1"),
                make_task("t1"),
                make_task("t3", depends_on=["ghost"]),
                make_task("t4", requires_tool=True),
            )
        )
    issues = excinfo.value.issues
    assert len(issues) >= 3
    assert any("duplicate task id" in issue for issue in issues)
    assert any("unknown task" in issue for issue in issues)
    assert any("no tool_hint" in issue for issue in issues)


def test_constructor_rejects_bad_limits() -> None:
    provider = MockProvider()
    with pytest.raises(ValueError):
        Planner(provider, max_tasks=0)
    with pytest.raises(ValueError):
        Planner(provider, repair_attempts=-1)


# ---------------------------------------------------------------------------
# Planner - LLM call flow with bounded repair
# ---------------------------------------------------------------------------


async def test_planner_returns_valid_plan_with_one_call() -> None:
    plan = make_plan(
        make_task("t1", requires_tool=True, tool_hint="web_search"),
        make_task("t2", depends_on=["t1"]),
    )
    provider = MockProvider([plan])
    planner = Planner(provider)

    result = await planner.plan("Plan a trip", ANALYSIS)

    assert [t.id for t in result.tasks] == ["t1", "t2"]
    assert len(provider.calls) == 1
    assert provider.calls[0].schema_name == "Plan"
    assert "Plan a trip" in provider.calls[0].prompt


async def test_blank_goal_is_rejected_without_llm_call() -> None:
    provider = MockProvider()
    planner = Planner(provider)

    with pytest.raises(ValueError):
        await planner.plan("   ", ANALYSIS)

    assert provider.calls == []


async def test_invalid_plan_triggers_exactly_one_repair_call() -> None:
    invalid = make_plan(make_task("t1", depends_on=["ghost"]))
    valid = make_plan(make_task("t1"))
    provider = MockProvider([invalid, valid])
    planner = Planner(provider)

    result = await planner.plan("Plan a trip", ANALYSIS)

    assert len(provider.calls) == 2
    repair_prompt = provider.calls[1].prompt
    assert "structural validation" in repair_prompt
    assert "ghost" in repair_prompt


async def test_still_invalid_after_repair_raises_with_all_issues() -> None:
    invalid = make_plan(make_task("t1", depends_on=["ghost"]))
    provider = MockProvider([invalid, invalid])
    planner = Planner(provider)

    with pytest.raises(PlanValidationError) as excinfo:
        await planner.plan("Plan a trip", ANALYSIS)

    assert len(provider.calls) == 2  # initial + one repair, never more
    assert any("ghost" in issue for issue in excinfo.value.issues)


async def test_zero_repair_attempts_fails_after_single_call() -> None:
    invalid = make_plan()
    provider = MockProvider([invalid])
    planner = Planner(provider, repair_attempts=0)

    with pytest.raises(PlanValidationError):
        await planner.plan("Plan a trip", ANALYSIS)

    assert len(provider.calls) == 1


async def test_mismatched_schema_output_propagates_as_invalid() -> None:
    provider = MockProvider([ANALYSIS])  # GoalAnalysis, not a Plan
    planner = Planner(provider)

    with pytest.raises(LLMOutputInvalidError):
        await planner.plan("Plan a trip", ANALYSIS)

    assert len(provider.calls) == 1


async def test_repair_prompt_lists_constraint_ids_for_reference() -> None:
    plan = make_plan(make_task("t1"))
    provider = MockProvider([plan])
    planner = Planner(provider)

    await planner.plan("Plan a trip", ANALYSIS)

    assert "c1" in provider.calls[0].prompt
