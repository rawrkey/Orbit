"""Model contract tests: construction, enum rejection, serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import CreateWorkflowRequest, WorkflowStatusResponse
from app.llm.schemas import ToolDecision
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    FailureContext,
    GoalAnalysis,
    Observation,
    Plan,
    Preference,
    ReplanDecision,
    ReplanStrategy,
    Task,
    TaskStatus,
    ToolCall,
    ToolResult,
    Verdict,
    Workflow,
    WorkflowState,
)

S = WorkflowState


def make_task(**overrides: object) -> Task:
    base: dict = {
        "description": "Search accommodation",
        "success_criteria": "At least 2 options under budget",
    }
    base.update(overrides)
    return Task(**base)  # type: ignore[arg-type]


def make_plan() -> Plan:
    return Plan(tasks=[make_task(id="t1"), make_task(id="t2", depends_on=["t1"])])


def make_analysis() -> GoalAnalysis:
    return GoalAnalysis(
        objective="Plan a 3-day Mumbai trip",
        constraints=[
            Constraint(
                id="c1",
                name="budget",
                type=ConstraintType.BUDGET,
                description="Total under 8000 INR",
                rule=ConstraintRule(operator=ConstraintOperator.LTE, value=8000, unit="INR"),
            )
        ],
        preferences=[Preference(name="food", value="vegetarian")],
        assumptions=["Travel in the off-season"],
        required_inputs=[],
        optional_inputs=["Preferred neighborhood"],
        clarification_required=False,
        reason_summary="Budget and duration are explicit; neighborhood is optional.",
        is_feasible=True,
    )


def make_workflow() -> Workflow:
    return Workflow(
        id="wf-1",
        goal="Plan a 3-day Mumbai trip for 2 people under 8000 INR",
        state=S.VALIDATING,
        analysis=make_analysis(),
        plan=make_plan(),
        replan_count=1,
        task_states={"t1": TaskStatus.PASSED, "t2": TaskStatus.RUNNING},
        observations=[
            Observation(task_id="t1", content_summary="Found 3 stays under 4000")
        ],
        event_count=7,
        final_answer=None,
        failure=None,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructs_every_core_model() -> None:
    assert make_workflow().state == S.VALIDATING
    assert make_analysis().objective
    assert make_analysis().reason_summary
    assert len(make_plan().tasks) == 2
    assert ToolCall(tool_name="calculator", arguments={"a": 2}).tool_name == "calculator"
    assert ToolResult(ok=True, data={"sum": 5}).ok is True
    assert Observation(task_id="t1", content_summary="ok").is_error is False
    assert Verdict(task_id="t1", passed=True, reason_summary="within budget").passed
    assert ReplanDecision(
        strategy=ReplanStrategy.REVISE_PLAN, reason_summary="over budget"
    ).strategy == ReplanStrategy.REVISE_PLAN
    assert FailureContext(
        stage=S.VALIDATING, error_code="BUDGET_EXCEEDED", detail="9500 > 8000"
    ).error_code == "BUDGET_EXCEEDED"
    assert Workflow(id="w", goal="g").state == S.RECEIVED


def test_workflow_defaults() -> None:
    workflow = Workflow(id="w", goal="g")
    assert workflow.replan_count == 0
    assert workflow.event_count == 0
    assert workflow.analysis is None
    assert workflow.plan is None
    assert workflow.task_states == {}
    assert workflow.observations == []


# ---------------------------------------------------------------------------
# Invalid values are rejected
# ---------------------------------------------------------------------------


def test_invalid_enum_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        WorkflowState("NOT_A_STATE")
    with pytest.raises(ValueError):
        TaskStatus("IN_PROGRESS")
    with pytest.raises(ValueError):
        ConstraintType("MONEY_LIMIT")
    with pytest.raises(ValueError):
        ReplanStrategy("IMPROVISE")


def test_invalid_field_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Workflow(id="w", goal="g", state="NOT_A_STATE")
    with pytest.raises(ValidationError):
        Workflow(id="w", goal="g", replan_count=-1)
    with pytest.raises(ValidationError):
        Workflow(id="", goal="g")
    with pytest.raises(ValidationError):
        Constraint(type=ConstraintType.BUDGET, description="")  # empty description
    with pytest.raises(ValidationError):
        Verdict(task_id="t1", passed="maybe", reason_summary="hmm")
    with pytest.raises(ValidationError):
        make_task(description="")  # empty description


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


def test_workflow_serialization_round_trip() -> None:
    original = make_workflow()
    restored = Workflow.model_validate_json(original.model_dump_json())
    assert restored == original


def test_nested_models_round_trip_individually() -> None:
    for model in (
        make_analysis(),
        make_plan(),
        ToolCall(tool_name="web_search", arguments={"query": "hostels", "limit": 3}),
        ToolResult(ok=False, error="timeout", error_code="TOOL_TIMEOUT"),
        Observation(task_id="t1", content_summary="failed", is_error=True),
        Verdict(task_id="t1", passed=False, failed_constraint_ids=["c1"],
                reason_summary="over budget"),
        ReplanDecision(strategy=ReplanStrategy.ADD_TASK, reason_summary="need cheaper option"),
        FailureContext(stage=S.EXECUTING, error_code="TOOL_TIMEOUT", detail="t1 timed out"),
    ):
        dump = model.model_dump_json()
        assert type(model).model_validate_json(dump) == model


def test_task_statuses_serialize_as_strings() -> None:
    task = make_task(status=TaskStatus.PASSED)
    assert task.model_dump()["status"] == "PASSED"
    assert Task.model_validate({"description": "x", "success_criteria": "y",
                                "status": "FAILED"}).status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# LLM decision schemas
# ---------------------------------------------------------------------------


def test_tool_decision_requires_reason_summary() -> None:
    with pytest.raises(ValidationError):
        ToolDecision(tool_name="calculator")


def test_tool_decision_arguments_are_typed() -> None:
    decision = ToolDecision(
        tool_name="calculator",
        arguments={"expression": "2+2", "precision": 2},
        reason_summary="Recompute the estimated total.",
    )
    assert decision.arguments["precision"] == 2
    with pytest.raises(ValidationError):
        ToolDecision(tool_name="calculator", reason_summary="r", arguments={"bad": object()})


# ---------------------------------------------------------------------------
# API request/response contracts
# ---------------------------------------------------------------------------


def test_create_workflow_request_rejects_blank_goal() -> None:
    with pytest.raises(ValidationError):
        CreateWorkflowRequest(goal="")
    with pytest.raises(ValidationError):
        CreateWorkflowRequest(goal="   ")
    assert CreateWorkflowRequest(goal="  plan a trip ").goal == "plan a trip"


def test_workflow_status_response_is_serializable() -> None:
    response = WorkflowStatusResponse(
        workflow_id="wf-1",
        goal="g",
        state=S.RECEIVED,
        replan_count=0,
        created_at=make_workflow().created_at,
        updated_at=make_workflow().updated_at,
        task_count=0,
        observation_count=0,
        event_count=1,
    )
    assert WorkflowStatusResponse.model_validate_json(
        response.model_dump_json()
    ) == response
