"""Synthesizer tests: presentation only after a PASS, never overclaims.

The Synthesizer runs no tools, mutates no state, and is refused unless the
stored validation is PASS. Constraint claims are cross-checked against the
validation's satisfied set (LLM-side AND controller-side). Fully offline.
"""

from __future__ import annotations

import pytest

from app.agent.controller import (
    ControllerError,
    IllegalTransitionError,
    WorkflowController,
)
from app.agent.synthesizer import (
    SYNTHESIS_SYSTEM_PROMPT,
    SynthesisError,
    Synthesizer,
)
from app.llm.base import LLMOutputInvalidError
from app.llm.mock import MockProvider
from app.state.events import EventType
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
    SynthesisResult,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
    WorkflowState,
)
from app.state.store import InMemoryStateStore

S = WorkflowState
GOAL = "Find options under a budget of 8000 INR"

ANALYSIS = GoalAnalysis(
    objective="Find options under 8000 INR",
    reason_summary="Budget cap explicit.",
    constraints=[
        Constraint(
            id="c1",
            name="budget",
            type=ConstraintType.BUDGET,
            description="Under 8000 INR",
            rule=ConstraintRule(
                operator=ConstraintOperator.LTE, value=8000.0, unit="INR"
            ),
        )
    ],
)

PLAN = Plan(
    tasks=[
        Task(
            id="t1",
            title="Compute total",
            description="Compute the option total",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="total computed",
        )
    ],
)

PASS_RESULT = ValidationResult(
    status=ValidationStatus.PASS,
    reason_summary="Total 7500 <= 8000.",
    satisfied_constraints=["c1"],
    evidence=[EvidenceRef(task_id="t1", note="total 7500")],
)

FAIL_RESULT = ValidationResult(
    status=ValidationStatus.FAIL,
    reason_summary="Total 9000 exceeds 8000.",
    violated_constraints=["c1"],
    replanning_required=True,
)

SYNTHESIS = SynthesisResult(
    final_answer="All options fit the budget: total 7500 INR.",
    key_results=["Total: 7500 INR", "Headroom: 500 INR"],
    constraints_satisfied=["c1"],
    caveats=["Estimates from calculator output."],
    next_actions=["Book the cheapest option"],
)


def make_workflow(*, observations: list[Observation] | None = None) -> Workflow:
    return Workflow(
        id="wf1",
        goal=GOAL,
        analysis=ANALYSIS,
        plan=PLAN,
        observations=observations
        if observations is not None
        else [
            Observation(
                task_id="t1",
                content_summary="total 7500",
                raw={"result": 7500.0},
                source=ObservationSource.TOOL,
                tool_name="calculator",
            )
        ],
        task_states={"t1": TaskStatus.COMPLETED},
        validation=PASS_RESULT,
    )


async def test_synthesis_returns_the_scripted_result() -> None:
    provider = MockProvider([SYNTHESIS])
    result = await Synthesizer(provider).synthesize(make_workflow(), PASS_RESULT)
    assert result is SYNTHESIS or result == SYNTHESIS
    assert result.final_answer.startswith("All options")
    assert len(provider.calls) == 1


def test_system_prompt_bans_chain_of_thought_and_overclaiming() -> None:
    assert "chain-of-thought" in SYNTHESIS_SYSTEM_PROMPT
    assert "subset of the validation's satisfied" in SYNTHESIS_SYSTEM_PROMPT
    assert "Never claim a constraint" in SYNTHESIS_SYSTEM_PROMPT
    assert "No fabrication" in SYNTHESIS_SYSTEM_PROMPT


async def test_prompt_carries_goal_evidence_and_validation() -> None:
    provider = MockProvider([SYNTHESIS])
    await Synthesizer(provider).synthesize(make_workflow(), PASS_RESULT)
    prompt = provider.calls[0].prompt
    assert GOAL in prompt
    assert "7500" in prompt  # evidence reaches the model
    assert '"status": "PASS"' in prompt
    assert "c1" in prompt
    assert len(prompt) < 50_000


# ---------------------------------------------------------------------------
# Guards: no PASS -> no call, no claims outside validation
# ---------------------------------------------------------------------------


async def test_non_pass_validation_is_refused_before_any_llm_call() -> None:
    provider = MockProvider([SYNTHESIS])
    with pytest.raises(SynthesisError, match="PASS"):
        await Synthesizer(provider).synthesize(make_workflow(), FAIL_RESULT)
    assert provider.calls == []


async def test_missing_analysis_is_refused() -> None:
    workflow = make_workflow()
    workflow.analysis = None
    provider = MockProvider([SYNTHESIS])
    with pytest.raises(SynthesisError, match="no goal analysis"):
        await Synthesizer(provider).synthesize(workflow, PASS_RESULT)
    assert provider.calls == []


async def test_missing_plan_is_refused() -> None:
    workflow = make_workflow()
    workflow.plan = None
    provider = MockProvider([SYNTHESIS])
    with pytest.raises(SynthesisError, match="no plan recorded"):
        await Synthesizer(provider).synthesize(workflow, PASS_RESULT)
    assert provider.calls == []


async def test_claiming_a_constraint_outside_the_pass_is_refused() -> None:
    overclaim = SYNTHESIS.model_copy(
        update={"constraints_satisfied": ["c1", "c2"]}  # c2 never validated
    )
    provider = MockProvider([overclaim])
    with pytest.raises(SynthesisError, match="outside the PASS validation"):
        await Synthesizer(provider).synthesize(make_workflow(), PASS_RESULT)
    assert len(provider.calls) == 1  # the LLM ran; the post-check refused it


async def test_subset_claims_are_accepted() -> None:
    subset = SYNTHESIS.model_copy(update={"constraints_satisfied": []})
    provider = MockProvider([subset])
    result = await Synthesizer(provider).synthesize(make_workflow(), PASS_RESULT)
    assert result.constraints_satisfied == []


async def test_malformed_synthesis_raises_output_invalid() -> None:
    provider = MockProvider([{"final_answer": ""}])  # fails min_length
    with pytest.raises(LLMOutputInvalidError):
        await Synthesizer(provider).synthesize(make_workflow(), PASS_RESULT)


async def test_huge_raw_payloads_are_bounded_out_of_the_prompt() -> None:
    big = {"blob": "x" * 10_000}
    workflow = make_workflow(
        observations=[
            Observation(
                task_id="t1",
                content_summary="total 7500",
                raw=big,
                source=ObservationSource.TOOL,
            )
        ]
    )
    provider = MockProvider([SYNTHESIS])
    await Synthesizer(provider).synthesize(workflow, PASS_RESULT)
    prompt = provider.calls[0].prompt
    assert "raw_omitted_chars" in prompt
    assert "x" * 1500 not in prompt
    assert len(prompt) < 50_000


# ---------------------------------------------------------------------------
# Controller: record_synthesis guards (Python-owned completion gate)
# ---------------------------------------------------------------------------


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


def validating_with_pass(controller: WorkflowController) -> str:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, PLAN)
    controller.begin_execution(workflow.id)
    controller.begin_task(workflow.id, "t1")
    controller.record_observation(
        workflow.id,
        Observation(
            task_id="t1",
            content_summary="total 7500",
            raw={"result": 7500.0},
            source=ObservationSource.TOOL,
            tool_name="calculator",
        ),
    )
    controller.complete_task(workflow.id, "t1")
    controller.finish_execution(workflow.id)
    controller.begin_validation(workflow.id)
    controller.record_validation(workflow.id, PASS_RESULT)
    return workflow.id


def test_record_synthesis_completes_with_a_new_event(controller) -> None:
    workflow_id = validating_with_pass(controller)
    updated = controller.record_synthesis(workflow_id, SYNTHESIS)
    assert updated.state is S.COMPLETED
    assert updated.synthesis is not None
    assert updated.synthesis.final_answer == SYNTHESIS.final_answer
    assert updated.final_answer == SYNTHESIS.final_answer
    types = [e.type for e in controller.events(workflow_id)]
    assert EventType.SYNTHESIS_COMPLETED in types
    assert types[-2:] == [EventType.STATE_CHANGED, EventType.WORKFLOW_COMPLETED]
    synthesis_event = [
        e for e in controller.events(workflow_id)
        if e.type is EventType.SYNTHESIS_COMPLETED
    ][0]
    assert synthesis_event.payload["constraints_satisfied"] == ["c1"]


def test_record_synthesis_rejected_without_pass(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, PLAN)
    controller.begin_execution(workflow.id)
    controller.begin_task(workflow.id, "t1")
    controller.record_observation(
        workflow.id,
        Observation(task_id="t1", content_summary="total 9000"),
    )
    controller.complete_task(workflow.id, "t1")
    controller.finish_execution(workflow.id)
    controller.begin_validation(workflow.id)
    controller.record_validation(workflow.id, FAIL_RESULT)
    with pytest.raises(ControllerError, match="not PASS"):
        controller.record_synthesis(workflow.id, SYNTHESIS)
    assert controller.get_workflow(workflow.id).state is S.VALIDATING
    assert controller.get_workflow(workflow.id).synthesis is None


def test_record_synthesis_rejected_outside_validating(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    with pytest.raises(IllegalTransitionError):
        controller.record_synthesis(workflow.id, SYNTHESIS)


def test_record_synthesis_rejects_overclaimed_constraints(controller) -> None:
    workflow_id = validating_with_pass(controller)
    poisoned = SYNTHESIS.model_copy(
        update={"constraints_satisfied": ["c1", "ghost"]}
    )
    with pytest.raises(ControllerError, match="not in the PASS validation"):
        controller.record_synthesis(workflow_id, poisoned)
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.VALIDATING
    assert workflow.synthesis is None
