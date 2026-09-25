"""Phase 5 hardening tests (Part A): the13 review fixes.

Replan mode selection (FAIL -> corrective, INSUFFICIENT -> evidence
gathering), constraint immutability (prompt rule + analysis fingerprint
guard + unknown-id rejection), plan-revision integrity fields, stale
evidence fencing, atomic rejections with typed errors, dependency-cascade
resets, semantic fingerprints with cross-process golden hashes, event/plan
consistency, and the extended security fragment bans. Fully offline.
"""

from __future__ import annotations

import pytest

from app.agent.controller import (
    ControllerError,
    PlanReplacementError,
    WorkflowController,
)
from app.agent.replanner import (
    MODE_INSTRUCTIONS,
    REPLANNER_SYSTEM_PROMPT,
    ReplanValidationError,
    Replanner,
    ReplannerError,
    replan_mode,
)
from app.agent.state_machine import IllegalTransitionError
from app.agent.validator import Validator
from app.llm.base import T
from app.llm.mock import MockProvider
from app.llm.schemas import ValidationProposal
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
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
    WorkflowState,
)
from app.state.store import InMemoryStateStore

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
        PLAN.tasks[1],  # unchanged definition
    ],
)

PLAN_DEP = Plan(
    version=1,
    tasks=[
        Task(id="t1", title="Prices", description="Collect stay prices",
             constraint_ids=["c1"], success_criteria="prices collected"),
        Task(id="t2", title="Total", description="Sum the prices",
             depends_on=["t1"], requires_tool=True, tool_hint="calculator",
             constraint_ids=["c1"], success_criteria="total computed"),
        Task(id="t3", title="Itinerary", description="Draft the itinerary",
             depends_on=["t2"], success_criteria="itinerary drafted"),
        Task(id="t4", title="Notes", description="Add general city notes",
             success_criteria="notes written"),
    ],
)

FAIL_RESULT = ValidationResult(
    status=ValidationStatus.FAIL,
    reason_summary="Trip total 9000 INR exceeds the 8000 INR budget.",
    violated_constraints=["c1"],
    replanning_required=True,
)

INSUFFICIENT_RESULT = ValidationResult(
    status=ValidationStatus.INSUFFICIENT,
    reason_summary="Price evidence covers only 3 of 10 bookings.",
    unresolved_constraints=["c1"],
    replanning_required=True,
)


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=2)


def ready(controller: WorkflowController, *, plan: Plan = PLAN) -> str:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, plan)
    return workflow.id


def executing_workflow(
    controller: WorkflowController, *, plan: Plan = PLAN
) -> str:
    workflow_id = ready(controller, plan=plan)
    controller.begin_execution(workflow_id)
    for task in plan.tasks:
        controller.begin_task(workflow_id, task.id)
        controller.record_observation(
            workflow_id,
            Observation(
                task_id=task.id,
                content_summary="Done.",
                source=ObservationSource.REASONING,
            ),
        )
        controller.complete_task(workflow_id, task.id)
    controller.finish_execution(workflow_id)
    return workflow_id


def replanning(controller: WorkflowController) -> str:
    workflow_id = executing_workflow(controller)
    controller.begin_validation(workflow_id)
    controller.record_validation(workflow_id, FAIL_RESULT)
    controller.begin_replanning(workflow_id)
    return workflow_id


def make_workflow(*, plan: Plan = PLAN) -> Workflow:
    return Workflow(
        id="wf1",
        goal=GOAL,
        analysis=ANALYSIS.model_copy(deep=True),
        plan=plan,
        task_states={task.id: TaskStatus.COMPLETED for task in plan.tasks},
        replan_count=1,
    )


def claim(*, evidence_task: str = "t1", goal: str = "satisfied"):
    return ValidationProposal(
        reason_summary="Budget verified from calculator output.",
        constraint_verdicts=[
            {
                "constraint_id": "c1",
                "outcome": "satisfied",
                "evidence": [{"task_id": evidence_task, "note": "total"}],
                "evidence_field": "result",
            }
        ],
        goal_outcome=goal,
    )


def one_event(controller, workflow_id, event_type):
    matches = [e for e in controller.events(workflow_id) if e.type is event_type]
    assert len(matches) == 1, f"expected exactly one {event_type}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# A1: replan mode selection (corrective vs evidence_gathering)
# ---------------------------------------------------------------------------


def test_replan_mode_classifies_validation_status() -> None:
    assert replan_mode(FAIL_RESULT) == "corrective"
    assert replan_mode(INSUFFICIENT_RESULT) == "evidence_gathering"
    assert replan_mode(
        ValidationResult(
            status=ValidationStatus.PASS,
            reason_summary="ok",
            satisfied_constraints=["c1"],
        )
    ) == "evidence_gathering"  # only FAIL proves a violation


async def test_fail_validation_prompt_uses_corrective_mode() -> None:
    provider = MockProvider([PLAN_V2])
    await Replanner(provider).propose(make_workflow(), FAIL_RESULT)
    prompt = provider.calls[0].prompt
    assert "MODE: corrective" in prompt
    assert '"replan_mode": "corrective"' in prompt
    assert "MODE: evidence_gathering" not in prompt
    assert MODE_INSTRUCTIONS["corrective"] in prompt


async def test_insufficient_validation_prompt_uses_evidence_gathering_mode() -> None:
    provider = MockProvider([PLAN_V2])
    await Replanner(provider).propose(make_workflow(), INSUFFICIENT_RESULT)
    prompt = provider.calls[0].prompt
    assert "MODE: evidence_gathering" in prompt
    assert '"replan_mode": "evidence_gathering"' in prompt
    assert "MODE: corrective" not in prompt
    assert "MISSING, not that a constraint was violated" in prompt


# ---------------------------------------------------------------------------
# A2: constraint immutability
# ---------------------------------------------------------------------------


def test_system_prompt_fixes_constraints_and_bans_invented_ids() -> None:
    assert "Constraints are FIXED by the user" in REPLANNER_SYSTEM_PROMPT
    assert "Never relax, weaken, or modify" in REPLANNER_SYSTEM_PROMPT
    assert "Constraint ids not present in the analysis will be" in (
        REPLANNER_SYSTEM_PROMPT
    )
    assert "rejected - only reference ids from the analysis" in (
        REPLANNER_SYSTEM_PROMPT
    )


async def test_unknown_constraint_id_is_rejected() -> None:
    poisoned = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Compute cheaper total",
                description="Recompute the total with cheaper stays",
                requires_tool=True,
                tool_hint="calculator",
                constraint_ids=["ghost"],  # not in the analysis
                success_criteria="revised total computed",
            ),
            PLAN.tasks[1],
        ]
    )
    provider = MockProvider([poisoned, poisoned])
    with pytest.raises(ReplanValidationError, match="unknown constraint"):
        await Replanner(provider).propose(make_workflow(), FAIL_RESULT)
    assert len(provider.calls) == 2  # one bounded repair attempt


async def test_analysis_survives_proposal_unchanged() -> None:
    workflow = make_workflow()
    before = workflow.analysis.fingerprint()  # type: ignore[union-attr]
    await Replanner(MockProvider([PLAN_V2])).propose(workflow, FAIL_RESULT)
    assert workflow.analysis is not None
    assert workflow.analysis.fingerprint() == before


async def test_analysis_tampering_during_llm_call_is_rejected() -> None:
    """Relaxing a limit while the replanner runs must abort with no plan."""

    class TamperingProvider:
        model_id = "mock/tamper"

        def __init__(self, inner: MockProvider, workflow: Workflow) -> None:
            self._inner = inner
            self._workflow = workflow

        async def complete(
            self,
            prompt: str,
            schema: type[T],
            *,
            system: str | None = None,
            temperature: float = 0.2,
            timeout_s: float = 30.0,
        ) -> T:
            assert self._workflow.analysis is not None
            rule = self._workflow.analysis.constraints[0].rule
            assert rule is not None
            rule.value = 12000.0  # attempt to relax the user's budget
            return await self._inner.complete(
                prompt, schema, system=system,
                temperature=temperature, timeout_s=timeout_s,
            )

    workflow = make_workflow()
    provider = TamperingProvider(MockProvider([PLAN_V2]), workflow)
    with pytest.raises(ReplannerError, match="immutability"):
        await Replanner(provider).propose(workflow, FAIL_RESULT)  # type: ignore[arg-type]


def test_replace_plan_preserves_analysis(controller) -> None:
    workflow_id = replanning(controller)
    before = controller.get_workflow(workflow_id).analysis.fingerprint()  # type: ignore[union-attr]
    controller.replace_plan(workflow_id, PLAN_V2, reason="Cheaper stays.")
    after = controller.get_workflow(workflow_id)
    assert after.analysis is not None
    assert after.analysis.fingerprint() == before  # constraints untouched


# ---------------------------------------------------------------------------
# A3: plan revision integrity fields
# ---------------------------------------------------------------------------


def test_plan_revision_records_validation_and_new_version(controller) -> None:
    workflow_id = replanning(controller)
    controller.replace_plan(workflow_id, PLAN_V2, reason="Cheaper stays.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.plan is not None
    revision = workflow.plan_history[0]
    assert revision.plan.version == 1
    assert revision.new_plan_version == 2 == workflow.plan.version
    assert revision.validation is not None
    assert revision.validation.status is ValidationStatus.FAIL
    assert revision.validation.violated_constraints == ["c1"]
    assert revision.validation.reason_summary == FAIL_RESULT.reason_summary


# ---------------------------------------------------------------------------
# A4: observation task fingerprints + stale evidence fencing
# ---------------------------------------------------------------------------


def test_record_observation_stamps_task_fingerprint(controller) -> None:
    workflow_id = ready(controller)
    workflow = controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t1")
    controller.record_observation(
        workflow_id,
        Observation(
            task_id="t1",
            content_summary="total 7500",
            raw={"result": 7500.0},
            source=ObservationSource.TOOL,
            tool_name="calculator",
        ),
    )
    workflow = controller.get_workflow(workflow_id)
    assert workflow.plan is not None
    expected = workflow.plan.tasks[0].definition_fingerprint()
    recorded = workflow.observations[-1]
    assert recorded.task_fingerprint == expected
    assert recorded.task_fingerprint is not None


async def test_fresh_stamped_evidence_satisfies_the_current_task() -> None:
    plan = Plan(tasks=[PLAN.tasks[0]])
    observation = Observation(
        task_id="t1",
        content_summary="total 7500",
        raw={"result": 7500.0},
        source=ObservationSource.TOOL,
        tool_name="calculator",
        task_fingerprint=plan.tasks[0].definition_fingerprint(),
    )
    workflow = Workflow(
        id="wf1", goal=GOAL, analysis=ANALYSIS, plan=plan,
        observations=[observation],
        task_states={"t1": TaskStatus.COMPLETED},
    )
    result = await Validator(MockProvider([claim()])).validate(workflow)
    assert result.status is ValidationStatus.PASS
    assert result.satisfied_constraints == ["c1"]


async def test_stale_evidence_cannot_satisfy_a_redefined_task() -> None:
    plan = Plan(tasks=[PLAN.tasks[0]])  # current t1 definition
    old_definition = Task(
        id="t1",
        title="Compute total",
        description="OLD wording that was replaced by the replan",
        requires_tool=True,
        tool_hint="calculator",
        constraint_ids=["c1"],
        success_criteria="total computed",
    )
    stale = Observation(
        task_id="t1",
        content_summary="total 7500 from the OLD definition",
        raw={"result": 7500.0},
        source=ObservationSource.TOOL,
        tool_name="calculator",
        task_fingerprint=old_definition.definition_fingerprint(),
    )
    workflow = Workflow(
        id="wf1", goal=GOAL, analysis=ANALYSIS, plan=plan,
        observations=[stale],
        task_states={"t1": TaskStatus.COMPLETED},
    )
    result = await Validator(MockProvider([claim(goal="insufficient")])).validate(
        workflow
    )
    # The 7500 would satisfy the rule if accepted - it must be ignored.
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.unresolved_constraints == ["c1"]
    assert result.evidence == []


async def test_unstamped_legacy_observations_remain_usable() -> None:
    plan = Plan(tasks=[PLAN.tasks[0]])
    observation = Observation(
        task_id="t1",
        content_summary="total 7500",
        raw={"result": 7500.0},
        source=ObservationSource.TOOL,
        tool_name="calculator",
        task_fingerprint=None,  # legacy/fixture data predating the stamp
    )
    workflow = Workflow(
        id="wf1", goal=GOAL, analysis=ANALYSIS, plan=plan,
        observations=[observation],
        task_states={"t1": TaskStatus.COMPLETED},
    )
    result = await Validator(MockProvider([claim()])).validate(workflow)
    assert result.status is ValidationStatus.PASS


# ---------------------------------------------------------------------------
# A5: typed rejection errors with an append-only diagnosis event
# ---------------------------------------------------------------------------


def test_plan_replacement_error_is_a_controller_error(controller) -> None:
    assert issubclass(PlanReplacementError, ControllerError)
    workflow_id = replanning(controller)
    events_before = len(controller.events(workflow_id))
    with pytest.raises(PlanReplacementError, match="identical"):
        controller.replace_plan(workflow_id, PLAN, reason="No real change.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.REPLANNING
    assert workflow.plan is not None and workflow.plan.version == 1
    assert workflow.plan_history == []
    assert workflow.task_states == {
        "t1": TaskStatus.COMPLETED,
        "t2": TaskStatus.COMPLETED,
    }  # states untouched by the rejection
    events = controller.events(workflow_id)
    assert len(events) == events_before + 1
    assert events[-1].type is E.REPLAN_REJECTED
    assert events[-1].payload["reason"] == "identical_plan"


def test_replace_without_existing_plan_is_rejected(controller) -> None:
    workflow = controller.create_workflow(GOAL)
    workflow.state = S.REPLANNING  # crafted invariant breach (no plan)
    controller._store.save_workflow(workflow)
    with pytest.raises(PlanReplacementError, match="does not exist"):
        controller.replace_plan(workflow.id, PLAN_V2, reason="Whatever.")
    updated = controller.get_workflow(workflow.id)
    assert updated.plan is None
    event = one_event(controller, workflow.id, E.REPLAN_REJECTED)
    assert event.payload["reason"] == "no_current_plan"


def test_wrong_state_rejection_keeps_state_and_history(controller) -> None:
    workflow_id = executing_workflow(controller)
    controller.begin_validation(workflow_id)
    controller.record_validation(workflow_id, FAIL_RESULT)  # VALIDATING
    events_before = len(controller.events(workflow_id))
    with pytest.raises(IllegalTransitionError):
        controller.replace_plan(workflow_id, PLAN_V2, reason="Too early.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.state is S.VALIDATING
    assert workflow.plan_history == []
    events = controller.events(workflow_id)
    assert len(events) == events_before + 1
    assert events[-1].type is E.REPLAN_REJECTED
    assert events[-1].payload["reason"] == "invalid_state"


# ---------------------------------------------------------------------------
# A7: dependency-cascade resets on replan
# ---------------------------------------------------------------------------


def test_dependency_cascade_resets_dependents_of_a_changed_task(controller) -> None:
    workflow_id = ready(controller, plan=PLAN_DEP)
    controller.begin_execution(workflow_id)
    for task in PLAN_DEP.tasks:
        controller.begin_task(workflow_id, task.id)
        controller.record_observation(
            workflow_id,
            Observation(
                task_id=task.id, content_summary="Done.",
                source=ObservationSource.REASONING,
            ),
        )
        controller.complete_task(workflow_id, task.id)
    controller.finish_execution(workflow_id)
    controller.begin_validation(workflow_id)
    controller.record_validation(workflow_id, FAIL_RESULT)
    controller.begin_replanning(workflow_id)

    # Only t1's definition changes; t2/t3/t4 are byte-identical.
    revised = Plan(
        version=1,
        tasks=[
            Task(id="t1", title="Prices", description="Collect stay prices ONLINE",
                 constraint_ids=["c1"], success_criteria="prices collected"),
            PLAN_DEP.tasks[1],
            PLAN_DEP.tasks[2],
            PLAN_DEP.tasks[3],
        ],
    )
    controller.replace_plan(workflow_id, revised, reason="Better prices.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.task_states["t1"] is TaskStatus.PENDING  # changed
    assert workflow.task_states["t2"] is TaskStatus.PENDING  # depends on t1
    assert workflow.task_states["t3"] is TaskStatus.PENDING  # cascade (t1->t2->t3)
    assert workflow.task_states["t4"] is TaskStatus.COMPLETED  # unrelated: preserved


def test_unchanged_chain_is_fully_preserved(controller) -> None:
    workflow_id = ready(controller, plan=PLAN_DEP)
    controller.begin_execution(workflow_id)
    for task in PLAN_DEP.tasks:
        controller.begin_task(workflow_id, task.id)
        controller.record_observation(
            workflow_id,
            Observation(
                task_id=task.id, content_summary="Done.",
                source=ObservationSource.REASONING,
            ),
        )
        controller.complete_task(workflow_id, task.id)
    controller.finish_execution(workflow_id)
    controller.begin_validation(workflow_id)
    controller.record_validation(workflow_id, FAIL_RESULT)
    controller.begin_replanning(workflow_id)

    # Only the unrelated standalone task changes; the t1->t2->t3 chain is intact.
    revised = Plan(
        version=1,
        tasks=[
            PLAN_DEP.tasks[0],
            PLAN_DEP.tasks[1],
            PLAN_DEP.tasks[2],
            Task(id="t4", title="Notes", description="Add LOCAL city notes",
                 success_criteria="notes written"),
        ],
    )
    controller.replace_plan(workflow_id, revised, reason="Local notes.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.task_states["t1"] is TaskStatus.COMPLETED
    assert workflow.task_states["t2"] is TaskStatus.COMPLETED
    assert workflow.task_states["t3"] is TaskStatus.COMPLETED
    assert workflow.task_states["t4"] is TaskStatus.PENDING


# ---------------------------------------------------------------------------
# A9: semantic fingerprints (ids excluded, edges structural)
# ---------------------------------------------------------------------------

GOLDEN_TASK_FP = "1130bb23a1b65156"
GOLDEN_PLAN_FP = "995093dc42e770df"


def golden_tasks() -> tuple[Task, Task]:
    t1 = Task(
        id="t1",
        title="Compute total",
        description="Compute the trip total from published prices",
        tool_hint="calculator",
        requires_tool=True,
        expected_output="total in INR",
        constraint_ids=["c2", "c1"],
        success_criteria="total computed",
    )
    t2 = Task(
        id="t2",
        title="Draft itinerary",
        description="Draft the day-by-day itinerary",
        success_criteria="itinerary drafted",
        depends_on=["t1"],
        constraint_ids=["c1"],
    )
    return t1, t2


def test_plan_fingerprint_excludes_raw_task_ids() -> None:
    t1, t2 = golden_tasks()
    original = Plan(tasks=[t1, t2], version=1)
    renamed1 = t1.model_copy(update={"id": "x9", "constraint_ids": ["c1", "c2"]})
    renamed2 = t2.model_copy(update={"id": "y8", "depends_on": ["x9"]})
    renamed = Plan(tasks=[renamed1, renamed2], version=7)
    # Same logical plan: ids renamed, list order shuffled, version bumped.
    assert original.fingerprint() == renamed.fingerprint()
    # Renaming must not force a full re-run (state merge uses task-level
    # fingerprints, which still include the id).


def test_plan_fingerprint_ignores_constraint_and_dependency_list_order() -> None:
    t1, _ = golden_tasks()
    reordered = t1.model_copy(update={"constraint_ids": ["c1", "c2"]})
    assert t1.definition_fingerprint() == reordered.definition_fingerprint()
    dep_a = Task(id="a", description="A", success_criteria="a ok",
                 depends_on=["x", "y"])
    dep_b = dep_a.model_copy(update={"depends_on": ["y", "x"]})
    assert dep_a.definition_fingerprint() == dep_b.definition_fingerprint()


def test_plan_fingerprint_changes_with_any_semantic_edit() -> None:
    t1, t2 = golden_tasks()
    base = Plan(tasks=[t1, t2], version=1).fingerprint()
    mutations = [
        {"description": "Different wording"},
        {"title": "Different title"},
        {"success_criteria": "Different criteria"},
        {"tool_hint": "web_search"},
        {"requires_tool": False},
        {"input_hint": "prices.csv"},
        {"expected_output": "total in USD"},
    ]
    for update in mutations:
        changed = Plan(
            tasks=[t1.model_copy(update=update), t2], version=1
        ).fingerprint()
        assert changed != base, f"change {update} must alter the fingerprint"


# ---------------------------------------------------------------------------
# A11: event payload <-> stored state consistency
# ---------------------------------------------------------------------------


def test_replan_accepted_version_matches_stored_plan(controller) -> None:
    workflow_id = replanning(controller)
    controller.replace_plan(workflow_id, PLAN_V2, reason="Cheaper stays.")
    workflow = controller.get_workflow(workflow_id)
    assert workflow.plan is not None
    accepted = one_event(controller, workflow_id, E.REPLAN_ACCEPTED)
    proposed = one_event(controller, workflow_id, E.REPLAN_PROPOSED)
    assert accepted.payload["version"] == workflow.plan.version == 2
    assert accepted.payload["previous_version"] == 1
    assert proposed.payload["version"] == workflow.plan.version


def test_validation_event_payload_matches_stored_result(controller) -> None:
    workflow_id = executing_workflow(controller)
    controller.begin_validation(workflow_id)
    controller.record_validation(workflow_id, FAIL_RESULT)
    workflow = controller.get_workflow(workflow_id)
    assert workflow.validation is not None
    event = one_event(controller, workflow_id, E.VALIDATION_FAILED)
    assert event.payload["status"] == workflow.validation.status.value
    assert (
        event.payload["violated_constraints"]
        == workflow.validation.violated_constraints
    )
    assert (
        event.payload["reason_summary"] == workflow.validation.reason_summary
    )


# ---------------------------------------------------------------------------
# A12: cross-process deterministic fingerprint goldens
# ---------------------------------------------------------------------------


def test_golden_fingerprints_are_pinned_from_a_fresh_process() -> None:
    """Values computed by a SEPARATE python process (hash-seed proof)."""
    t1, t2 = golden_tasks()
    assert t1.definition_fingerprint() == GOLDEN_TASK_FP
    assert Plan(tasks=[t1, t2], version=1).fingerprint() == GOLDEN_PLAN_FP
    for value in (GOLDEN_TASK_FP, GOLDEN_PLAN_FP):
        assert len(value) == 16
        assert all(c in "0123456789abcdef" for c in value)


def test_fingerprints_do_not_depend_on_time_or_reconstruction() -> None:
    t1, t2 = golden_tasks()
    first = Plan(tasks=[t1, t2], version=1).fingerprint()
    # Reconstructing identical content (defaults re-evaluated) is stable:
    # no wall-clock, no random id, no dict-order leakage.
    again = Plan(tasks=[t1.model_copy(deep=True), t2.model_copy(deep=True)]).fingerprint()
    assert first == again
    analysis_a = ANALYSIS.model_copy(deep=True)
    analysis_b = ANALYSIS.model_copy(deep=True)
    assert analysis_a.fingerprint() == analysis_b.fingerprint()
    assert analysis_a.fingerprint() != first  # distinct documents differ
