"""Workflow Controller - the only component allowed to change workflow state.

Responsibilities:
- create workflow state
- perform state transitions through the explicit transition table
- maintain and enforce the replan counter (MAX_REPLANS)
- emit append-only events
- reject illegal transitions with ``IllegalTransitionError``
- record validation results and replace plans after a failed validation
  (Phase 5): validate -> replan -> re-execute is owned entirely here

Not yet implemented (later phases): LLM calls, synthesis, and the top-level
agent loop (an orchestrator drives LLM stages through this controller).
"""

from __future__ import annotations

from uuid import uuid4

from app.agent.state_machine import IllegalTransitionError, assert_legal, is_legal
from app.state.events import Event, EventType
from app.state.models import (
    FailureContext,
    GoalAnalysis,
    JsonValue,
    Observation,
    Plan,
    PlanRevision,
    SynthesisResult,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
    WorkflowState,
    utcnow,
)
from app.state.store import StateStore, WorkflowNotFoundError

__all__ = [
    "ControllerError",
    "MaxReplansExceededError",
    "PlanReplacementError",
    "WorkflowController",
    "IllegalTransitionError",
    "WorkflowNotFoundError",
]

DEFAULT_MAX_REPLANS = 3


class ControllerError(Exception):
    """Base class for controller-level failures."""


class PlanReplacementError(ControllerError):
    """A replan proposal was rejected; workflow state is untouched."""


class MaxReplansExceededError(ControllerError):
    """Raised when a workflow has used all of its allowed replanning attempts."""

    def __init__(self, workflow_id: str, replan_count: int, limit: int) -> None:
        self.workflow_id = workflow_id
        self.replan_count = replan_count
        self.limit = limit
        super().__init__(
            f"workflow {workflow_id}: replan limit exceeded "
            f"({replan_count} of {limit} attempts used)"
        )


class WorkflowController:
    """Owns workflow state, transitions, replan limits, and event emission.

    The controller is deterministic: it performs no LLM reasoning and
    executes no tools. Other components propose actions; the controller
    decides whether the application state may change.
    """

    def __init__(self, store: StateStore, *, max_replans: int = DEFAULT_MAX_REPLANS) -> None:
        if max_replans < 0:
            raise ValueError(f"max_replans must be >= 0, got {max_replans}")
        self._store = store
        self._max_replans = max_replans

    @property
    def max_replans(self) -> int:
        return self._max_replans

    # ------------------------------------------------------------------
    # Workflow creation
    # ------------------------------------------------------------------

    def create_workflow(self, goal: str) -> Workflow:
        """Create a workflow in RECEIVED state and emit ``workflow.started``."""
        cleaned = goal.strip()
        if not cleaned:
            raise ValueError("goal must be a non-empty string")
        workflow = Workflow(
            id=uuid4().hex,
            goal=cleaned,
            state=WorkflowState.RECEIVED,
        )
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.WORKFLOW_STARTED,
            decision_summary="Goal received.",
            payload={"goal": cleaned},
        )
        return self.get_workflow(workflow.id)

    # ------------------------------------------------------------------
    # Transitions (state validation happens BEFORE any mutation)
    # ------------------------------------------------------------------

    def transition(
        self,
        workflow_id: str,
        target: WorkflowState,
        *,
        decision_summary: str | None = None,
    ) -> Workflow:
        """Move a workflow to ``target`` if the transition table allows it.

        On an illegal transition: raises ``IllegalTransitionError`` and
        leaves workflow state and the event log untouched.
        """
        workflow = self._store.get_workflow(workflow_id)
        assert_legal(workflow.state, target)  # raises before any mutation

        previous = workflow.state
        workflow.state = target
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.STATE_CHANGED,
            decision_summary=decision_summary,
            payload={"from": previous.value, "to": target.value},
        )
        return self.get_workflow(workflow_id)

    # ------------------------------------------------------------------
    # Lifecycle outcomes
    # ------------------------------------------------------------------

    def complete(self, workflow_id: str, final_answer: str) -> Workflow:
        """Transition to COMPLETED with a final answer, emit workflow.completed."""
        answer = final_answer.strip()
        if not answer:
            raise ValueError("final_answer must be a non-empty string")
        workflow = self._store.get_workflow(workflow_id)
        assert_legal(workflow.state, WorkflowState.COMPLETED)  # before mutation

        workflow.final_answer = answer
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        workflow = self.transition(
            workflow_id,
            WorkflowState.COMPLETED,
            decision_summary="All tasks validated; final answer produced.",
        )
        self._emit(
            workflow,
            EventType.WORKFLOW_COMPLETED,
            decision_summary="Workflow completed.",
        )
        return self.get_workflow(workflow_id)

    def fail(self, workflow_id: str, failure: FailureContext) -> Workflow:
        """Transition to FAILED with a structured failure record."""
        workflow = self._store.get_workflow(workflow_id)
        assert_legal(workflow.state, WorkflowState.FAILED)  # before mutation

        workflow.failure = failure
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        workflow = self.transition(
            workflow_id,
            WorkflowState.FAILED,
            decision_summary=f"Workflow failed: {failure.error_code}",
        )
        self._emit(
            workflow,
            EventType.WORKFLOW_FAILED,
            decision_summary=f"Failed during {failure.stage.value}: {failure.error_code}",
            task_id=failure.task_id,
            payload={
                "error_code": failure.error_code,
                "detail": failure.detail,
            },
        )
        return self.get_workflow(workflow_id)

    # ------------------------------------------------------------------
    # Analysis and planning (Phase 3)
    # ------------------------------------------------------------------

    def begin_analysis(self, workflow_id: str) -> Workflow:
        """RECEIVED -> ANALYZING. Marks that a GoalAnalyzer call is starting."""
        return self.transition(
            workflow_id,
            WorkflowState.ANALYZING,
            decision_summary="Goal analysis started.",
        )

    def record_analysis(
        self, workflow_id: str, analysis: GoalAnalysis
    ) -> Workflow:
        """Store a validated analysis and move ANALYZING -> PLANNING.

        The caller (an orchestrator) passes the object returned by the
        GoalAnalyzer; the controller only stores it and owns the transition.
        Validates the transition BEFORE mutating anything.
        """
        workflow = self._store.get_workflow(workflow_id)
        assert_legal(workflow.state, WorkflowState.PLANNING)  # before mutation

        workflow.analysis = analysis
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.GOAL_ANALYZED,
            decision_summary=analysis.reason_summary,
            payload={
                "objective": analysis.objective,
                "is_feasible": analysis.is_feasible,
                "constraints": len(analysis.constraints),
                "required_inputs": len(analysis.required_inputs),
                "clarification_required": analysis.clarification_required,
            },
        )
        return self.transition(
            workflow_id,
            WorkflowState.PLANNING,
            decision_summary="Analysis recorded; planning started.",
        )

    def require_clarification(
        self, workflow_id: str, analysis: GoalAnalysis, questions: list[str]
    ) -> Workflow:
        """ANALYZING -> AWAITING_CLARIFICATION: the goal needs user answers.

        Stores the analysis (planning resumes from it after the answers) and
        the required question list, then emits goal.analyzed +
        clarification.requested. Only legal while ANALYZING.
        """
        workflow = self._store.get_workflow(workflow_id)
        assert_legal(
            workflow.state, WorkflowState.AWAITING_CLARIFICATION
        )  # before mutation
        cleaned = [question.strip() for question in questions if question.strip()]
        if not cleaned:
            raise ControllerError(
                f"workflow {workflow_id}: clarification requires at least "
                f"one non-empty question"
            )  # before mutation
        if len(cleaned) > 20:
            raise ControllerError(
                f"workflow {workflow_id}: at most 20 clarification questions"
            )  # before mutation

        workflow.analysis = analysis
        workflow.clarification_questions = cleaned
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.GOAL_ANALYZED,
            decision_summary=analysis.reason_summary,
            payload={
                "objective": analysis.objective,
                "is_feasible": analysis.is_feasible,
                "constraints": len(analysis.constraints),
                "required_inputs": len(analysis.required_inputs),
                "clarification_required": analysis.clarification_required,
                "awaiting_clarification": True,
            },
        )
        updated = self.transition(
            workflow_id,
            WorkflowState.AWAITING_CLARIFICATION,
            decision_summary="User clarification required before planning.",
        )
        self._emit(
            updated,
            EventType.CLARIFICATION_REQUESTED,
            decision_summary=f"{len(cleaned)} clarification question(s).",
            payload={
                "questions": cleaned,
                "count": len(cleaned),
            },
        )
        return self.get_workflow(workflow_id)

    def record_clarification(
        self, workflow_id: str, answers: dict[str, str]
    ) -> Workflow:
        """AWAITING_CLARIFICATION -> PLANNING with every answer recorded.

        Answers are keyed by the exact question text; a missing or blank
        answer for any required question rejects the call before mutation.
        """
        workflow = self._store.get_workflow(workflow_id)
        assert_legal(
            workflow.state, WorkflowState.PLANNING
        )  # before mutation
        if not workflow.clarification_questions:
            raise ControllerError(
                f"workflow {workflow_id}: workflow is not awaiting "
                f"clarification"
            )  # before mutation
        missing = [
            question
            for question in workflow.clarification_questions
            if not str(answers.get(question, "")).strip()
        ]
        if missing:
            raise ControllerError(
                f"workflow {workflow_id}: missing answers for: "
                f"{'; '.join(missing)[:300]}"
            )  # before mutation

        workflow.clarifications = {
            question: str(answers[question]).strip()[:500]
            for question in workflow.clarification_questions
        }
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.CLARIFICATION_ANSWERED,
            decision_summary="All clarification questions answered.",
            payload={
                "answered": len(workflow.clarifications),
                "questions": workflow.clarification_questions,
            },
        )
        return self.transition(
            workflow_id,
            WorkflowState.PLANNING,
            decision_summary="Clarifications recorded; planning resumed.",
        )

    def record_plan(self, workflow_id: str, plan: Plan) -> Workflow:
        """Store a validated plan and move PLANNING -> READY.

        Initializes per-task state (all PENDING) for every task in the plan.
        Validates the transition BEFORE mutating anything.
        """
        workflow = self._store.get_workflow(workflow_id)
        assert_legal(workflow.state, WorkflowState.READY)  # before mutation

        workflow.plan = plan
        for task in plan.tasks:
            workflow.task_states.setdefault(task.id, TaskStatus.PENDING)
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.WORKFLOW_PLANNED,
            decision_summary=f"Plan recorded with {len(plan.tasks)} task(s).",
            payload={
                "tasks": len(plan.tasks),
                "version": plan.version,
                "task_ids": [task.id for task in plan.tasks],
            },
        )
        return self.transition(
            workflow_id,
            WorkflowState.READY,
            decision_summary="Plan validated; ready to execute.",
        )

    def record_plan_validation_failure(
        self, workflow_id: str, issues: list[str]
    ) -> Workflow:
        """Emit ``plan.validation_failed`` without changing workflow state.

        Only permitted while the workflow is PLANNING - a structural plan
        failure is a planning-stage diagnosis, not a global state change.
        Raises ``IllegalTransitionError`` (state untouched) otherwise.
        """
        if not issues:
            raise ValueError("issues must be a non-empty list")
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.PLANNING:
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.PLANNING,
                reason="plan validation failure may only be recorded while planning",
            )
        self._emit(
            workflow,
            EventType.PLAN_VALIDATION_FAILED,
            decision_summary=issues[0],
            payload={
                "issues": issues[:20],
                "issue_count": len(issues),
            },
        )
        return self.get_workflow(workflow_id)

    # ------------------------------------------------------------------
    # Execution (Phase 4)
    # ------------------------------------------------------------------

    def begin_execution(self, workflow_id: str) -> Workflow:
        """READY -> EXECUTING. Requires a stored plan; emits execution.started."""
        workflow = self._store.get_workflow(workflow_id)
        if workflow.plan is None or not workflow.plan.tasks:
            raise ControllerError(
                f"workflow {workflow_id}: cannot begin execution without a plan"
            )  # raised before any mutation
        updated = self.transition(
            workflow_id,
            WorkflowState.EXECUTING,
            decision_summary="Execution started.",
        )
        self._emit(
            updated,
            EventType.EXECUTION_STARTED,
            decision_summary=f"Executing a plan of {len(updated.plan.tasks)} task(s).",
            payload={
                "tasks": len(updated.plan.tasks),
                "plan_version": updated.plan.version,
            },
        )
        return self.get_workflow(workflow_id)

    def recover_interrupted_tasks(self, workflow_id: str) -> Workflow:
        """Repair a crashed pass: RUNNING tasks -> PENDING, state unchanged.

        A pass that died mid-task leaves tasks stuck RUNNING. Recovery is
        only legal while EXECUTING (the pass it belongs to is still open);
        it performs no state transition and emits no events - just resets
        the interrupted task states so the executor can re-run them.
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.EXECUTING:
            raise ControllerError(
                f"workflow {workflow_id}: cannot recover tasks while "
                f"{workflow.state.value} (expects EXECUTING)"
            )  # before mutation
        recovered = [
            task_id
            for task_id, status in workflow.task_states.items()
            if status is TaskStatus.RUNNING
        ]
        if not recovered:
            return workflow  # nothing to repair
        for task_id in recovered:
            workflow.task_states[task_id] = TaskStatus.PENDING
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        return self.get_workflow(workflow_id)

    def begin_task(self, workflow_id: str, task_id: str) -> Workflow:
        """Mark a PENDING task as RUNNING and emit task.started.

        Only legal while the workflow is EXECUTING; a task that is not
        PENDING cannot start (no double execution within a pass).
        """
        workflow = self._store.get_workflow(workflow_id)
        task = self._require_task(workflow, task_id)
        if workflow.state is not WorkflowState.EXECUTING:
            raise ControllerError(
                f"workflow {workflow_id}: cannot start a task while "
                f"{workflow.state.value}"
            )  # before mutation
        if workflow.task_states.get(task_id, TaskStatus.PENDING) is not TaskStatus.PENDING:
            raise ControllerError(
                f"workflow {workflow_id}: task {task_id} is "
                f"{workflow.task_states.get(task_id).value}, expected PENDING"
            )  # before mutation

        workflow.task_states[task_id] = TaskStatus.RUNNING
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        index = next(
            i for i, t in enumerate(workflow.plan.tasks, start=1) if t.id == task_id
        )
        label = (task.title or task.description)[:120]
        self._emit(
            workflow,
            EventType.TASK_STARTED,
            task_id=task_id,
            decision_summary=(
                f"Executing task {index}/{len(workflow.plan.tasks)}: {label}"
            ),
            payload={
                "index": index,
                "total": len(workflow.plan.tasks),
                "requires_tool": task.requires_tool,
            },
        )
        return self.get_workflow(workflow_id)

    def record_observation(
        self, workflow_id: str, observation: Observation
    ) -> Workflow:
        """Append an observation and move the task RUNNING -> OBSERVED/FAILED.

        A failed observation marks the task FAILED (never COMPLETED). Only
        legal while the workflow is EXECUTING with the task RUNNING.
        """
        workflow = self._store.get_workflow(workflow_id)
        task_id = observation.task_id
        task = self._require_task(workflow, task_id)
        if workflow.state is not WorkflowState.EXECUTING:
            raise ControllerError(
                f"workflow {workflow_id}: cannot record an observation while "
                f"{workflow.state.value}"
            )  # before mutation
        status = workflow.task_states.get(task_id, TaskStatus.PENDING)
        if status is not TaskStatus.RUNNING:
            raise ControllerError(
                f"workflow {workflow_id}: task {task_id} is {status.value}, "
                f"expected RUNNING"
            )  # before mutation

        # Version safety: stamp the producing task's definition fingerprint so
        # evidence from an OLD task definition can never satisfy a NEW one.
        observation = observation.model_copy(
            update={"task_fingerprint": task.definition_fingerprint()}
        )
        workflow.observations.append(observation)
        workflow.task_states[task_id] = (
            TaskStatus.FAILED if observation.is_error else TaskStatus.OBSERVED
        )
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        payload: dict[str, JsonValue] = {
            "status": "error" if observation.is_error else "ok",
            "source": observation.source.value,
        }
        if observation.tool_name is not None:
            payload["tool_name"] = observation.tool_name
        if observation.error_code is not None:
            payload["error_code"] = observation.error_code
        self._emit(
            workflow,
            EventType.OBSERVATION_CREATED,
            task_id=task_id,
            decision_summary=observation.content_summary,
            payload=payload,
        )
        return self.get_workflow(workflow_id)

    def complete_task(self, workflow_id: str, task_id: str) -> Workflow:
        """Mark an OBSERVED task COMPLETED and emit task.completed.

        Requires a successful observation to exist for the task - a task can
        only be completed after its execution produced a valid observation.
        """
        workflow = self._store.get_workflow(workflow_id)
        task = self._require_task(workflow, task_id)
        if workflow.state is not WorkflowState.EXECUTING:
            raise ControllerError(
                f"workflow {workflow_id}: cannot complete a task while "
                f"{workflow.state.value}"
            )  # before mutation
        status = workflow.task_states.get(task_id, TaskStatus.PENDING)
        if status is not TaskStatus.OBSERVED:
            raise ControllerError(
                f"workflow {workflow_id}: task {task_id} is {status.value}, "
                f"expected OBSERVED before completion"
            )  # before mutation
        successful = [
            o for o in workflow.observations if o.task_id == task_id and not o.is_error
        ]
        if not successful:
            raise ControllerError(
                f"workflow {workflow_id}: task {task_id} has no successful "
                f"observation to complete against"
            )  # before mutation

        workflow.task_states[task_id] = TaskStatus.COMPLETED
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        index = next(
            i for i, t in enumerate(workflow.plan.tasks, start=1) if t.id == task_id
        )
        label = (task.title or task.description)[:120]
        self._emit(
            workflow,
            EventType.TASK_COMPLETED,
            task_id=task_id,
            decision_summary=f"Task {index}/{len(workflow.plan.tasks)} complete: {label}",
            payload={
                "index": index,
                "total": len(workflow.plan.tasks),
                "observation_count": len(
                    [o for o in workflow.observations if o.task_id == task_id]
                ),
            },
        )
        return self.get_workflow(workflow_id)

    def finish_execution(self, workflow_id: str) -> Workflow:
        """EXECUTING -> AWAITING_VALIDATION once every task is executed.

        Deliberately NOT COMPLETED: finishing the plan does not prove the
        user's goal was satisfied - that decision belongs to validation.
        """
        workflow = self._store.get_workflow(workflow_id)
        unfinished = [
            t.id
            for t in workflow.plan.tasks
            if workflow.task_states.get(t.id, TaskStatus.PENDING)
            in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]
        if unfinished:
            raise ControllerError(
                f"workflow {workflow_id}: cannot finish execution with "
                f"unfinished tasks: {', '.join(unfinished)}"
            )  # before mutation
        updated = self.transition(
            workflow_id,
            WorkflowState.AWAITING_VALIDATION,
            decision_summary="All planned tasks executed; awaiting validation.",
        )
        self._emit(
            updated,
            EventType.EXECUTION_FINISHED,
            decision_summary=(
                f"All {len(updated.plan.tasks)} task(s) executed; "
                f"awaiting validation."
            ),
            payload={
                "tasks": len(updated.plan.tasks),
                "observations": len(updated.observations),
            },
        )
        return self.get_workflow(workflow_id)

    def record_execution_failure(
        self, workflow_id: str, failure: FailureContext
    ) -> Workflow:
        """Emit execution.failed, then transition EXECUTING -> FAILED.

        The event is recorded while still EXECUTING (so it reflects what was
        happening), then the workflow moves to FAILED with a structured
        failure record. No mutation happens before the transition is checked.
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.EXECUTING:
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.FAILED,
                reason="execution failure may only be recorded while EXECUTING",
            )
        self._emit(
            workflow,
            EventType.EXECUTION_FAILED,
            task_id=failure.task_id,
            decision_summary=f"{failure.error_code}: {failure.detail}"[:500],
            payload={
                "error_code": failure.error_code,
                "detail": failure.detail[:500],
                "stage": failure.stage.value,
            },
        )
        return self.fail(workflow_id, failure)

    # ------------------------------------------------------------------
    # Validation and replanning (Phase 5)
    # ------------------------------------------------------------------

    def begin_validation(self, workflow_id: str) -> Workflow:
        """AWAITING_VALIDATION -> VALIDATING. Marks that validation starts.

        Clears any previous verdict: a fresh validation round starts empty,
        so a workflow resumed in VALIDATING can always tell "no verdict yet
        this round" from "verdict recorded this round".
        """
        updated = self.transition(
            workflow_id,
            WorkflowState.VALIDATING,
            decision_summary="Goal validation started.",
        )
        updated.validation = None
        updated.updated_at = utcnow()
        self._store.save_workflow(updated)
        self._emit(
            updated,
            EventType.VALIDATION_STARTED,
            decision_summary="Validating execution against the goal.",
            payload={"replan_count": updated.replan_count},
        )
        return self.get_workflow(workflow_id)

    def record_validation(
        self, workflow_id: str, result: ValidationResult
    ) -> Workflow:
        """Store a ValidationResult and emit validation.passed/failed.

        The workflow stays VALIDATING: storing the verdict is not a state
        transition. The subsequent action (complete, replan, or fail) owns
        the transition. A PASS carrying violated/unresolved constraints is
        rejected outright - deterministic results always win.
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.VALIDATING:
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.VALIDATING,
                reason="validation may only be recorded while VALIDATING",
            )  # before any mutation
        if result.status is ValidationStatus.PASS and (
            result.violated_constraints or result.unresolved_constraints
        ):
            raise ControllerError(
                f"workflow {workflow_id}: PASS is impossible with "
                f"violated={result.violated_constraints} "
                f"unresolved={result.unresolved_constraints}"
            )  # before any mutation

        workflow.validation = result
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        payload: dict[str, JsonValue] = {
            "status": result.status.value,
            "reason_summary": result.reason_summary,
            "satisfied_constraints": result.satisfied_constraints,
            "violated_constraints": result.violated_constraints,
            "unresolved_constraints": result.unresolved_constraints,
            "evidence_count": len(result.evidence),
            "replan_count": workflow.replan_count,
        }
        event_type = (
            EventType.VALIDATION_PASSED
            if result.status is ValidationStatus.PASS
            else EventType.VALIDATION_FAILED
        )
        summary = (
            "Goal satisfied by the available evidence."
            if result.status is ValidationStatus.PASS
            else f"Validation {result.status.value}: {result.reason_summary}"
        )
        self._emit(
            workflow,
            event_type,
            decision_summary=summary[:500],
            payload=payload,
        )
        return self.get_workflow(workflow_id)

    def begin_replanning(
        self, workflow_id: str, *, reason: str | None = None
    ) -> Workflow:
        """VALIDATING -> REPLANNING and consume one replan attempt.

        The replan budget is checked BEFORE any mutation: when the limit is
        reached this raises ``MaxReplansExceededError`` with the workflow
        still VALIDATING (and with its validation result intact), so the
        caller can terminate the workflow without entering REPLANNING.
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.VALIDATING:
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.REPLANNING,
                reason="replanning may only begin while VALIDATING",
            )  # before any mutation
        if workflow.validation is None:
            raise ControllerError(
                f"workflow {workflow_id}: cannot replan without a stored "
                f"validation result"
            )  # before any mutation
        if workflow.replan_count >= self._max_replans:
            raise MaxReplansExceededError(
                workflow.id, workflow.replan_count, self._max_replans
            )  # before any mutation

        detail = reason or workflow.validation.reason_summary
        updated = self.transition(
            workflow_id,
            WorkflowState.REPLANNING,
            decision_summary=f"Replanning: {detail}"[:500],
        )
        self._emit(
            updated,
            EventType.REPLANNING_STARTED,
            decision_summary=detail[:500],
            payload={
                "reason": detail[:500],
                "validation_status": updated.validation.status.value
                if updated.validation
                else None,
                "replan_count": updated.replan_count,
                "limit": self._max_replans,
            },
        )
        self.record_replan(workflow_id)  # budget re-checked; increments + emits
        return self.get_workflow(workflow_id)

    def replace_plan(
        self, workflow_id: str, plan: Plan, *, reason: str
    ) -> Workflow:
        """Accept a validated replan proposal: REPLANNING -> PLANNING.

        - appends the superseded plan to ``plan_history`` (append-only)
        - bumps ``plan.version`` (old version + 1)
        - merges task states: identical ``definition_fingerprint`` keeps its
          COMPLETED/OBSERVED status (preserved work will not re-run); new or
          redefined tasks reset to PENDING; orphaned ids are dropped
        - emits replan.proposed + replan.accepted, then the transition

        Validates everything BEFORE mutating; an identical plan (same
        fingerprint) is rejected with ``PlanReplacementError``. Any rejection
        emits ``replan.rejected`` first (append-only diagnosis), and the
        workflow state/plan/history are left untouched.
        """
        workflow = self._store.get_workflow(workflow_id)

        # --- Phase 1: validate BEFORE any mutation ---------------------
        if workflow.state is not WorkflowState.REPLANNING:
            self._emit(
                workflow,
                EventType.REPLAN_REJECTED,
                decision_summary="Plan replacement attempted outside REPLANNING.",
                payload={
                    "reason": "invalid_state",
                    "state": workflow.state.value,
                },
            )  # bookkeeping event only; no plan/state mutation
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.PLANNING,
                reason="a plan may only be replaced while REPLANNING",
            )  # before any mutation
        if workflow.plan is None:
            self._emit(
                workflow,
                EventType.REPLAN_REJECTED,
                decision_summary="No current plan to replace.",
                payload={"reason": "no_current_plan"},
            )
            raise PlanReplacementError(
                f"workflow {workflow_id}: cannot replace a plan that does "
                f"not exist"
            )  # before any mutation
        if plan.fingerprint() == workflow.plan.fingerprint():
            self._emit(
                workflow,
                EventType.REPLAN_REJECTED,
                decision_summary="Proposed plan is identical to the current plan.",
                payload={
                    "reason": "identical_plan",
                    "current_version": workflow.plan.version,
                },
            )
            raise PlanReplacementError(
                f"workflow {workflow_id}: refused pointless replan - the "
                f"proposed plan is identical to the current one"
            )  # before any mutation

        # --- Phase 2: compute everything (pure, no mutation) -----------
        old_plan = workflow.plan
        revision = PlanRevision(
            plan=old_plan.model_copy(deep=True),
            reason=reason.strip()[:500] or "Replanning.",
            validation=(
                workflow.validation.model_copy(deep=True)
                if workflow.validation is not None
                else None
            ),
            new_plan_version=old_plan.version + 1,
        )
        new_plan = plan.model_copy(deep=True)
        new_plan.version = old_plan.version + 1

        old_tasks = {task.id: task for task in old_plan.tasks}
        merged: dict[str, TaskStatus] = {}
        preserved: list[str] = []
        invalidated: set[str] = set()
        for task in new_plan.tasks:
            prior = old_tasks.get(task.id)
            prior_state = (
                workflow.task_states.get(task.id)
                if prior is not None
                else None
            )
            unchanged = (
                prior is not None
                and prior_state is not None
                and prior.definition_fingerprint() == task.definition_fingerprint()
            )
            dep_changed = bool(invalidated & set(task.depends_on))
            if unchanged and not dep_changed:
                merged[task.id] = prior_state
                if prior_state is TaskStatus.COMPLETED:
                    preserved.append(task.id)
            else:
                # new, redefined, or depends-on-a-reset task -> must re-run
                merged[task.id] = TaskStatus.PENDING
                invalidated.add(task.id)

        # --- Phase 3: apply atomically (single save) --------------------
        workflow.plan = new_plan
        workflow.task_states = merged
        workflow.plan_history.append(revision)
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)

        # --- Phase 4: events + transition --------------------------------
        self._emit(
            workflow,
            EventType.REPLAN_PROPOSED,
            decision_summary=f"Revised plan v{new_plan.version} proposed.",
            payload={
                "version": new_plan.version,
                "tasks": len(new_plan.tasks),
                "task_ids": [task.id for task in new_plan.tasks],
            },
        )
        self._emit(
            workflow,
            EventType.REPLAN_ACCEPTED,
            decision_summary=reason[:500],
            payload={
                "version": new_plan.version,
                "previous_version": old_plan.version,
                "preserved_completed": preserved,
                "reason": reason[:500],
            },
        )
        return self.transition(
            workflow_id,
            WorkflowState.PLANNING,
            decision_summary="Revised plan accepted; recording it.",
        )

    def finish_with_validation(
        self, workflow_id: str, final_answer: str
    ) -> Workflow:
        """VALIDATING -> COMPLETED, only with a stored PASS validation."""
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.VALIDATING:
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.COMPLETED,
                reason="completion via validation requires VALIDATING",
            )  # before any mutation
        if workflow.validation is None:
            raise ControllerError(
                f"workflow {workflow_id}: cannot complete before validation "
                f"is recorded"
            )  # before any mutation
        if workflow.validation.status is not ValidationStatus.PASS:
            raise ControllerError(
                f"workflow {workflow_id}: cannot complete - validation is "
                f"{workflow.validation.status.value}, not PASS"
            )  # before any mutation
        return self.complete(workflow_id, final_answer)

    def record_synthesis(
        self, workflow_id: str, synthesis: SynthesisResult
    ) -> Workflow:
        """Store the synthesizer's final answer and complete the workflow.

        Only legal while VALIDATING with a stored PASS (same guard as
        ``finish_with_validation``), plus a defense-in-depth check: every
        constraint the synthesis claims satisfied must be in the PASS
        validation's satisfied set - presentation can never overclaim.
        Emits synthesis.completed, then completes (workflow.completed).
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.VALIDATING:
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.COMPLETED,
                reason="synthesis may only be recorded while VALIDATING",
            )  # before any mutation
        if workflow.validation is None:
            raise ControllerError(
                f"workflow {workflow_id}: cannot record synthesis before "
                f"validation is recorded"
            )  # before any mutation
        if workflow.validation.status is not ValidationStatus.PASS:
            raise ControllerError(
                f"workflow {workflow_id}: cannot record synthesis - "
                f"validation is {workflow.validation.status.value}, not PASS"
            )  # before any mutation
        validated = set(workflow.validation.satisfied_constraints)
        overclaimed = [
            constraint_id
            for constraint_id in synthesis.constraints_satisfied
            if constraint_id not in validated
        ]
        if overclaimed:
            raise ControllerError(
                f"workflow {workflow_id}: synthesis claims constraints not "
                f"in the PASS validation: {sorted(overclaimed)}"
            )  # before any mutation

        workflow.synthesis = synthesis
        workflow.final_answer = synthesis.final_answer
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.SYNTHESIS_COMPLETED,
            decision_summary=synthesis.final_answer[:500],
            payload={
                "key_results": len(synthesis.key_results),
                "constraints_satisfied": synthesis.constraints_satisfied,
                "caveats": len(synthesis.caveats),
                "next_actions": len(synthesis.next_actions),
            },
        )
        return self.complete(workflow_id, synthesis.final_answer)

    def fail_unsatisfied_goal(
        self, workflow_id: str, failure: FailureContext
    ) -> Workflow:
        """VALIDATING -> FAILED because validation did not PASS.

        Requires a stored non-PASS validation: ORBIT fails only on evidence,
        never silently.
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow.state is not WorkflowState.VALIDATING:
            raise IllegalTransitionError(
                workflow.state,
                WorkflowState.FAILED,
                reason="unsatisfied-goal failure may only be recorded while "
                "VALIDATING",
            )  # before any mutation
        if workflow.validation is None:
            raise ControllerError(
                f"workflow {workflow_id}: cannot fail an unsatisfied goal "
                f"before validation is recorded"
            )  # before any mutation
        if workflow.validation.status is ValidationStatus.PASS:
            raise ControllerError(
                f"workflow {workflow_id}: cannot fail an unsatisfied goal "
                f"when validation passed"
            )  # before any mutation
        return self.fail(workflow_id, failure)

    # ------------------------------------------------------------------
    # Replan accounting
    # ------------------------------------------------------------------

    def record_replan(self, workflow_id: str) -> int:
        """Count one replanning attempt.

        Raises ``MaxReplansExceededError`` without modifying state when the
        workflow has already used all ``max_replans`` attempts. The caller
        (later phase) converts that into FAILED.
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow.replan_count >= self._max_replans:
            raise MaxReplansExceededError(
                workflow.id, workflow.replan_count, self._max_replans
            )
        workflow.replan_count += 1
        workflow.updated_at = utcnow()
        self._store.save_workflow(workflow)
        self._emit(
            workflow,
            EventType.WORKFLOW_REPLANNED,
            decision_summary="Replanning after validation failure.",
            payload={
                "replan_count": workflow.replan_count,
                "limit": self._max_replans,
            },
        )
        return workflow.replan_count

    # ------------------------------------------------------------------
    # Event emission (used by the controller and, later, by agent stages)
    # ------------------------------------------------------------------

    def emit_event(
        self,
        workflow_id: str,
        event_type: EventType,
        *,
        decision_summary: str | None = None,
        task_id: str | None = None,
        payload: dict[str, JsonValue] | None = None,
    ) -> Event:
        """Append a domain event to the workflow's log (state-checked access)."""
        workflow = self._store.get_workflow(workflow_id)
        return self._emit(
            workflow,
            event_type,
            decision_summary=decision_summary,
            task_id=task_id,
            payload=payload,
        )

    def _emit(
        self,
        workflow: Workflow,
        event_type: EventType,
        *,
        decision_summary: str | None = None,
        task_id: str | None = None,
        payload: dict[str, JsonValue] | None = None,
    ) -> Event:
        log = self._store.get_event_log(workflow.id)
        event = log.append(
            event_type=event_type,
            state=workflow.state,
            task_id=task_id,
            decision_summary=decision_summary,
            payload=payload,
        )
        self._store.save_event_log(workflow.id, log)
        workflow.event_count = len(log)
        self._store.save_workflow(workflow)
        return event

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def _require_task(self, workflow: Workflow, task_id: str) -> Task:
        """Return the plan task with ``task_id`` or raise (no mutation)."""
        if workflow.plan is None:
            raise ControllerError(
                f"workflow {workflow.id}: workflow has no plan"
            )
        for task in workflow.plan.tasks:
            if task.id == task_id:
                return task
        raise ControllerError(
            f"workflow {workflow.id}: unknown task {task_id!r}"
        )

    def get_workflow(self, workflow_id: str) -> Workflow:
        return self._store.get_workflow(workflow_id)

    def events(self, workflow_id: str) -> list[Event]:
        """All events for a workflow (raises if the workflow is unknown)."""
        self._store.get_workflow(workflow_id)  # existence check
        return list(self._store.get_event_log(workflow_id).all())

    def events_after(self, workflow_id: str, seq: int) -> list[Event]:
        """Events after ``seq`` for streaming/polling clients."""
        self._store.get_workflow(workflow_id)  # existence check
        return self._store.get_event_log(workflow_id).after(seq)

    def can_transition(self, workflow_id: str, target: WorkflowState) -> bool:
        """Non-raising check of whether a transition would be legal."""
        workflow = self._store.get_workflow(workflow_id)
        return is_legal(workflow.state, target)
