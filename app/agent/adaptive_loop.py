"""Adaptive execution loop: execute -> validate -> replan -> execute.

One bounded loop ties the Phase 4 executor and the Phase 5 validator and
replanner together through the Controller, which owns every mutation:

    READY --execute--> AWAITING_VALIDATION --begin_validation--> VALIDATING
        |-- PASS --(optional synthesis)--> COMPLETED
        |-- FAIL/INSUFFICIENT --begin_replanning--> REPLANNING
                |-- proposal OK --replace_plan--> PLANNING --record_plan--> READY
                |-- budget exhausted / invalid proposal -----------> FAILED

The loop is state-driven, so it also RESUMES mid-flight workflows:
PLANNING (plan awaiting record), REPLANNING (proposal round), EXECUTING
(interrupted pass), AWAITING_VALIDATION, and VALIDATING (verdict pending
or just recorded) all continue from where they stopped. Terminal states
return an ``already_terminal`` report; unusable entry states (RECEIVED,
ANALYZING, AWAITING_CLARIFICATION - the orchestrator's responsibility)
raise ``AdaptiveLoopError``.

Failure semantics:
- an execution failure (tool/LLM/deadlock) terminates the loop immediately
  with NO replan
- a validation failure (FAIL or INSUFFICIENT) triggers bounded replanning
- replan budget exhaustion and invalid replan proposals terminate the loop
  with structured failures - exceptions are never converted into replans
- PASS with violated/unresolved constraints cannot occur (Controller guard)
- with a Synthesizer attached, PASS -> synthesis -> COMPLETED; a synthesis
  refusal or LLM error terminates with ``synthesis_error`` (never silent)

The loop is structurally bounded: every non-terminal pass either completes,
fails, or consumes one replan attempt, so runs <= max_replans + 1 passes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.controller import (
    MaxReplansExceededError,
    WorkflowController,
)
from app.agent.executor import Executor, UnexecutableTasksError
from app.agent.planner import PlanValidationError
from app.agent.replanner import Replanner
from app.agent.synthesizer import SynthesisError, Synthesizer
from app.agent.validator import InconsistentValidationError, Validator
from app.llm.base import LLMError, LLMOutputInvalidError, LLMTimeoutError
from app.state.events import EventType
from app.state.models import (
    FailureContext,
    SynthesisResult,
    ValidationResult,
    ValidationStatus,
    Workflow,
    WorkflowState,
)

__all__ = ["AdaptiveLoop", "AdaptiveLoopError", "AdaptiveRunReport"]


class AdaptiveLoopError(Exception):
    """The loop exceeded its defensive bound (indicates a logic bug)."""


class AdaptiveRunReport(BaseModel):
    """Structured outcome of one adaptive run (terminal by construction)."""

    workflow_id: str
    success: bool
    final_state: WorkflowState
    executions: int = Field(default=0, ge=0)
    validations: int = Field(default=0, ge=0)
    replans: int = Field(default=0, ge=0)
    termination: str = Field(min_length=1)
    failure: FailureContext | None = None


def _llm_error_code(exc: LLMError) -> str:
    if isinstance(exc, LLMTimeoutError):
        return "LLM_TIMEOUT"
    if isinstance(exc, LLMOutputInvalidError):
        return "LLM_OUTPUT_INVALID"
    return "LLM_ERROR"


_UNUSABLE_ENTRY = frozenset(
    {
        WorkflowState.RECEIVED,
        WorkflowState.ANALYZING,
        WorkflowState.AWAITING_CLARIFICATION,
    }
)


class AdaptiveLoop:
    """Drives READY workflows to a terminal state, adaptively."""

    def __init__(
        self,
        controller: WorkflowController,
        executor: Executor,
        validator: Validator,
        replanner: Replanner,
        *,
        max_passes: int | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self._controller = controller
        self._executor = executor
        self._validator = validator
        self._replanner = replanner
        self._synthesizer = synthesizer
        # Every non-terminal pass consumes a replan attempt; +1 for the
        # pass that completes without replanning.
        self._max_passes = (
            max_passes if max_passes is not None else controller.max_replans + 1
        )

    async def run(self, workflow_id: str) -> AdaptiveRunReport:
        """Run until the workflow reaches COMPLETED or FAILED.

        Entry may be READY (fresh run) or any mid-flight state (resume).
        Raises only for programming errors (unknown workflow, unusable
        entry state, or the defensive bound) - all goal/execution outcomes
        come back in the report.
        """
        executions = validations = replans = 0
        workflow = self._controller.get_workflow(workflow_id)

        while True:
            state = workflow.state

            # ---- terminal: nothing left to run --------------------------
            if state in (WorkflowState.COMPLETED, WorkflowState.FAILED):
                return self._report(
                    workflow_id, executions, validations, replans,
                    termination="already_terminal",
                )

            if state in _UNUSABLE_ENTRY:
                raise AdaptiveLoopError(
                    f"workflow {workflow_id}: the loop cannot run from "
                    f"{state.value} - analysis/clarification belongs to the "
                    f"orchestrator"
                )

            # ---- PLANNING: record the already-validated plan ------------
            if state is WorkflowState.PLANNING:
                assert workflow.plan is not None
                self._controller.record_plan(workflow_id, workflow.plan)
                workflow = self._controller.get_workflow(workflow_id)
                continue

            # ---- REPLANNING: propose + replace (fresh or resumed) --------
            if state is WorkflowState.REPLANNING:
                stored = workflow.validation
                if stored is None:
                    raise AdaptiveLoopError(
                        f"workflow {workflow_id}: REPLANNING without a "
                        f"stored validation result"
                    )
                try:
                    proposal = await self._replanner.propose(workflow, stored)
                except PlanValidationError as error:
                    self._controller.emit_event(
                        workflow_id,
                        EventType.REPLAN_REJECTED,
                        decision_summary=error.issues[0][:500],
                        payload={
                            "issues": [
                                issue[:300] for issue in error.issues[:20]
                            ],
                            "issue_count": len(error.issues),
                            "replan_count": workflow.replan_count,
                        },
                    )
                    failure = FailureContext(
                        stage=WorkflowState.REPLANNING,
                        error_code="REPLAN_INVALID",
                        detail=error.issues[0][:500],
                    )
                    self._controller.fail(workflow_id, failure)
                    return self._report(
                        workflow_id, executions, validations, replans,
                        termination="replan_invalid", failure=failure,
                    )
                except LLMError as error:
                    failure = FailureContext(
                        stage=WorkflowState.REPLANNING,
                        error_code=_llm_error_code(error),
                        detail=str(error)[:500] or "replanner LLM call failed",
                    )
                    self._controller.fail(workflow_id, failure)
                    return self._report(
                        workflow_id, executions, validations, replans,
                        termination="replan_error", failure=failure,
                    )

                replans += 1
                self._controller.replace_plan(
                    workflow_id,
                    proposal,
                    reason=f"Validation {stored.status.value}: "
                    f"{stored.reason_summary}"[:500],
                )
                workflow = self._controller.get_workflow(workflow_id)
                continue  # PLANNING -> recorded at the top of the loop

            # ---- READY / EXECUTING: one bounded execution pass -----------
            if state in (WorkflowState.READY, WorkflowState.EXECUTING):
                executions += 1
                if executions > self._max_passes:
                    raise AdaptiveLoopError(
                        f"workflow {workflow_id}: adaptive loop exceeded its "
                        f"bound of {self._max_passes} passes"
                    )
                try:
                    execution = await self._executor.execute(workflow_id)
                except UnexecutableTasksError:
                    # Executor already recorded the typed failure (-> FAILED).
                    return self._report(
                        workflow_id, executions, validations, replans,
                        termination="execution_failed",
                    )
                if not execution.success:
                    return self._report(
                        workflow_id, executions, validations, replans,
                        termination="execution_failed",
                        failure=execution.failure,
                    )
                workflow = self._controller.get_workflow(workflow_id)
                continue  # AWAITING_VALIDATION

            # ---- AWAITING_VALIDATION: start the validation round ---------
            if state is WorkflowState.AWAITING_VALIDATION:
                self._controller.begin_validation(workflow_id)
                workflow = self._controller.get_workflow(workflow_id)
                continue  # VALIDATING, no verdict yet this round

            # ---- VALIDATING: recorded verdict, or run the LLM call --------
            if state is WorkflowState.VALIDATING:
                stored = workflow.validation
                if stored is not None:
                    # A verdict was already recorded this round (resume).
                    if stored.status is ValidationStatus.PASS:
                        return await self._complete(
                            workflow_id,
                            workflow,
                            stored,
                            executions,
                            validations,
                            replans,
                        )
                    # FAIL / INSUFFICIENT -> bounded replanning.
                    try:
                        self._controller.begin_replanning(workflow_id)
                    except MaxReplansExceededError:
                        self._controller.emit_event(
                            workflow_id,
                            EventType.REPLANNING_EXHAUSTED,
                            decision_summary=(
                                "Replan budget exhausted; failing goal."
                            ),
                            payload={
                                "replan_count": workflow.replan_count,
                                "limit": self._controller.max_replans,
                                "validation_status": stored.status.value,
                            },
                        )
                        failure = FailureContext(
                            stage=WorkflowState.VALIDATING,
                            error_code="REPLAN_LIMIT_EXCEEDED",
                            detail=(
                                f"Goal not satisfied after "
                                f"{self._controller.max_replans} replanning "
                                f"attempt(s): {stored.reason_summary}"
                            )[:500],
                        )
                        self._controller.fail_unsatisfied_goal(
                            workflow_id, failure
                        )
                        return self._report(
                            workflow_id, executions, validations, replans,
                            termination="replan_exhausted", failure=failure,
                        )
                    workflow = self._controller.get_workflow(workflow_id)
                    continue  # REPLANNING

                # Fresh verdict round: deterministic-first LLM validation.
                try:
                    result = await self._validator.validate(workflow)
                except InconsistentValidationError as error:
                    failure = FailureContext(
                        stage=WorkflowState.VALIDATING,
                        error_code="VALIDATION_INCONSISTENT",
                        detail="; ".join(error.issues)[:500],
                    )
                    self._controller.fail(workflow_id, failure)
                    return self._report(
                        workflow_id, executions, validations, replans,
                        termination="validation_error", failure=failure,
                    )
                except LLMError as error:
                    failure = FailureContext(
                        stage=WorkflowState.VALIDATING,
                        error_code=_llm_error_code(error),
                        detail=str(error)[:500]
                        or "validation LLM call failed",
                    )
                    self._controller.fail(workflow_id, failure)
                    return self._report(
                        workflow_id, executions, validations, replans,
                        termination="validation_error", failure=failure,
                    )
                validations += 1
                self._controller.record_validation(workflow_id, result)
                workflow = self._controller.get_workflow(workflow_id)
                continue  # VALIDATING with the recorded verdict

            raise AdaptiveLoopError(
                f"workflow {workflow_id}: unhandled state {state.value}"
            )

    # ------------------------------------------------------------------
    # PASS -> completion (optionally via synthesis)
    # ------------------------------------------------------------------

    async def _complete(
        self,
        workflow_id: str,
        workflow: Workflow,
        validation: ValidationResult,
        executions: int,
        validations: int,
        replans: int,
    ) -> AdaptiveRunReport:
        if self._synthesizer is None:
            self._controller.finish_with_validation(
                workflow_id, validation.reason_summary
            )
            return self._report(
                workflow_id, executions, validations, replans,
                termination="validated",
            )
        try:
            synthesis: SynthesisResult = await self._synthesizer.synthesize(
                workflow, validation
            )
        except SynthesisError as error:
            failure = FailureContext(
                stage=WorkflowState.VALIDATING,
                error_code="SYNTHESIS_REFUSED",
                detail=str(error)[:500],
            )
            self._controller.fail(workflow_id, failure)
            return self._report(
                workflow_id, executions, validations, replans,
                termination="synthesis_error", failure=failure,
            )
        except LLMError as error:
            failure = FailureContext(
                stage=WorkflowState.VALIDATING,
                error_code=_llm_error_code(error),
                detail=str(error)[:500] or "synthesis LLM call failed",
            )
            self._controller.fail(workflow_id, failure)
            return self._report(
                workflow_id, executions, validations, replans,
                termination="synthesis_error", failure=failure,
            )
        self._controller.record_synthesis(workflow_id, synthesis)
        return self._report(
            workflow_id, executions, validations, replans,
            termination="validated",
        )

    # ------------------------------------------------------------------

    def _report(
        self,
        workflow_id: str,
        executions: int,
        validations: int,
        replans: int,
        *,
        termination: str,
        failure: FailureContext | None = None,
    ) -> AdaptiveRunReport:
        workflow = self._controller.get_workflow(workflow_id)
        return AdaptiveRunReport(
            workflow_id=workflow_id,
            success=workflow.state is WorkflowState.COMPLETED,
            final_state=workflow.state,
            executions=executions,
            validations=validations,
            replans=replans,
            termination=termination,
            failure=failure,
        )
