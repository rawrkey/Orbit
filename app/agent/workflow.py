"""WorkflowOrchestrator: one call from goal to terminal report.

Dispatches on workflow state, so the SAME entry point handles a fresh run,
a clarification round, and resume after interruption:

    RECEIVED/ANALYZING -> analyze -> (clarification | plan) -> loop
    AWAITING_CLARIFICATION + answers -> record answers -> plan -> loop
    PLANNING (plan awaiting record or first planning) -> loop/plan -> loop
    READY/EXECUTING/AWAITING_VALIDATION/VALIDATING/REPLANNING -> loop
    COMPLETED/FAILED -> already_terminal report

Stage failures become structured workflow failures (analysis_error,
planning_error, infeasible) with the workflow in FAILED - the orchestrator
never raises for goal outcomes; exceptions are reserved for programming
errors (``OrchestratorError``) and unknown workflows.
"""

from __future__ import annotations

from app.agent.adaptive_loop import (
    AdaptiveLoop,
    AdaptiveRunReport,
)
from app.agent.controller import WorkflowController
from app.agent.executor import Executor
from app.agent.goal_analyzer import GoalAnalysisError, GoalAnalyzer
from app.agent.planner import PlanValidationError, Planner
from app.agent.replanner import Replanner
from app.agent.synthesizer import Synthesizer
from app.agent.validator import Validator
from app.llm.base import LLMError, LLMOutputInvalidError, LLMTimeoutError
from app.state.models import FailureContext, Workflow, WorkflowState

__all__ = ["OrchestratorError", "WorkflowOrchestrator"]

S = WorkflowState


class OrchestratorError(Exception):
    """A programming error: the workflow is not in a dispatchable shape."""


def _llm_error_code(exc: LLMError) -> str:
    if isinstance(exc, LLMTimeoutError):
        return "LLM_TIMEOUT"
    if isinstance(exc, LLMOutputInvalidError):
        return "LLM_OUTPUT_INVALID"
    return "LLM_ERROR"


class WorkflowOrchestrator:
    """Drives a workflow from its current state to a terminal report."""

    def __init__(
        self,
        controller: WorkflowController,
        analyzer: GoalAnalyzer,
        planner: Planner,
        executor: Executor,
        validator: Validator,
        replanner: Replanner,
        *,
        synthesizer: Synthesizer | None = None,
        max_passes: int | None = None,
    ) -> None:
        self._controller = controller
        self._analyzer = analyzer
        self._planner = planner
        self._loop = AdaptiveLoop(
            controller,
            executor,
            validator,
            replanner,
            max_passes=max_passes,
            synthesizer=synthesizer,
        )

    async def run(
        self, workflow_id: str, *, answers: dict[str, str] | None = None
    ) -> AdaptiveRunReport:
        """Dispatch on state and drive the workflow to a terminal outcome.

        ``answers`` (question -> answer) is only meaningful when the
        workflow awaits clarification; at analysis time it short-circuits
        the clarification round when it covers every required question.
        """
        workflow = self._controller.get_workflow(workflow_id)  # NotFound
        state = workflow.state

        if state in (S.COMPLETED, S.FAILED):
            return self._state_report(workflow, termination="already_terminal")

        if state in (S.RECEIVED, S.ANALYZING):
            return await self._analyze(workflow, answers)

        if state is S.AWAITING_CLARIFICATION:
            if not answers:
                return self._state_report(
                    workflow, termination="awaiting_clarification"
                )
            self._controller.record_clarification(workflow_id, answers)
            return await self._plan(workflow_id)

        if state is S.PLANNING:
            return await self._plan(workflow_id)

        # READY / EXECUTING / AWAITING_VALIDATION / VALIDATING / REPLANNING
        return await self._loop.run(workflow_id)

    # ------------------------------------------------------------------
    # Analysis stage
    # ------------------------------------------------------------------

    async def _analyze(
        self,
        workflow: Workflow,
        answers: dict[str, str] | None,
    ) -> AdaptiveRunReport:
        workflow_id = workflow.id
        if workflow.state is S.RECEIVED:
            self._controller.begin_analysis(workflow_id)
            workflow = self._controller.get_workflow(workflow_id)

        try:
            analysis = await self._analyzer.analyze(workflow.goal)
        except GoalAnalysisError as error:
            failure = FailureContext(
                stage=S.ANALYZING,
                error_code="ANALYSIS_INVALID",
                detail="; ".join(error.issues)[:500],
            )
            self._controller.fail(workflow_id, failure)
            return self._state_report(
                self._controller.get_workflow(workflow_id),
                termination="analysis_error",
            )
        except LLMError as error:
            failure = FailureContext(
                stage=S.ANALYZING,
                error_code=_llm_error_code(error),
                detail=str(error)[:500] or "goal analysis LLM call failed",
            )
            self._controller.fail(workflow_id, failure)
            return self._state_report(
                self._controller.get_workflow(workflow_id),
                termination="analysis_error",
            )

        if not analysis.is_feasible:
            failure = FailureContext(
                stage=S.ANALYZING,
                error_code="GOAL_INFEASIBLE",
                detail=analysis.reason_summary[:500],
            )
            self._controller.fail(workflow_id, failure)
            return self._state_report(
                self._controller.get_workflow(workflow_id),
                termination="infeasible",
            )

        if analysis.clarification_required:
            questions = [
                f"Please provide: {name}" for name in analysis.required_inputs
            ]
            covered = bool(answers) and all(
                str(answers.get(question, "")).strip() for question in questions
            )
            if not covered:
                self._controller.require_clarification(
                    workflow_id, analysis, questions
                )
                return self._state_report(
                    self._controller.get_workflow(workflow_id),
                    termination="awaiting_clarification",
                )
            # Answers already cover every question: skip the round trip.
            self._controller.record_analysis(workflow_id, analysis)
            covered_answers = {
                question: str(answers[question]).strip()[:500]
                for question in questions
                if answers is not None
            }
            return await self._plan(
                workflow_id, clarifications=covered_answers
            )

        self._controller.record_analysis(workflow_id, analysis)
        return await self._plan(workflow_id)

    # ------------------------------------------------------------------
    # Planning stage
    # ------------------------------------------------------------------

    async def _plan(
        self,
        workflow_id: str,
        *,
        clarifications: dict[str, str] | None = None,
    ) -> AdaptiveRunReport:
        workflow = self._controller.get_workflow(workflow_id)

        if workflow.plan is not None:
            # A plan is already stored and awaits recording (resume after a
            # replan): the loop's PLANNING branch records it.
            return await self._loop.run(workflow_id)

        analysis = workflow.analysis
        if analysis is None:
            raise OrchestratorError(
                f"workflow {workflow_id}: PLANNING without a stored analysis"
            )
        effective = clarifications or workflow.clarifications or None

        try:
            plan = await self._planner.plan(
                workflow.goal,
                analysis,
                clarifications=effective,
            )
        except PlanValidationError as error:
            self._controller.record_plan_validation_failure(
                workflow_id, error.issues
            )
            failure = FailureContext(
                stage=S.PLANNING,
                error_code="PLAN_INVALID",
                detail=error.issues[0][:500],
            )
            self._controller.fail(workflow_id, failure)
            return self._state_report(
                self._controller.get_workflow(workflow_id),
                termination="planning_error",
            )
        except LLMError as error:
            failure = FailureContext(
                stage=S.PLANNING,
                error_code=_llm_error_code(error),
                detail=str(error)[:500] or "planning LLM call failed",
            )
            self._controller.fail(workflow_id, failure)
            return self._state_report(
                self._controller.get_workflow(workflow_id),
                termination="planning_error",
            )

        self._controller.record_plan(workflow_id, plan)
        return await self._loop.run(workflow_id)

    # ------------------------------------------------------------------

    @staticmethod
    def _state_report(workflow: Workflow, *, termination: str) -> AdaptiveRunReport:
        """Report for dispatch outcomes that did not run the loop."""
        return AdaptiveRunReport(
            workflow_id=workflow.id,
            success=workflow.state is S.COMPLETED,
            final_state=workflow.state,
            termination=termination,
            failure=workflow.failure,
        )
