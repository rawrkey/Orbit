"""Execution engine: runs an already-validated Plan one task at a time.

Responsibilities (and nothing more):
1. select the next executable task deterministically (never via the LLM)
2. respect task dependencies and plan ordering
3. ask the LLM for a ToolDecision when a task requires a tool
4. validate that proposal through the existing decision path
5. execute the validated ToolCall through the existing execution primitive
6. convert outcomes into Observations via the Observer
7. store observations and task lifecycle changes through the Controller
8. stop with a typed failure on the first failed task
9. finish the pass at AWAITING_VALIDATION when every task executed

The Executor does NOT create plans, does NOT analyze the goal, does NOT
modify the plan, does NOT decide whether the goal was satisfied, and does
NOT own any state transition - the Controller remains the sole owner of
workflow state mutation. It contains no replanning or validation logic;
those later stages attach to the workflow after this pass.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.agent.controller import WorkflowController
from app.agent.observer import Observer
from app.llm.base import (
    LLMError,
    LLMOutputInvalidError,
    LLMProvider,
    LLMTimeoutError,
)
from app.llm.schemas import TaskResult, ToolDecision
from app.state.events import EventType
from app.state.models import (
    Constraint,
    FailureContext,
    JsonValue,
    Observation,
    Plan,
    Task,
    TaskStatus,
    Workflow,
    WorkflowState,
)
from app.tools.base import MalformedToolInputError, ToolSpec
from app.tools.decision import resolve_tool_decision
from app.tools.execution import execute_tool_call
from app.tools.registry import ToolNotFoundError, ToolRegistry

__all__ = [
    "MAX_PREVIOUS_OBSERVATIONS",
    "ExecutionContext",
    "ExecutionError",
    "ExecutionReport",
    "Observer",
    "UnexecutableTasksError",
    "Executor",
    "ObservationDigest",
    "build_execution_context",
    "select_next_task",
]

MAX_PREVIOUS_OBSERVATIONS = 5
"""How many prior observations are exposed to the LLM (bounded context)."""

# Task statuses that count as executed for dependency purposes.
_SATISFIED: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.PASSED}
)


class ExecutionError(Exception):
    """Typed failure of the execution engine itself (invalid state/bounds)."""


class UnexecutableTasksError(ExecutionError):
    """Unfinished tasks remain but none of them can be executed.

    Raised instead of silently skipping blocked tasks - dependencies are
    deterministic and the LLM never gets a say in ordering.
    """

    def __init__(self, blocked_task_ids: list[str]) -> None:
        self.blocked_task_ids = list(blocked_task_ids)
        super().__init__(
            "no executable task: blocked by incomplete dependencies: "
            + ", ".join(self.blocked_task_ids)
        )


# ---------------------------------------------------------------------------
# Deterministic task selection
# ---------------------------------------------------------------------------


def select_next_task(
    plan: Plan, task_states: dict[str, TaskStatus]
) -> Task | None:
    """Return the earliest executable PENDING task, or None if all are done.

    Deterministic rules (the LLM is never consulted):
    - handled tasks (OBSERVED/COMPLETED/PASSED/FAILED/SKIPPED) are ignored
    - a PENDING task is executable only if every dependency is in a
      satisfied status (COMPLETED or PASSED)
    - if unfinished (PENDING) tasks remain but none is executable, raise
      ``UnexecutableTasksError`` - blocked tasks are never silently skipped
    """
    pending = [
        task
        for task in plan.tasks
        if task_states.get(task.id, TaskStatus.PENDING) is TaskStatus.PENDING
    ]
    if not pending:
        return None

    for task in plan.tasks:
        if task_states.get(task.id, TaskStatus.PENDING) is not TaskStatus.PENDING:
            continue
        dependencies = task.depends_on
        if all(
            task_states.get(dep) in _SATISFIED for dep in dependencies
        ):
            return task

    raise UnexecutableTasksError([task.id for task in pending])


# ---------------------------------------------------------------------------
# Bounded execution context
# ---------------------------------------------------------------------------


class ObservationDigest(BaseModel):
    """One-line view of a previous observation (bounded context entry)."""

    task_id: str
    summary: str = Field(min_length=1, max_length=500)
    is_error: bool = False


class ExecutionContext(BaseModel):
    """Structured, bounded context handed to the LLM for one task.

    Never contains the raw event log, full prompts, or chain-of-thought.
    """

    goal: str
    objective: str
    constraints: list[Constraint] = Field(default_factory=list)
    task: Task
    previous_observations: list[ObservationDigest] = Field(default_factory=list)
    available_tools: list[ToolSpec] = Field(default_factory=list)


def build_execution_context(
    workflow: Workflow,
    task: Task,
    registry: ToolRegistry,
    *,
    max_observations: int = MAX_PREVIOUS_OBSERVATIONS,
) -> ExecutionContext:
    """Build the bounded context for ``task`` from the current workflow."""
    analysis = workflow.analysis
    objective = analysis.objective if analysis is not None else workflow.goal

    constraints: list[Constraint] = []
    if analysis is not None and analysis.constraints:
        referenced = set(task.constraint_ids)
        if referenced:
            constraints = [c for c in analysis.constraints if c.id in referenced]
        if not constraints:
            constraints = list(analysis.constraints)

    previous = [
        ObservationDigest(
            task_id=obs.task_id,
            summary=obs.content_summary,
            is_error=obs.is_error,
        )
        for obs in workflow.observations
        if obs.task_id != task.id
    ][-max_observations:]

    return ExecutionContext(
        goal=workflow.goal,
        objective=objective,
        constraints=constraints,
        task=task,
        previous_observations=previous,
        available_tools=registry.list_tools(),
    )


# ---------------------------------------------------------------------------
# System prompts (bounded; no chain-of-thought requested or stored)
# ---------------------------------------------------------------------------

TOOL_DECISION_SYSTEM = """\
You choose exactly one tool for the current task in an adaptive workflow agent.
Respond with a single JSON object matching the ToolDecision schema:
{"tool_name": <string>, "arguments": <object>, "reason_summary": <string>}.
Pick the tool ONLY from available_tools in the execution context. Arguments
must satisfy that tool's input schema. reason_summary is a single concise
sentence - never chain-of-thought. If no available tool fits, choose the
closest capable tool and explain briefly.
"""

REASONING_SYSTEM = """\
You complete one reasoning-only task (no tool) in an adaptive workflow agent.
Respond with a single JSON object matching the TaskResult schema:
{"reason_summary": <string>, "data": <any JSON or null>}.
reason_summary states what was determined, in one concise sentence (max 500
characters). data holds any structured result. Never output chain-of-thought.
Base the result only on the execution context - never invent tool output.
"""


def _context_block(context: ExecutionContext) -> str:
    return (
        "Execution context (JSON):\n"
        + context.model_dump_json(exclude_none=True)
    )


def _tool_decision_prompt(context: ExecutionContext) -> str:
    return (
        "Choose a tool for this task and respond with a ToolDecision.\n\n"
        + _context_block(context)
    )


def _reasoning_prompt(context: ExecutionContext) -> str:
    return (
        "Complete this reasoning-only task and respond with a TaskResult.\n\n"
        + _context_block(context)
    )


def _llm_error_code(exc: LLMError) -> str:
    if isinstance(exc, LLMTimeoutError):
        return "LLM_TIMEOUT"
    if isinstance(exc, LLMOutputInvalidError):
        return "LLM_OUTPUT_INVALID"
    return "LLM_ERROR"


def _preview(value: JsonValue, limit: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ExecutionReport(BaseModel):
    """Outcome of one bounded execution pass."""

    workflow_id: str
    success: bool
    final_state: WorkflowState
    completed_task_ids: list[str] = Field(default_factory=list)
    failed_task_id: str | None = None
    failure: FailureContext | None = None
    observation_count: int = Field(default=0, ge=0)


class Executor:
    """Orchestrates one bounded execution pass over a validated plan.

    Stateless between calls except through the Controller's store: the
    workflow itself carries plan, task states, and observations.
    """

    def __init__(
        self,
        controller: WorkflowController,
        provider: LLMProvider,
        registry: ToolRegistry,
        *,
        observer: Observer | None = None,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> None:
        self._controller = controller
        self._provider = provider
        self._registry = registry
        self._observer = observer or Observer()
        self._temperature = temperature
        self._timeout_s = timeout_s

    async def execute(self, workflow_id: str) -> ExecutionReport:
        """Run every executable task in order, one pass, no replanning.

        Raises (without any plan modification):
        - ``ExecutionError`` if the workflow has no/empty plan or the loop
          exceeds the plan's task bound (defensive)
        - ``UnexecutableTasksError`` when blocked tasks remain (also records
          an execution failure so the workflow does not hang in EXECUTING)
        - ``IllegalTransitionError`` if the workflow is not READY (and not
          EXECUTING, which resumes an interrupted pass)
        - ``ControllerError`` on invalid task lifecycle operations

        On a failed task: records the failed observation, records a typed
        execution failure (workflow -> FAILED), and returns a report with
        ``success=False`` - it never continues past the first failure.
        """
        workflow = self._controller.get_workflow(workflow_id)
        plan = workflow.plan
        if plan is None:
            raise ExecutionError(f"workflow {workflow_id}: workflow has no plan")
        if not plan.tasks:
            raise ExecutionError(f"workflow {workflow_id}: plan has no tasks")

        if workflow.state is WorkflowState.EXECUTING:
            # Resume after an interruption: tasks left RUNNING by the dead
            # pass are reset to PENDING, then the pass continues in place.
            self._controller.recover_interrupted_tasks(workflow_id)
        else:
            # READY -> EXECUTING (controller-enforced; raises if not READY).
            self._controller.begin_execution(workflow_id)

        completed: list[str] = []
        total_observations = 0

        try:
            while True:
                workflow = self._controller.get_workflow(workflow_id)
                task = select_next_task(workflow.plan, workflow.task_states)
                if task is None:
                    break
                if len(completed) >= len(plan.tasks):
                    raise ExecutionError(
                        f"workflow {workflow_id}: execution exceeded the "
                        f"plan's {len(plan.tasks)}-task bound"
                    )

                self._controller.begin_task(workflow_id, task.id)
                observation = await self._run_task(workflow, task)
                self._controller.record_observation(workflow_id, observation)
                total_observations += 1

                if observation.is_error:
                    failure = FailureContext(
                        stage=WorkflowState.EXECUTING,
                        error_code=observation.error_code or "TASK_EXECUTION_FAILED",
                        detail=observation.content_summary,
                        task_id=task.id,
                    )
                    self._controller.record_execution_failure(
                        workflow_id, failure
                    )
                    return ExecutionReport(
                        workflow_id=workflow_id,
                        success=False,
                        final_state=WorkflowState.FAILED,
                        completed_task_ids=completed,
                        failed_task_id=task.id,
                        failure=failure,
                        observation_count=total_observations,
                    )

                self._controller.complete_task(workflow_id, task.id)
                completed.append(task.id)
        except UnexecutableTasksError as error:
            failure = FailureContext(
                stage=WorkflowState.EXECUTING,
                error_code="DEPENDENCY_DEADLOCK",
                detail=str(error),
            )
            self._controller.record_execution_failure(workflow_id, failure)
            raise

        self._controller.finish_execution(workflow_id)
        final = self._controller.get_workflow(workflow_id)
        return ExecutionReport(
            workflow_id=workflow_id,
            success=True,
            final_state=final.state,
            completed_task_ids=completed,
            observation_count=total_observations,
        )

    # ------------------------------------------------------------------
    # Per-task execution paths
    # ------------------------------------------------------------------

    async def _run_task(self, workflow: Workflow, task: Task) -> Observation:
        """Run one task and return its observation (never raises for task
        outcomes - LLM/tool failures become failed observations)."""
        context = build_execution_context(workflow, task, self._registry)
        try:
            if task.requires_tool:
                return await self._run_tool_task(workflow, task, context)
            return await self._run_reasoning_task(workflow, task, context)
        except LLMError as exc:
            return self._observer.observe_error(
                task_id=task.id,
                detail=str(exc),
                error_code=_llm_error_code(exc),
            )

    async def _run_tool_task(
        self, workflow: Workflow, task: Task, context: ExecutionContext
    ) -> Observation:
        """LLM proposal -> deterministic validation -> execution -> observation."""
        decision = await self._provider.complete(
            _tool_decision_prompt(context),
            ToolDecision,
            system=TOOL_DECISION_SYSTEM,
            temperature=self._temperature,
            timeout_s=self._timeout_s,
        )
        self._controller.emit_event(
            workflow.id,
            EventType.TOOL_DECISION_PROPOSED,
            task_id=task.id,
            decision_summary=decision.reason_summary[:500],
            payload={"tool_name": decision.tool_name},
        )

        # Deterministic gate: the proposal is not executable until the
        # registry and input schema both accept it.
        try:
            call = resolve_tool_decision(self._registry, decision)
        except ToolNotFoundError as exc:
            return self._observer.observe_rejected_decision(
                task_id=task.id,
                tool_name=decision.tool_name,
                detail=str(exc),
                error_code="TOOL_NOT_FOUND",
            )
        except MalformedToolInputError as exc:
            return self._observer.observe_rejected_decision(
                task_id=task.id,
                tool_name=decision.tool_name,
                detail=str(exc),
                error_code="MALFORMED_INPUT",
            )

        self._controller.emit_event(
            workflow.id,
            EventType.TOOL_CALLED,
            task_id=task.id,
            decision_summary=f"Selected {call.tool_name} with validated arguments.",
            payload={
                "tool_name": call.tool_name,
                "arguments_preview": _preview(call.arguments),
            },
        )
        result = execute_tool_call(self._registry, call)

        payload: dict[str, JsonValue] = {
            "tool_name": call.tool_name,
            "duration_ms": round(result.duration_ms, 2),
        }
        if result.error_code is not None:
            payload["error_code"] = result.error_code
        self._controller.emit_event(
            workflow.id,
            EventType.TOOL_COMPLETED if result.ok else EventType.TOOL_FAILED,
            task_id=task.id,
            decision_summary=(
                f"Tool {call.tool_name!r} "
                + ("returned successfully."
                   if result.ok else f"failed: {result.error_code}")
            ),
            payload=payload,
        )
        return self._observer.observe_tool_result(
            task_id=task.id, tool_name=call.tool_name, result=result
        )

    async def _run_reasoning_task(
        self, workflow: Workflow, task: Task, context: ExecutionContext
    ) -> Observation:
        """LLM structured result -> observation (schema-validated, no prose)."""
        result = await self._provider.complete(
            _reasoning_prompt(context),
            TaskResult,
            system=REASONING_SYSTEM,
            temperature=self._temperature,
            timeout_s=self._timeout_s,
        )
        return self._observer.observe_reasoning_result(task_id=task.id, result=result)
