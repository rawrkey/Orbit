"""Planner: structured ``GoalAnalysis`` -> validated execution ``Plan``.

Responsibilities (LLM proposes, deterministic code disposes):
- ask the ``LLMProvider`` for a structured ``Plan`` given the goal and analysis
- validate the plan STRUCTURALLY with deterministic code:
  non-empty, unique task ids, dependency ids exist, no cycles, tasks listed
  in topological order, bounded task count, fresh PENDING status, tool_hint
  consistency, constraint references resolve to the analysis
- allow at most ONE bounded structural-repair call, then fail loudly with
  ``PlanValidationError`` listing every issue

The planner never mutates workflow state or the event log - the controller
does that after the caller receives a validated plan.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from app.llm.base import LLMProvider
from app.state.models import GoalAnalysis, Plan, Task, TaskStatus

__all__ = [
    "DEFAULT_MAX_PLAN_TASKS",
    "PlanValidationError",
    "Planner",
    "validate_plan",
]

DEFAULT_MAX_PLAN_TASKS = 7


class PlanValidationError(Exception):
    """The produced plan failed deterministic structural validation."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)
        summary = "; ".join(self.issues)
        super().__init__(f"plan invalid: {summary}")


PLANNER_SYSTEM_PROMPT = """\
You are the planning stage of ORBIT, an adaptive personal workflow agent.
Produce a JSON object matching the Plan schema: an ordered "tasks" array.

Rules:
- Every task: a stable unique id (e.g. "t1"), a short title, a description,
  expected_output, success_criteria, depends_on (ids of EARLIER tasks),
  requires_tool, tool_hint, and constraint_ids (ids of constraints from the
  analysis this task must respect).
- requires_tool / tool_hint contract - deterministic validation enforces
  it and rejects any plan that breaks it:
  - requires_tool=false -> tool_hint must be null or omitted entirely.
  - requires_tool=true -> tool_hint is REQUIRED: never null, never an
    invented or generic name - it must be EXACTLY one registered tool
    ({available_tools}).
  - Research, comparison, price, fact-finding, or web tasks need live
    results: requires_tool=true with tool_hint="web_search".
- Tasks must be listed in execution order: every dependency appears at a
  lower index than the task that needs it.
- depends_on may only reference ids present in the same tasks array.
- No cycles. Status of every task must be "pending".
- Keep the plan minimal and concrete: at most {max_tasks} tasks. Prefer
  fewer, clearly verifiable steps over many vague ones.
- Only mark requires_tool when the task genuinely needs a tool; otherwise
  write a task a person could complete from the given information.
- Do not invent constraints; reference only constraint ids from the analysis.
"""


def validate_plan(
    plan: Plan,
    *,
    analysis: GoalAnalysis | None = None,
    max_tasks: int = DEFAULT_MAX_PLAN_TASKS,
    available_tools: Collection[str] | None = None,
) -> None:
    """Deterministic structural validation of a proposed plan.

    Raises ``PlanValidationError`` listing EVERY issue found. This is pure
    structure: no LLM, no network, no state.

    ``available_tools`` (registry tool names) is optional: when given, every
    non-blank ``tool_hint`` must name one of them, so an invented tool can
    never reach execution. When omitted (legacy/direct callers), only the
    structural requires_tool/tool_hint consistency rules apply.
    """
    issues: list[str] = []
    known_tool_names = (
        set(available_tools) if available_tools is not None else None
    )

    if not plan.tasks:
        issues.append("plan has no tasks")
    if len(plan.tasks) > max_tasks:
        issues.append(
            f"plan has {len(plan.tasks)} tasks, limit is {max_tasks}"
        )

    ids: list[str] = [task.id for task in plan.tasks]
    seen: set[str] = set()
    for task_id in ids:
        if not task_id.strip():
            issues.append("task id is blank")
        elif task_id in seen:
            issues.append(f"duplicate task id: {task_id!r}")
        seen.add(task_id)

    index_of = {task_id: i for i, task_id in enumerate(ids)}
    known_constraint_ids = (
        {c.id for c in analysis.constraints} if analysis is not None else None
    )

    for position, task in enumerate(plan.tasks):
        if task.status is not TaskStatus.PENDING:
            issues.append(
                f"task {task.id!r} must start with status pending, "
                f"got {task.status.value}"
            )
        if not task.success_criteria.strip():
            issues.append(f"task {task.id!r} has blank success_criteria")
        if task.requires_tool and not (task.tool_hint or "").strip():
            issues.append(
                f"task {task.id!r} requires a tool but has no tool_hint"
            )
        if not task.requires_tool and task.tool_hint is not None:
            issues.append(
                f"task {task.id!r} declares a tool_hint but requires_tool "
                f"is false"
            )
        hint = (task.tool_hint or "").strip()
        if known_tool_names is not None and hint and hint not in known_tool_names:
            issues.append(
                f"task {task.id!r} tool_hint {task.tool_hint!r} is not a "
                f"registered tool (available: "
                f"{', '.join(sorted(known_tool_names))})"
            )
        if known_constraint_ids is not None:
            for constraint_id in task.constraint_ids:
                if constraint_id not in known_constraint_ids:
                    issues.append(
                        f"task {task.id!r} references unknown constraint "
                        f"{constraint_id!r}"
                    )
        for dependency in task.depends_on:
            if dependency not in index_of:
                issues.append(
                    f"task {task.id!r} depends on unknown task "
                    f"{dependency!r}"
                )
            elif index_of[dependency] >= position:
                issues.append(
                    f"task {task.id!r} depends on {dependency!r}, which is "
                    f"not listed earlier (order or cycle problem)"
                )

    if not issues:
        cycle = _find_cycle(plan.tasks)
        if cycle:
            issues.append("dependency cycle: " + " -> ".join(cycle))

    if issues:
        raise PlanValidationError(issues)


def _find_cycle(tasks: list[Task]) -> list[str] | None:
    """Return one dependency cycle as a task-id path, or None if acyclic."""
    graph = {task.id: [d for d in task.depends_on] for task in tasks}
    WHITE, GREY, BLACK = 0, 1, 2
    color = {task_id: WHITE for task_id in graph}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = GREY
        stack.append(node)
        for neighbor in graph[node]:
            if color[neighbor] == GREY:
                start = stack.index(neighbor)
                return stack[start:] + [neighbor]
            if color[neighbor] == WHITE:
                found = visit(neighbor)
                if found is not None:
                    return found
        stack.pop()
        color[node] = BLACK
        return None

    for task_id in graph:
        if color[task_id] == WHITE:
            found = visit(task_id)
            if found is not None:
                return found
    return None


class Planner:
    """Deterministic wrapper around the planning LLM call + validation."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_tasks: int = DEFAULT_MAX_PLAN_TASKS,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
        repair_attempts: int = 1,
        available_tools: Sequence[str] | None = None,
    ) -> None:
        if max_tasks < 1:
            raise ValueError(f"max_tasks must be >= 1, got {max_tasks}")
        if repair_attempts < 0:
            raise ValueError(
                f"repair_attempts must be >= 0, got {repair_attempts}"
            )
        self._provider = provider
        self._max_tasks = max_tasks
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._repair_attempts = repair_attempts
        self._available_tools = (
            tuple(available_tools) if available_tools is not None else None
        )

    async def plan(
        self,
        goal: str,
        analysis: GoalAnalysis,
        *,
        clarifications: dict[str, str] | None = None,
    ) -> Plan:
        """Produce a structurally valid plan for ``goal`` + ``analysis``.

        ``clarifications`` (question -> answer) carries the user's answers
        from an earlier clarification round; they are treated as binding
        constraints on the plan.

        Failure behavior:
        - blank goal -> ``ValueError`` (before any LLM call)
        - malformed / mismatched LLM output -> ``LLMOutputInvalidError``
        - LLM timeout / API failure -> ``LLMError`` / ``LLMTimeoutError``
        - structurally invalid plan after all repair attempts ->
          ``PlanValidationError`` listing every issue
        """
        cleaned = goal.strip()
        if not cleaned:
            raise ValueError("goal must be a non-empty string")

        prompt = self._build_prompt(
            cleaned, analysis, clarifications=clarifications, issues=None
        )
        last_error: PlanValidationError | None = None

        for attempt in range(self._repair_attempts + 1):
            if attempt > 0:
                assert last_error is not None
                prompt = self._build_prompt(
                    cleaned,
                    analysis,
                    clarifications=clarifications,
                    issues=last_error.issues,
                )
            plan = await self._provider.complete(
                prompt,
                Plan,
                system=self._system_prompt(),
                temperature=self._temperature,
                timeout_s=self._timeout_s,
            )
            try:
                validate_plan(
                    plan,
                    analysis=analysis,
                    max_tasks=self._max_tasks,
                    available_tools=self._available_tools,
                )
                return plan
            except PlanValidationError as error:
                last_error = error

        assert last_error is not None
        raise last_error

    def _system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT.format(
            max_tasks=self._max_tasks,
            available_tools=self._tool_list_clause(),
        )

    def _tool_list_clause(self) -> str:
        """Render the registered tool names (or a neutral fallback)."""
        if self._available_tools:
            return ", ".join(repr(name) for name in self._available_tools)
        return "any tool registered in this runtime"

    def _build_prompt(
        self,
        goal: str,
        analysis: GoalAnalysis,
        *,
        clarifications: dict[str, str] | None,
        issues: list[str] | None,
    ) -> str:
        constraint_ids = ", ".join(c.id for c in analysis.constraints) or "none"
        parts = [
            "Plan this user goal:",
            goal,
            "",
            "Analysis summary: " + analysis.reason_summary,
            "Available constraint ids: " + constraint_ids,
        ]
        if self._available_tools:
            parts.append(
                "Registered tools (the only valid tool_hint values): "
                + ", ".join(self._available_tools)
            )
        if clarifications:
            parts += [
                "",
                "User clarifications (binding - honor every answer):",
                *[
                    f"- {question}: {answer}"
                    for question, answer in clarifications.items()
                ],
            ]
        if issues:
            parts += [
                "",
                "Your previous plan failed structural validation:",
                *[f"- {issue}" for issue in issues],
                "Return a corrected plan that resolves every issue.",
            ]
        return "\n".join(parts)
