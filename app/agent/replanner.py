"""Replanner: propose a targeted revised Plan after a failed validation.

Answers "What should ORBIT do differently?" - it returns a *proposed* Plan
and never mutates workflow state. The proposal must pass the same
deterministic structural validation the Planner uses (unique ids, valid
dependencies, no cycles, bounded size, resolvable constraint references)
plus replan-specific rules:

- the proposal must actually DIFFER from the plan it replaces (fingerprint),
  preventing pointless replan loops
- one bounded repair attempt feeds the issues back to the LLM, then a typed
  ``ReplanValidationError`` is raised
- constraints are IMMUTABLE: the analysis the replanner sees is fingerprinted
  before the LLM call and re-checked after it - any attempt to relax a limit,
  operator, objective, or required input raises ``ReplannerError`` before the
  proposal is considered, and unknown constraint ids are rejected

Replan mode: FAIL validation (concrete evidence of a violated constraint)
means a CORRECTIVE replan; INSUFFICIENT validation (missing evidence) means
an EVIDENCE-GATHERING replan that adds lookups/measurements instead of
assuming a violation.

Targeted adaptation: the prompt carries the validation failure, the violated
constraint ids, the observations that caused them, the tasks that already
succeeded, and instructions to REUSE unchanged completed task ids so useful
work is preserved (the Controller keeps their COMPLETED status).
"""

from __future__ import annotations

from collections.abc import Sequence

from app.llm.base import LLMProvider
from app.state.models import (
    GoalAnalysis,
    Plan,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
)
from app.agent.planner import (
    DEFAULT_MAX_PLAN_TASKS,
    PlanValidationError,
    validate_plan,
)

__all__ = [
    "MODE_INSTRUCTIONS",
    "ReplanValidationError",
    "Replanner",
    "ReplannerError",
    "replan_mode",
]


class ReplannerError(Exception):
    """The workflow lacks the structured state replanning requires."""


class ReplanValidationError(PlanValidationError):
    """A replan proposal failed deterministic validation."""


def replan_mode(result: ValidationResult) -> str:
    """Classify the replan from the validation status.

    ``corrective``        -> evidence PROVES a violation: fix the plan.
    ``evidence_gathering``-> evidence is MISSING: gather it, do not assume.
    """
    if result.status is ValidationStatus.FAIL:
        return "corrective"
    return "evidence_gathering"


MODE_INSTRUCTIONS = {
    "corrective": (
        "MODE: corrective. Validation FAIL means concrete evidence PROVES a "
        "constraint was violated. Produce a corrected plan that achieves a "
        "compliant outcome (e.g. alternatives within limits, fixing the "
        "failing step). The violation is factual - do not re-litigate it."
    ),
    "evidence_gathering": (
        "MODE: evidence_gathering. Validation INSUFFICIENT means evidence is "
        "MISSING, not that a constraint was violated. Add or revise tasks "
        "that COLLECT the missing evidence (lookups, measurements, tool "
        "calls). Do not assume a violation and do not fabricate values."
    ),
}


REPLANNER_SYSTEM_PROMPT = """\
You are the replanning stage of ORBIT, an adaptive workflow agent. The
previous plan executed but validation against the goal FAILED or was
INSUFFICIENT. Propose a revised plan as a single JSON object matching the
Plan schema.

Rules:
- Constraints are FIXED by the user. Never relax, weaken, or modify
  constraint limits, operators, types, the objective, required inputs, or
  user requirements. Constraint ids not present in the analysis will be
  rejected - only reference ids from the analysis.
- Be TARGETED: change only what the validation failure requires. Do not
  restart from scratch when parts of the plan already succeeded.
- Preserve completed work: to keep an already-completed task (one that did
  not contribute to the failure), list it with its EXACT same id and
  UNCHANGED description - it will not re-run.
- To redo or redo differently a task, give it a CHANGED description or a
  NEW id - only then will it run again.
- New tasks: fresh unique ids (e.g. "t3"), dependencies only on ids present
  in this plan and listed earlier, status "pending".
- Reference constraint ids from the analysis on tasks that must respect
  them (especially the violated ones).
- If a tool is needed: requires_tool=true plus a tool_hint that is
  EXACTLY one registered tool ({available_tools}) - never null, never an
  invented name; otherwise requires_tool=false and tool_hint null.
- At most {max_tasks} tasks total. success_criteria must be concrete.
- No chain-of-thought. reason text is not part of the Plan schema.
"""


class Replanner:
    """Deterministic wrapper around the replanning LLM call + validation."""

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
            raise ValueError(f"repair_attempts must be >= 0, got {repair_attempts}")
        self._provider = provider
        self._max_tasks = max_tasks
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._repair_attempts = repair_attempts
        self._available_tools = (
            tuple(available_tools) if available_tools is not None else None
        )

    def _tool_list_clause(self) -> str:
        """Render the registered tool names (or a neutral fallback)."""
        if self._available_tools:
            return ", ".join(repr(name) for name in self._available_tools)
        return "any tool registered in this runtime"

    async def propose(
        self, workflow: Workflow, result: ValidationResult
    ) -> Plan:
        """Propose a validated replacement plan for ``workflow``.

        Failure behavior:
        - missing analysis/plan -> ``ReplannerError`` (no LLM call)
        - malformed / mismatched LLM output -> ``LLMOutputInvalidError``
        - structurally invalid or identical proposal after repairs ->
          ``ReplanValidationError`` (a ``PlanValidationError``)
        """
        analysis = workflow.analysis
        current = workflow.plan
        if analysis is None:
            raise ReplannerError(f"workflow {workflow.id}: no goal analysis recorded")
        if current is None:
            raise ReplannerError(f"workflow {workflow.id}: no plan recorded")

        # Constraint immutability: pin the analysis before any LLM call and
        # re-check it before the proposal is considered. A provider (or any
        # code path) that relaxes a limit, operator, objective, or required
        # input during complete() is rejected with no plan proposed.
        analysis_fp = analysis.fingerprint()
        mode = replan_mode(result)
        prompt = self._build_prompt(
            workflow, analysis, current, result, mode=mode, issues=None
        )
        last_error: PlanValidationError | None = None

        for attempt in range(self._repair_attempts + 1):
            if attempt > 0:
                assert last_error is not None
                prompt = self._build_prompt(
                    workflow,
                    analysis,
                    current,
                    result,
                    mode=mode,
                    issues=last_error.issues,
                )
            proposal = await self._provider.complete(
                prompt,
                Plan,
                system=REPLANNER_SYSTEM_PROMPT.format(
                    max_tasks=self._max_tasks,
                    available_tools=self._tool_list_clause(),
                ),
                temperature=self._temperature,
                timeout_s=self._timeout_s,
            )
            if analysis.fingerprint() != analysis_fp:
                raise ReplannerError(
                    f"workflow {workflow.id}: goal analysis changed during "
                    f"replanning - constraint immutability violated"
                )
            try:
                self._validate_proposal(proposal, analysis, current)
                return proposal
            except PlanValidationError as error:
                last_error = error

        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Deterministic proposal validation
    # ------------------------------------------------------------------

    def _validate_proposal(
        self, proposal: Plan, analysis: GoalAnalysis, current: Plan
    ) -> None:
        issues: list[str] = []
        try:
            validate_plan(
                proposal,
                analysis=analysis,
                max_tasks=self._max_tasks,
                available_tools=self._available_tools,
            )
        except PlanValidationError as error:
            issues.extend(error.issues)

        if proposal.fingerprint() == current.fingerprint():
            issues.append(
                "proposal is identical to the current plan; a replan must "
                "change something (redo affected tasks or gather missing evidence)"
            )
        if issues:
            raise ReplanValidationError(issues)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        workflow: Workflow,
        analysis: GoalAnalysis,
        current: Plan,
        result: ValidationResult,
        *,
        mode: str,
        issues: list[str] | None,
    ) -> str:
        import json as _json

        completed = [
            task.id
            for task in current.tasks
            if workflow.task_states.get(task.id) in _COMPLETED_STATES
        ]
        task_views = [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": workflow.task_states.get(task.id, TaskStatus.PENDING).value,
                "depends_on": task.depends_on,
                "requires_tool": task.requires_tool,
                "tool_hint": task.tool_hint,
                "constraint_ids": task.constraint_ids,
                "success_criteria": task.success_criteria,
            }
            for task in current.tasks
        ]
        observation_views = [
            {
                "task_id": observation.task_id,
                "summary": observation.content_summary,
                "is_error": observation.is_error,
                "error_code": observation.error_code,
            }
            for observation in workflow.observations
        ]
        context = {
            "goal": workflow.goal,
            "objective": analysis.objective,
            "constraints": [
                {
                    "id": c.id,
                    "description": c.description,
                    "rule": c.rule.model_dump(mode="json") if c.rule else None,
                }
                for c in analysis.constraints
            ],
            "current_plan": {"version": current.version, "tasks": task_views},
            "completed_task_ids": completed,
            "validation": {
                "status": result.status.value,
                "reason_summary": result.reason_summary,
                "satisfied_constraints": result.satisfied_constraints,
                "violated_constraints": result.violated_constraints,
                "unresolved_constraints": result.unresolved_constraints,
                "evidence": [ref.model_dump(mode="json") for ref in result.evidence],
            },
            "observations": observation_views,
            "replan_mode": mode,
            "replan_count": workflow.replan_count,
            "previous_plan_versions": [
                revision.plan.version for revision in workflow.plan_history
            ],
        }
        parts = [
            "Propose a revised plan for this failed validation.",
            "",
            MODE_INSTRUCTIONS[mode],
            "",
            _json.dumps(context, ensure_ascii=False),
        ]
        if issues:
            parts += [
                "",
                "Your previous proposal failed validation:",
                *[f"- {issue}" for issue in issues],
                "Return a corrected Plan that resolves every issue.",
            ]
        return "\n".join(parts)


_COMPLETED_STATES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.PASSED}
)
