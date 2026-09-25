"""Validator: did the observed execution satisfy the user's goal?

Answers "Has the goal been satisfied based on the evidence currently
available?" and nothing else. It never mutates workflow state (the
Controller records results) and never replans.

Pipeline (deterministic-first):

    OBSERVATIONS
         -> LLM locates evidence per constraint (ValidationProposal)
         -> DETERMINISTIC CHECKS (numeric rules recomputed in Python)
         -> DETERMINISTIC CONSISTENCY CHECK
         -> ValidationResult

For every constraint carrying a numeric rule, Python extracts the value
from the referenced observation and applies the operator itself - the LLM's
claimed outcome can never override that computation. Semantic (rule-less)
constraints are interpreted by the LLM, but only against evidence that
really exists; fabricated references are dropped and the claim degrades to
INSUFFICIENT rather than PASS.

Status composition (Python-owned):
    any violated constraint      -> FAIL
    else any unresolved          -> INSUFFICIENT
    else goal not satisfied      -> FAIL
    else goal insufficient       -> INSUFFICIENT
    else                         -> PASS
"""

from __future__ import annotations

import json

from app.llm.base import LLMProvider
from app.llm.schemas import ValidationProposal
from app.state.models import (
    Constraint,
    ConstraintRule,
    EvidenceRef,
    GoalAnalysis,
    JsonValue,
    Observation,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
)

__all__ = [
    "InconsistentValidationError",
    "Validator",
    "ValidatorError",
    "evaluate_rule",
    "extract_number",
    "validate_result_consistency",
]

_NUMERIC = (int, float)


class ValidatorError(Exception):
    """The workflow lacks the structured state validation requires."""


class InconsistentValidationError(Exception):
    """A ValidationResult violated the deterministic consistency rules."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)
        super().__init__("inconsistent validation result: " + "; ".join(self.issues))


# ---------------------------------------------------------------------------
# Deterministic primitives
# ---------------------------------------------------------------------------


def evaluate_rule(value: float, rule: ConstraintRule) -> bool:
    """Apply a ConstraintRule operator to a numeric value. No eval(), ever."""
    epsilon = 1e-9
    operator = rule.operator.value
    if operator in ("LTE", "MAX"):
        return value <= rule.value + epsilon
    if operator in ("GTE", "MIN"):
        return value >= rule.value - epsilon
    if operator == "EQ":
        return abs(value - rule.value) <= epsilon
    raise ValidatorError(f"unsupported constraint operator: {operator}")


def extract_number(raw: JsonValue | None, path: str) -> float | None:
    """Walk a dot path into observation raw data and return a number.

    Returns None (never guesses) when the path is missing, the value is not
    a number, or the input is malformed. Booleans are not numbers.
    """
    if raw is None or not path:
        return None
    node: JsonValue = raw
    for segment in path.split("."):
        if isinstance(node, dict):
            if segment not in node:
                return None
            node = node[segment]
        elif isinstance(node, list):
            try:
                index = int(segment)
            except ValueError:
                return None
            if index < 0 or index >= len(node):
                return None
            node = node[index]
        else:
            return None
    if isinstance(node, bool) or not isinstance(node, _NUMERIC):
        return None
    return float(node)


def validate_result_consistency(
    result: ValidationResult,
    analysis: GoalAnalysis,
    *,
    goal_outcome: str | None = None,
    evidence_available: bool | None = None,
) -> None:
    """Deterministic invariants every result must satisfy.

    Most important: PASS can never coexist with a violated constraint, so a
    deterministic violation (e.g. total=9000 vs budget<=8000) makes PASS
    impossible regardless of what the LLM claimed. PASS is also impossible
    when ``evidence_available`` is known to be False (nothing was observed).
    """
    issues: list[str] = []
    known = {constraint.id for constraint in analysis.constraints}

    satisfied = set(result.satisfied_constraints)
    violated = set(result.violated_constraints)
    unresolved = set(result.unresolved_constraints)

    for label, ids in (
        ("satisfied", satisfied),
        ("violated", violated),
        ("unresolved", unresolved),
    ):
        unknown = ids - known
        if unknown:
            issues.append(f"unknown constraint ids in {label}: {sorted(unknown)}")

    if satisfied & violated:
        issues.append("constraints both satisfied and violated")
    if satisfied & unresolved:
        issues.append("constraints both satisfied and unresolved")
    if violated & unresolved:
        issues.append("constraints both violated and unresolved")

    if result.status is ValidationStatus.PASS:
        if violated or unresolved:
            issues.append(
                "PASS is impossible with violated or unresolved constraints"
            )
        if goal_outcome == "not_satisfied":
            issues.append("PASS is impossible when the goal assessment is not_satisfied")
        if evidence_available is False:
            issues.append("PASS is impossible when no evidence was observed")
    if result.status is ValidationStatus.FAIL:
        if not violated and goal_outcome != "not_satisfied":
            issues.append(
                "FAIL requires a violated constraint or a failing goal assessment"
            )
    if result.status is ValidationStatus.INSUFFICIENT:
        if (
            not unresolved
            and goal_outcome != "insufficient"
            and evidence_available is not False
        ):
            issues.append(
                "INSUFFICIENT requires an unresolved constraint, an "
                "insufficient goal assessment, or missing evidence"
            )

    if issues:
        raise InconsistentValidationError(issues)


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------


def _observations_by_task(workflow: Workflow) -> dict[str, Observation]:
    """Latest valid observation per task.

    - latest-wins: after a replan a task may have run several times and only
      the most recent result reflects current state
    - version safety: an observation stamped with an OLD task definition
      fingerprint is stale evidence and is skipped, so evidence produced by a
      pre-replan definition can never satisfy the redefined task. Observations
      without a stamp (``None`` - e.g. legacy/fixtures) stay usable.
    """
    current = (
        {task.id: task.definition_fingerprint() for task in workflow.plan.tasks}
        if workflow.plan is not None
        else {}
    )
    by_task: dict[str, Observation] = {}
    for observation in workflow.observations:
        stamped = observation.task_fingerprint
        if stamped is not None:
            expected = current.get(observation.task_id)
            if expected is not None and stamped != expected:
                continue  # stale evidence from an older task definition
        by_task[observation.task_id] = observation
    return by_task


def _existing_evidence(
    refs: list[EvidenceRef], observed_tasks: set[str]
) -> list[EvidenceRef]:
    """Drop references to tasks that have no observation (no fabrication)."""
    return [ref for ref in refs if ref.task_id in observed_tasks]


def _determine_constraint_outcome(
    constraint: Constraint,
    outcome: str,
    evidence: list[EvidenceRef],
    evidence_field: str | None,
    by_task: dict[str, Observation],
    observed_tasks: set[str],
) -> tuple[str, list[EvidenceRef]]:
    """Return (outcome, accepted_evidence) with Python owning rule checks."""
    existing = _existing_evidence(evidence, observed_tasks)

    if constraint.rule is None:
        # Semantic constraint: trust the LLM only if it cites real evidence.
        if outcome == "insufficient":
            return "insufficient", existing
        if not existing:
            return "insufficient", []
        return outcome, existing

    # Numeric rule: deterministic evaluation is mandatory and final.
    value: float | None = None
    accepted: list[EvidenceRef] = []
    if evidence_field:
        for ref in existing:
            observation = by_task.get(ref.task_id)
            if observation is None:
                continue
            found = extract_number(observation.raw, evidence_field)
            if found is not None:
                value = found
                accepted.append(ref)
                break
    if value is None:
        # Cannot verify deterministically - never guess, never pass.
        return "insufficient", existing
    satisfied = evaluate_rule(value, constraint.rule)
    return ("satisfied" if satisfied else "violated"), accepted


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM_PROMPT = """\
You are the validation stage of ORBIT, an adaptive workflow agent. Decide
whether the recorded execution satisfies the user's goal, using ONLY the
provided evidence. Respond with a single JSON object matching the
ValidationProposal schema.

Rules:
- For EVERY constraint, emit one constraint_verdict with its constraint_id.
- evidence must reference task ids that actually appear in observations.
  Never invent task ids or values.
- For constraints with a numeric rule (operator/value/unit), you MUST set
  evidence_field to the dot path inside that observation's raw JSON holding
  the value (e.g. "result"). Python will recompute the comparison itself.
- outcome is "satisfied", "violated", or "insufficient". Use "insufficient"
  whenever evidence is missing or ambiguous - absence of evidence is NOT
  satisfaction.
- goal_outcome: "satisfied" only if the goal is fully met by the evidence;
  "not_satisfied" if clearly not met; "insufficient" otherwise.
- reason_summary: one concise sentence. No chain-of-thought.
"""


def _observation_view(observation: Observation) -> dict[str, JsonValue]:
    view: dict[str, JsonValue] = {
        "task_id": observation.task_id,
        "summary": observation.content_summary,
        "source": observation.source.value,
        "is_error": observation.is_error,
    }
    if observation.raw is not None:
        text = json.dumps(observation.raw, ensure_ascii=False)
        if len(text) <= 4096:
            view["raw"] = observation.raw
    if observation.error_code is not None:
        view["error_code"] = observation.error_code
    return view


class Validator:
    """Deterministic-first goal validation over recorded evidence."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._timeout_s = timeout_s

    async def validate(self, workflow: Workflow) -> ValidationResult:
        """Validate ``workflow`` against its analysis and observations.

        Failure behavior:
        - missing analysis/plan -> ``ValidatorError`` (no LLM call)
        - malformed LLM output -> ``LLMOutputInvalidError``
        - LLM timeout / API failure -> ``LLMError`` / ``LLMTimeoutError``
        - internally inconsistent result -> ``InconsistentValidationError``
        """
        analysis = workflow.analysis
        if analysis is None:
            raise ValidatorError(f"workflow {workflow.id}: no goal analysis recorded")
        if workflow.plan is None:
            raise ValidatorError(f"workflow {workflow.id}: no plan recorded")

        proposal = await self._provider.complete(
            self._build_prompt(workflow, analysis),
            ValidationProposal,
            system=VALIDATOR_SYSTEM_PROMPT,
            temperature=self._temperature,
            timeout_s=self._timeout_s,
        )
        result = self._compose(workflow, analysis, proposal)
        validate_result_consistency(
            result,
            analysis,
            goal_outcome=proposal.goal_outcome,
            evidence_available=bool(workflow.observations),
        )
        return result

    # ------------------------------------------------------------------
    # Composition (Python-owned status)
    # ------------------------------------------------------------------

    def _compose(
        self,
        workflow: Workflow,
        analysis: GoalAnalysis,
        proposal: ValidationProposal,
    ) -> ValidationResult:
        by_task = _observations_by_task(workflow)
        observed_tasks = set(by_task)

        verdicts = {}
        for verdict in proposal.constraint_verdicts:
            verdicts.setdefault(verdict.constraint_id, verdict)

        satisfied: list[str] = []
        violated: list[str] = []
        unresolved: list[str] = []
        evidence: list[EvidenceRef] = []

        for constraint in analysis.constraints:
            verdict = verdicts.get(constraint.id)
            if verdict is None:
                unresolved.append(constraint.id)  # no verdict -> no guessing
                continue
            outcome, accepted = _determine_constraint_outcome(
                constraint,
                verdict.outcome,
                verdict.evidence,
                verdict.evidence_field,
                by_task,
                observed_tasks,
            )
            evidence.extend(accepted)
            if outcome == "satisfied":
                satisfied.append(constraint.id)
            elif outcome == "violated":
                violated.append(constraint.id)
            else:
                unresolved.append(constraint.id)

        # Status composition: deterministic facts first.
        if violated:
            status = ValidationStatus.FAIL
        elif unresolved:
            status = ValidationStatus.INSUFFICIENT
        elif proposal.goal_outcome == "not_satisfied":
            status = ValidationStatus.FAIL
        elif proposal.goal_outcome == "insufficient":
            status = ValidationStatus.INSUFFICIENT
        elif not workflow.observations:
            # Nothing was ever observed: a bare "satisfied" claim is not
            # proof. Absence of evidence is never PASS.
            status = ValidationStatus.INSUFFICIENT
        else:
            status = ValidationStatus.PASS

        return ValidationResult(
            status=status,
            reason_summary=proposal.reason_summary[:500],
            satisfied_constraints=satisfied,
            violated_constraints=violated,
            unresolved_constraints=unresolved,
            evidence=_dedupe(evidence)[:20],
            replanning_required=status is not ValidationStatus.PASS,
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, workflow: Workflow, analysis: GoalAnalysis) -> str:
        observations = [_observation_view(o) for o in workflow.observations]
        tasks = [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": workflow.task_states.get(
                    task.id, TaskStatus.PENDING
                ).value,
                "success_criteria": task.success_criteria,
            }
            for task in workflow.plan.tasks
        ]
        context = {
            "goal": workflow.goal,
            "objective": analysis.objective,
            "constraints": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "description": c.description,
                    "rule": c.rule.model_dump(mode="json") if c.rule else None,
                    "hard": c.hard,
                }
                for c in analysis.constraints
            ],
            "tasks": tasks,
            "observations": observations,
            "replan_count": workflow.replan_count,
        }
        return (
            "Validate this workflow's execution against the goal.\n\n"
            + json.dumps(context, ensure_ascii=False)
        )


def _dedupe(refs: list[EvidenceRef]) -> list[EvidenceRef]:
    seen: set[tuple[str, str]] = set()
    unique: list[EvidenceRef] = []
    for ref in refs:
        key = (ref.task_id, ref.note)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique
