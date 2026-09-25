"""Core data models for ORBIT workflows.

These models are pure data. They contain no business logic and never call
the LLM layer. The Controller owns all behavior and state transitions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# JSON-compatible value types (explicit, not dict[str, Any])
# PEP 695 aliases so recursion is resolved correctly by Pydantic.
# ---------------------------------------------------------------------------

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

# ---------------------------------------------------------------------------
# Finite-valued enums
# ---------------------------------------------------------------------------


class WorkflowState(str, Enum):
    """Explicit workflow state machine states."""

    RECEIVED = "RECEIVED"
    ANALYZING = "ANALYZING"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
    PLANNING = "PLANNING"
    READY = "READY"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VALIDATING = "VALIDATING"
    AWAITING_VALIDATION = "AWAITING_VALIDATION"
    REPLANNING = "REPLANNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskStatus(str, Enum):
    """Lifecycle status of a single task inside a plan.

    Execution (Phase 4) moves PENDING -> RUNNING -> OBSERVED -> COMPLETED.
    Validation (a later phase) moves COMPLETED -> PASSED/FAILED.
    FAILED/SKIPPED are terminal for a task within one execution pass.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    OBSERVED = "OBSERVED"
    COMPLETED = "COMPLETED"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ConstraintType(str, Enum):
    """Kinds of constraints extracted from a user goal."""

    BUDGET = "BUDGET"
    TIME = "TIME"
    COUNT = "COUNT"
    PREFERENCE = "PREFERENCE"
    CUSTOM = "CUSTOM"


class ConstraintOperator(str, Enum):
    """Operators available to a structured constraint rule."""

    LTE = "LTE"
    GTE = "GTE"
    EQ = "EQ"
    MAX = "MAX"
    MIN = "MIN"


class ReplanStrategy(str, Enum):
    """How the replanner chooses to respond to a validation failure."""

    REVISE_PLAN = "REVISE_PLAN"
    ADD_TASK = "ADD_TASK"
    DROP_TASK = "DROP_TASK"
    ABORT = "ABORT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Timezone-aware current time (single definition for consistency)."""
    return datetime.now(timezone.utc)


def _short_id() -> str:
    return uuid4().hex[:12]


def _hash_payload(payload: object) -> str:
    """Stable content hash for fingerprints (sha-256, truncated)."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Goal analysis and constraints
# ---------------------------------------------------------------------------


class ConstraintRule(BaseModel):
    """Structured, machine-checkable part of a constraint."""

    operator: ConstraintOperator
    value: float
    unit: str | None = None


class Constraint(BaseModel):
    """A single constraint extracted from the user's goal.

    Generic by design - no domain-specific subclasses. ``name`` is a short
    machine-friendly label (e.g. "budget"); ``description`` explains it in
    prose; ``rule`` carries the checkable operator/value/unit triple.
    """

    id: str = Field(min_length=1, default_factory=_short_id)
    name: str | None = Field(default=None, max_length=120)
    type: ConstraintType
    description: str = Field(min_length=1)
    rule: ConstraintRule | None = None
    hard: bool = True


class Preference(BaseModel):
    """An explicitly stated user preference (not a hard constraint)."""

    name: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=500)


class GoalAnalysis(BaseModel):
    """LLM-produced interpretation of the user's goal (structured).

    Contains everything the Planner needs so it never reparses raw text:
    - ``objective``: what the user wants achieved
    - ``constraints``: explicit, checkable limits
    - ``preferences``: explicitly stated wishes (softer than constraints)
    - ``required_inputs``: genuinely missing information needed to plan
      (an empty list means planning can proceed)
    - ``optional_inputs``: nice-to-have information (never blocks planning)
    - ``clarification_required``: true ONLY when ``required_inputs`` is
      non-empty - essential questions only, never a questionnaire
    - ``reason_summary``: one short explanation (no chain-of-thought)
    """

    objective: str = Field(min_length=1, max_length=2000)
    constraints: list[Constraint] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    clarification_required: bool = False
    assumptions: list[str] = Field(default_factory=list)
    reason_summary: str = Field(min_length=1, max_length=500)
    is_feasible: bool = True

    def fingerprint(self) -> str:
        """Deterministic hash of the analysis content.

        Used to prove constraint immutability: the analysis that replanning
        receives must be byte-identical to the analysis the user's goal
        produced. Any attempt to relax a limit, operator, objective, or
        required input changes this fingerprint.
        """
        payload = {
            "objective": self.objective,
            "constraints": [
                {
                    "id": c.id,
                    "name": c.name,
                    "type": c.type.value,
                    "description": c.description,
                    "rule": c.rule.model_dump(mode="json") if c.rule else None,
                    "hard": c.hard,
                }
                for c in self.constraints
            ],
            "preferences": [p.model_dump(mode="json") for p in self.preferences],
            "required_inputs": self.required_inputs,
            "optional_inputs": self.optional_inputs,
            "clarification_required": self.clarification_required,
            "assumptions": self.assumptions,
            "is_feasible": self.is_feasible,
        }
        return _hash_payload(payload)


# ---------------------------------------------------------------------------
# Plan and tasks
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """One step of a plan. Data only; execution lives elsewhere.

    Ordering is the position in ``Plan.tasks``; dependencies are declared
    via ``depends_on`` (validated structurally by the planner).
    """

    id: str = Field(min_length=1, default_factory=_short_id)
    title: str | None = Field(default=None, max_length=200)
    description: str = Field(min_length=1)
    tool_hint: str | None = None
    input_hint: str | None = None
    requires_tool: bool = False
    expected_output: str | None = Field(default=None, max_length=500)
    constraint_ids: list[str] = Field(default_factory=list)
    success_criteria: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)

    def definition_fingerprint(self) -> str:
        """Hash of WHAT the task asks for (identity for replan reuse).

        Two tasks with the same definition fingerprint are interchangeable:
        a preserved completion of one still satisfies the other. Changing any
        definitional field changes the fingerprint, which forces a reset.
        List ordering of constraint/dependency ids is irrelevant, so it is
        normalized before hashing.
        """
        payload = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tool_hint": self.tool_hint,
            "input_hint": self.input_hint,
            "requires_tool": self.requires_tool,
            "expected_output": self.expected_output,
            "constraint_ids": sorted(self.constraint_ids),
            "success_criteria": self.success_criteria,
            "depends_on": sorted(self.depends_on),
        }
        return _hash_payload(payload)


class Plan(BaseModel):
    """An ordered set of tasks plus a hash used for replan cycle detection."""

    tasks: list[Task] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    plan_hash: str | None = None

    def fingerprint(self) -> str:
        """Deterministic content hash of the plan's LOGICAL identity.

        Used to detect pointless replans: a proposal identical to the plan it
        replaces must be rejected. Raw task ids are deliberately EXCLUDED so
        that renaming ids cannot make an identical logical plan look
        different (which would reset every task and force a full re-run).
        Dependencies are captured structurally (positional edges), and id
        lists are sorted, so irrelevant ordering never changes the hash.
        """
        index_of = {task.id: i for i, task in enumerate(self.tasks)}
        payload = [
            {
                "title": task.title,
                "description": task.description,
                "success_criteria": task.success_criteria,
                "depends_on": sorted(
                    index_of.get(dep, -1) for dep in task.depends_on
                ),
                "tool_hint": task.tool_hint,
                "input_hint": task.input_hint,
                "requires_tool": task.requires_tool,
                "expected_output": task.expected_output,
                "constraint_ids": sorted(task.constraint_ids),
            }
            for task in self.tasks
        ]
        return _hash_payload(payload)


# ---------------------------------------------------------------------------
# Tool I/O
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """An approved decision to run a specific tool with specific arguments."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    reason_summary: str = Field(default="", max_length=500)


class ToolResult(BaseModel):
    """Raw outcome of executing a tool. Never interpreted in place."""

    ok: bool
    data: JsonValue | None = None
    error: str | None = None
    error_code: str | None = None
    duration_ms: float = Field(default=0.0, ge=0)


class ObservationSource(str, Enum):
    """Where an observation came from (provenance for later validation)."""

    TOOL = "tool"
    REASONING = "reasoning"
    DECISION = "decision"  # a rejected tool proposal
    SYSTEM = "system"  # typed execution/LLM errors


class Observation(BaseModel):
    """Normalized record of what happened while executing one task.

    The Observer answers "what happened?" - never "was the goal satisfied?".
    ``raw`` holds the structured result (tool output or reasoning data),
    bounded in size; chain-of-thought and prompts are never stored.
    """

    task_id: str = Field(min_length=1)
    content_summary: str = Field(min_length=1, max_length=500)
    raw: JsonValue | None = None
    is_error: bool = False
    source: ObservationSource = ObservationSource.SYSTEM
    tool_name: str | None = Field(default=None, max_length=120)
    error_code: str | None = Field(default=None, max_length=64)
    task_fingerprint: str | None = Field(
        default=None,
        description=(
            "definition_fingerprint of the task version that produced this "
            "observation - stamped by the Controller so evidence from an OLD "
            "task definition cannot satisfy a NEW one"
        ),
    )
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Validation and replanning
# ---------------------------------------------------------------------------


class ValidationStatus(str, Enum):
    """Outcome of goal-level validation.

    PASS: every required condition is satisfied by existing evidence.
    FAIL: clear evidence shows a required condition is violated.
    INSUFFICIENT: evidence is missing or ambiguous - never collapsed into PASS.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT"


class EvidenceRef(BaseModel):
    """Lightweight pointer to the observation backing a validation decision.

    References observations by task id instead of duplicating payloads.
    """

    task_id: str = Field(min_length=1)
    note: str = Field(default="", max_length=300)


class ValidationResult(BaseModel):
    """Did the observed execution satisfy the goal? Evidence-based, generic.

    Deterministic constraint checks are authoritative: a constraint with a
    violated entry can never coexist with a PASS status.
    """

    status: ValidationStatus
    reason_summary: str = Field(min_length=1, max_length=500)
    satisfied_constraints: list[str] = Field(default_factory=list)
    violated_constraints: list[str] = Field(default_factory=list)
    unresolved_constraints: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    replanning_required: bool = False


class PlanRevision(BaseModel):
    """One superseded plan plus why it was replaced (append-only history).

    Preserves the full replacement chain: previous plan -> validation result
    that triggered the replan -> reason -> new plan version.
    """

    plan: Plan
    reason: str = Field(min_length=1, max_length=500)
    validation: ValidationResult | None = None
    new_plan_version: int | None = Field(default=None, ge=1)
    replaced_at: datetime = Field(default_factory=utcnow)


class SynthesisResult(BaseModel):
    """Presentation-only final answer produced after a validation PASS.

    The Synthesizer never executes tools, never mutates state, and may only
    run when the stored ValidationResult is PASS. Constraint claims are
    cross-checked against the validation result before storage. No
    chain-of-thought is stored - only concise, user-facing content.
    """

    final_answer: str = Field(min_length=1, max_length=4000)
    key_results: list[str] = Field(default_factory=list, max_length=20)
    constraints_satisfied: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list, max_length=20)
    next_actions: list[str] = Field(default_factory=list, max_length=20)


class Verdict(BaseModel):
    """Result of validating an observation against the goal's constraints."""

    task_id: str = Field(min_length=1)
    passed: bool
    failed_constraint_ids: list[str] = Field(default_factory=list)
    reason_summary: str = Field(min_length=1, max_length=500)


class ReplanDecision(BaseModel):
    """Structured proposal for how to recover from a validation failure."""

    strategy: ReplanStrategy
    revised_plan: Plan | None = None
    reason_summary: str = Field(min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Failure and workflow
# ---------------------------------------------------------------------------


class FailureContext(BaseModel):
    """Structured record of why a workflow failed. Never a silent guess."""

    stage: WorkflowState
    error_code: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    task_id: str | None = None


class Workflow(BaseModel):
    """Complete current state of one workflow run."""

    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    state: WorkflowState = WorkflowState.RECEIVED
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    analysis: GoalAnalysis | None = None
    plan: Plan | None = None
    plan_history: list[PlanRevision] = Field(default_factory=list)
    validation: ValidationResult | None = None
    synthesis: SynthesisResult | None = None
    clarification_questions: list[str] = Field(default_factory=list)
    clarifications: dict[str, str] = Field(default_factory=dict)
    replan_count: int = Field(default=0, ge=0)
    task_states: dict[str, TaskStatus] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    event_count: int = Field(default=0, ge=0)
    final_answer: str | None = None
    failure: FailureContext | None = None
