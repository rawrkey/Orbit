"""Goal Analyzer: raw user goal -> structured ``GoalAnalysis``.

Responsibilities (LLM proposes, deterministic code disposes):
- send the goal to the ``LLMProvider`` with a structured-output schema
- receive a typed ``GoalAnalysis`` (objective, constraints, preferences,
  required vs optional missing inputs, clarification flag, short summary)
- deterministically verify internal consistency of the analysis

The analyzer NEVER executes tools, NEVER searches the web, NEVER invents
missing facts, and NEVER touches workflow state or the event log - it
returns a validated object and leaves state changes to the controller.
"""

from __future__ import annotations

from app.llm.base import LLMProvider
from app.state.models import Constraint, GoalAnalysis

__all__ = [
    "GoalAnalysisError",
    "GoalAnalyzer",
    "InvalidGoalError",
    "validate_analysis",
]


class InvalidGoalError(Exception):
    """The submitted goal text is empty or unusable."""


class GoalAnalysisError(Exception):
    """The analysis failed deterministic consistency checks."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        summary = "; ".join(issues)
        super().__init__(f"goal analysis invalid: {summary}")


ANALYZER_SYSTEM_PROMPT = """\
You are the goal-analysis stage of ORBIT, an adaptive personal workflow agent.
Interpret the user's goal and respond with a single JSON object matching the
required schema.

Rules:
- objective: one clear sentence describing what the user wants.
- constraints: ONLY explicit limits the user stated. Each gets a short
  machine-friendly name, a type (BUDGET, TIME, COUNT, PREFERENCE, CUSTOM),
  a description, and when numeric a rule with operator (LTE, GTE, EQ, MAX,
  MIN), value, and unit.
- preferences: ONLY explicitly stated wishes that are softer than
  constraints (key/value pairs).
- required_inputs: information that is GENUINELY required to produce a
  useful plan and is missing from the goal. Keep this list minimal - ask
  only essential questions, never a questionnaire.
- optional_inputs: nice-to-have information that does NOT block planning.
- clarification_required: true if AND ONLY IF required_inputs is non-empty.
  If the goal has enough information to produce a useful plan, planning
  must proceed: required_inputs = [] and clarification_required = false.
- assumptions: safe assumptions you made so the plan can proceed.
- reason_summary: one short sentence explaining your interpretation.
- Never include chain-of-thought, step-by-step reasoning, or invented facts.
"""


def validate_analysis(analysis: GoalAnalysis) -> None:
    """Deterministic consistency checks on an LLM-produced analysis.

    Raises ``GoalAnalysisError`` listing every issue. This is NOT the future
    result Validator - it only asks: "is this analysis internally coherent?"
    """
    issues: list[str] = []

    if analysis.clarification_required and not analysis.required_inputs:
        issues.append(
            "clarification_required is true but required_inputs is empty"
        )
    if not analysis.clarification_required and analysis.required_inputs:
        issues.append(
            "required_inputs is non-empty but clarification_required is false"
        )

    seen_ids: set[str] = set()
    for constraint in analysis.constraints:
        if constraint.id in seen_ids:
            issues.append(f"duplicate constraint id: {constraint.id!r}")
        seen_ids.add(constraint.id)
        _check_constraint(constraint, issues)

    if issues:
        raise GoalAnalysisError(issues)


def _check_constraint(constraint: Constraint, issues: list[str]) -> None:
    if not constraint.description.strip():
        issues.append(f"constraint {constraint.id!r} has a blank description")
    if constraint.name is not None and not constraint.name.strip():
        issues.append(f"constraint {constraint.id!r} has a blank name")
    if constraint.rule is not None and constraint.rule.value != constraint.rule.value:
        issues.append(f"constraint {constraint.id!r} has a NaN value")


class GoalAnalyzer:
    """Deterministic wrapper around one structured LLM call."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._timeout_s = timeout_s

    async def analyze(self, goal: str) -> GoalAnalysis:
        """Analyze ``goal`` and return a structurally consistent analysis.

        Failure behavior:
        - blank goal -> ``InvalidGoalError`` (before any LLM call)
        - malformed / mismatched LLM output -> ``LLMOutputInvalidError``
          (provider-internal, at most one repair attempt)
        - LLM timeout / API failure -> ``LLMError`` / ``LLMTimeoutError``
        - internally inconsistent analysis -> ``GoalAnalysisError``
        """
        cleaned = goal.strip()
        if not cleaned:
            raise InvalidGoalError("goal must be a non-empty string")

        prompt = f"Analyze this user goal:\n\n{cleaned}"
        analysis = await self._provider.complete(
            prompt,
            GoalAnalysis,
            system=ANALYZER_SYSTEM_PROMPT,
            temperature=self._temperature,
            timeout_s=self._timeout_s,
        )
        validate_analysis(analysis)
        return analysis
