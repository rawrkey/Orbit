"""Synthesizer: present the user-facing final answer after a PASS.

Answers "What did ORBIT achieve?" - a bounded LLM call that turns the
validated evidence into a concise SynthesisResult (final answer, key
results, constraint claims, caveats, next actions). Guardrails:

- synthesis only runs on a stored PASS validation (``SynthesisError``
  otherwise) - presentation can never precede verification
- constraint claims are cross-checked against the validation's satisfied
  set: claiming anything validation did not confirm raises ``SynthesisError``
- no tools, no state mutation, no chain-of-thought; observation raw payloads
  are size-bounded before they reach the prompt
"""

from __future__ import annotations

import json

from app.llm.base import LLMProvider
from app.state.models import (
    GoalAnalysis,
    SynthesisResult,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
)

__all__ = ["SYNTHESIS_SYSTEM_PROMPT", "SynthesisError", "Synthesizer"]

_RAW_BUDGET = 1500  # per-observation raw JSON character budget


class SynthesisError(Exception):
    """Synthesis was refused: no PASS validation or invalid claims."""


SYNTHESIS_SYSTEM_PROMPT = """\
You are the synthesis stage of ORBIT, an adaptive workflow agent. Execution
finished and validation PASSED. Write the user-facing final answer as a
single JSON object matching the SynthesisResult schema.

Rules:
- constraints_satisfied MUST be a subset of the validation's satisfied
  constraint ids. Never claim a constraint validation did not confirm.
- Use ONLY the evidence summaries provided. No fabrication, no new numbers,
  no chain-of-thought.
- final_answer: one concise user-facing summary (<= 4000 chars) with the
  key numbers from the evidence. key_results: up to 20 short bullets.
- caveats: honest limitations of the evidence (empty list if none).
- next_actions: at most 20 concrete follow-ups the user might take.
- Do not mention prompts, schemas, tools plumbing, or internal IDs.
"""


class Synthesizer:
    """Deterministic wrapper around the synthesis LLM call + post-checks."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._timeout_s = timeout_s

    async def synthesize(
        self, workflow: Workflow, validation: ValidationResult
    ) -> SynthesisResult:
        """Produce a validated ``SynthesisResult`` for ``workflow``.

        Failure behavior:
        - missing analysis/plan or non-PASS validation -> ``SynthesisError``
          (no LLM call)
        - malformed LLM output -> ``LLMOutputInvalidError``
        - claims outside the PASS validation -> ``SynthesisError``
        """
        analysis = workflow.analysis
        if analysis is None:
            raise SynthesisError(
                f"workflow {workflow.id}: no goal analysis recorded"
            )
        if workflow.plan is None:
            raise SynthesisError(f"workflow {workflow.id}: no plan recorded")
        if validation.status is not ValidationStatus.PASS:
            raise SynthesisError(
                f"workflow {workflow.id}: synthesis requires a PASS "
                f"validation, got {validation.status.value}"
            )

        prompt = self._build_prompt(workflow, analysis, validation)
        result = await self._provider.complete(
            prompt,
            SynthesisResult,
            system=SYNTHESIS_SYSTEM_PROMPT,
            temperature=self._temperature,
            timeout_s=self._timeout_s,
        )

        allowed = set(validation.satisfied_constraints)
        overclaimed = [
            constraint_id
            for constraint_id in result.constraints_satisfied
            if constraint_id not in allowed
        ]
        if overclaimed:
            raise SynthesisError(
                f"workflow {workflow.id}: synthesis claimed constraints "
                f"outside the PASS validation: {sorted(overclaimed)}"
            )
        return result

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        workflow: Workflow,
        analysis: GoalAnalysis,
        validation: ValidationResult,
    ) -> str:
        observation_views = []
        for observation in workflow.observations:
            view: dict[str, object] = {
                "task_id": observation.task_id,
                "summary": observation.content_summary,
                "source": observation.source.value,
                "is_error": observation.is_error,
            }
            if observation.raw is not None:
                text = json.dumps(observation.raw, ensure_ascii=False)
                if len(text) <= _RAW_BUDGET:
                    view["raw"] = observation.raw
                else:
                    view["raw_omitted_chars"] = len(text)
            observation_views.append(view)

        context = {
            "goal": workflow.goal,
            "objective": analysis.objective,
            "constraints": [
                {
                    "id": constraint.id,
                    "description": constraint.description,
                    "rule": (
                        constraint.rule.model_dump(mode="json")
                        if constraint.rule
                        else None
                    ),
                }
                for constraint in analysis.constraints
            ],
            "tasks": [
                {
                    "title": task.title,
                    "description": task.description,
                    "success_criteria": task.success_criteria,
                    "status": (
                        (workflow.task_states.get(task.id) or TaskStatus.PENDING)
                        .value
                    ),
                }
                for task in (workflow.plan.tasks if workflow.plan else [])
            ],
            "validation": {
                "status": validation.status.value,
                "reason_summary": validation.reason_summary,
                "satisfied_constraints": validation.satisfied_constraints,
                "evidence": [
                    ref.model_dump(mode="json") for ref in validation.evidence
                ],
            },
            "observations": observation_views,
            "clarifications": workflow.clarifications or None,
        }
        return "\n".join(
            [
                "Write the final answer for this validated workflow.",
                "",
                json.dumps(context, ensure_ascii=False, default=str),
            ]
        )
