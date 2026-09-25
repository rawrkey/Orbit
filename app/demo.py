"""Offline demo scenario: one scripted goal exercising the full loop.

Used by the API when no ``OPENROUTER_API_KEY`` is configured and by the
end-to-end tests. Fully deterministic and offline: no network, no key,
no fabrication. The scripted story is:

    analyze (budget cap) -> plan -> calculator says 9000 -> FAIL
    -> corrective replan -> calculator says 7500 -> PASS
    -> synthesis -> COMPLETED
"""

from __future__ import annotations

from app.llm.mock import RoutedMockProvider
from app.llm.schemas import TaskResult, ToolDecision, ValidationProposal
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    GoalAnalysis,
    Plan,
    SynthesisResult,
    Task,
)

__all__ = ["DEMO_GOAL", "create_demo_provider"]

DEMO_GOAL = "Find options under a budget of ₹8,000."

DEMO_ANALYSIS = GoalAnalysis(
    objective="Find stay options under a budget of ₹8,000.",
    reason_summary="Budget cap is explicit; totals must be computed and checked.",
    constraints=[
        Constraint(
            id="c1",
            name="budget",
            type=ConstraintType.BUDGET,
            description="Total under 8000 INR",
            rule=ConstraintRule(
                operator=ConstraintOperator.LTE, value=8000.0, unit="INR"
            ),
        )
    ],
    clarification_required=False,
    is_feasible=True,
)

DEMO_PLAN_A = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Compute option total",
            description="Compute the total cost of the shortlisted options",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="total computed",
        ),
        Task(
            id="t2",
            title="Summarize options",
            description="Summarize the shortlisted options",
            success_criteria="options summarized",
        ),
    ],
)

DEMO_PLAN_B = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Recompute cheaper options",
            description="Recompute the total using cheaper options",
            requires_tool=True,
            tool_hint="calculator",
            constraint_ids=["c1"],
            success_criteria="cheaper total computed",
        ),
        DEMO_PLAN_A.tasks[1],  # preserved as completed work
    ],
)

DEMO_SYNTHESIS = SynthesisResult(
    final_answer=(
        "Found stay options totaling ₹7,500 - within the ₹8,000 budget "
        "with ₹500 to spare."
    ),
    key_results=[
        "Total cost: ₹7,500",
        "Budget cap: ₹8,000",
        "Headroom: ₹500",
    ],
    constraints_satisfied=["c1"],
    caveats=["Priced from calculator-provided estimates; verify before booking."],
    next_actions=["Book the selected option", "Re-check prices before booking"],
)


def _calc(expression: str) -> ToolDecision:
    return ToolDecision(
        tool_name="calculator",
        arguments={"expression": expression},
        reason_summary=f"Evaluate {expression}.",
    )


def _claim(note: str) -> ValidationProposal:
    """Claims the budget is satisfied - Python recomputes from evidence."""
    return ValidationProposal(
        reason_summary="Budget verified from the calculator result.",
        constraint_verdicts=[
            {
                "constraint_id": "c1",
                "outcome": "satisfied",
                "evidence": [{"task_id": "t1", "note": note}],
                "evidence_field": "result",
            }
        ],
        goal_outcome="satisfied",
    )


def create_demo_provider() -> RoutedMockProvider:
    """Schema-routed scripted provider for the demo goal (offline)."""
    return RoutedMockProvider(
        GoalAnalysis=[DEMO_ANALYSIS],
        Plan=[DEMO_PLAN_A, DEMO_PLAN_B],
        ToolDecision=[_calc("9000+0"), _calc("7500+0")],
        TaskResult=[
            TaskResult(
                reason_summary="Options summarized for the user.",
                data={"options": 3},
            )
        ],
        ValidationProposal=[_claim("first total"), _claim("revised total")],
        SynthesisResult=[DEMO_SYNTHESIS],
    )
