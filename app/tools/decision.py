"""Deterministic validation of an LLM tool proposal.

Flow (Phase 2 contract, orchestration arrives later):

    LLM -> ToolDecision -> [THIS MODULE] -> ToolCall
        -> ToolRegistry lookup -> input validation -> execution

This module never executes anything and never imports the LLM layer: it
accepts any object matching ``ToolDecisionLike`` (structurally compatible
with the ToolDecision schema defined in the llm package).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from app.state.models import JsonValue, ToolCall
from app.tools.registry import ToolRegistry

__all__ = ["ToolDecisionLike", "resolve_tool_decision"]


class ToolDecisionLike(Protocol):
    """Structural contract of an LLM tool proposal (no LLM import needed)."""

    tool_name: str
    arguments: Mapping[str, JsonValue]
    reason_summary: str


def resolve_tool_decision(
    registry: ToolRegistry, decision: ToolDecisionLike
) -> ToolCall:
    """Validate an LLM tool proposal into an approved ``ToolCall``.

    Deterministic steps only:
    1. registry lookup (``ToolNotFoundError`` for unknown tools)
    2. input-schema validation (``MalformedToolInputError`` on bad arguments)

    Nothing is executed here. The proposal is a *suggestion* until it passes
    both checks.
    """
    tool = registry.get(decision.tool_name)  # typed error if unknown
    validated = tool.validate_input(dict(decision.arguments))
    return ToolCall(
        tool_name=tool.name,
        arguments=validated.model_dump(mode="json"),
        reason_summary=decision.reason_summary[:500],
    )
