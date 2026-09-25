"""LLM contract tests: provider protocol and structured-output validation.

All tests are offline: the MockProvider makes no network calls and needs
no API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.llm.base import LLMError, LLMOutputInvalidError, LLMProvider
from app.llm.mock import LLMCallRecord, MockProvider
from app.llm.schemas import LLMDecision, ToolDecision

ROOT = Path(__file__).resolve().parents[1]


class SampleDecision(BaseModel):
    value: int
    reason_summary: str = Field(min_length=1)


class OtherDecision(BaseModel):
    label: str


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_mock_provider_satisfies_the_protocol() -> None:
    provider = MockProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.model_id == "mock/model"


def test_protocol_requires_structured_complete_method() -> None:
    class NotAProvider:
        def complete(self, prompt: str) -> str:
            return prompt

    assert not isinstance(NotAProvider(), LLMProvider)


# ---------------------------------------------------------------------------
# Structured output behavior
# ---------------------------------------------------------------------------


async def test_returns_validated_schema_instance() -> None:
    provider = MockProvider(
        responses=[SampleDecision(value=7, reason_summary="computed")]
    )
    result = await provider.complete("prompt", schema=SampleDecision)
    assert isinstance(result, SampleDecision)
    assert result.value == 7


async def test_raw_mapping_is_validated_against_schema() -> None:
    provider = MockProvider(responses=[{"value": 3, "reason_summary": "ok"}])
    result = await provider.complete("prompt", schema=SampleDecision)
    assert isinstance(result, SampleDecision)
    assert result.value == 3


async def test_invalid_mapping_raises_structured_output_error() -> None:
    provider = MockProvider(responses=[{"value": "not-a-number",
                                        "reason_summary": "ok"}])
    with pytest.raises(LLMOutputInvalidError):
        await provider.complete("prompt", schema=SampleDecision)


async def test_missing_required_field_raises_structured_output_error() -> None:
    provider = MockProvider(responses=[{"value": 1}])  # missing reason_summary
    with pytest.raises(LLMOutputInvalidError):
        await provider.complete("prompt", schema=SampleDecision)


async def test_mismatched_schema_instance_raises() -> None:
    provider = MockProvider(responses=[OtherDecision(label="wrong schema")])
    with pytest.raises(LLMOutputInvalidError):
        await provider.complete("prompt", schema=SampleDecision)


async def test_unsupported_response_type_raises() -> None:
    provider = MockProvider(responses=["free-form text"])  # type: ignore[list-item]
    with pytest.raises(LLMOutputInvalidError):
        await provider.complete("prompt", schema=SampleDecision)


async def test_exhausted_responses_raise_llm_error() -> None:
    provider = MockProvider()
    with pytest.raises(LLMError):
        await provider.complete("prompt", schema=SampleDecision)


# ---------------------------------------------------------------------------
# Call recording and defaults
# ---------------------------------------------------------------------------


async def test_records_every_call_with_parameters() -> None:
    provider = MockProvider(responses=[{"value": 1, "reason_summary": "r"}])
    await provider.complete(
        "do the thing",
        schema=SampleDecision,
        system="be concise",
        temperature=0.1,
        timeout_s=5.0,
    )
    record = provider.calls[0]
    assert isinstance(record, LLMCallRecord)
    assert record.prompt == "do the thing"
    assert record.schema_name == "SampleDecision"
    assert record.system == "be concise"
    assert record.temperature == 0.1
    assert record.timeout_s == 5.0


async def test_responses_are_consumed_in_order() -> None:
    provider = MockProvider(
        responses=[
            {"value": 1, "reason_summary": "first"},
            {"value": 2, "reason_summary": "second"},
        ]
    )
    first = await provider.complete("p", schema=SampleDecision)
    second = await provider.complete("p", schema=SampleDecision)
    assert first.value == 1
    assert second.value == 2
    with pytest.raises(LLMError):
        await provider.complete("p", schema=SampleDecision)


# ---------------------------------------------------------------------------
# Decision schemas
# ---------------------------------------------------------------------------


def test_decision_schemas_require_short_reason_summaries() -> None:
    with pytest.raises(ValidationError):
        LLMDecision(reason_summary="")
    with pytest.raises(ValidationError):
        ToolDecision(tool_name="calculator", reason_summary="x" * 501)


def test_tool_decision_is_a_typed_proposal() -> None:
    decision = ToolDecision(
        tool_name="calculator",
        arguments={"expression": "8000 / 2"},
        reason_summary="Recompute per-person budget.",
    )
    assert isinstance(decision, LLMDecision)
    assert decision.tool_name == "calculator"


# ---------------------------------------------------------------------------
# Layering rules
# ---------------------------------------------------------------------------


def test_controller_does_not_import_llm_or_tools() -> None:
    """The controller owns state; it must never call an LLM or a tool."""
    for relative in ("app/agent/controller.py", "app/agent/state_machine.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "app.llm" not in source, f"{relative} must not import app.llm"
        assert "app.tools" not in source, f"{relative} must not import app.tools"
