"""Observer tests: execution outcomes -> structured Observations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.observer import MAX_RAW_BYTES, Observer
from app.llm.schemas import TaskResult
from app.state.models import Observation, ObservationSource, ToolResult


@pytest.fixture
def observer() -> Observer:
    return Observer()


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------


def test_successful_tool_result_becomes_ok_observation(observer) -> None:
    obs = observer.observe_tool_result(
        task_id="t1",
        tool_name="calculator",
        result=ToolResult(ok=True, data={"expression": "2+2", "result": 4.0}, duration_ms=1.5),
    )
    assert obs.task_id == "t1"
    assert obs.is_error is False
    assert obs.source is ObservationSource.TOOL
    assert obs.tool_name == "calculator"
    assert obs.error_code is None
    assert obs.raw == {"expression": "2+2", "result": 4.0}
    assert "calculator" in obs.content_summary


def test_successful_tool_result_without_data_gets_generic_summary(observer) -> None:
    obs = observer.observe_tool_result(
        task_id="t1", tool_name="file_writer", result=ToolResult(ok=True, data=None)
    )
    assert obs.is_error is False
    assert "successfully" in obs.content_summary
    assert obs.raw is None


def test_failed_tool_result_becomes_failed_observation(observer) -> None:
    obs = observer.observe_tool_result(
        task_id="t1",
        tool_name="web_search",
        result=ToolResult(ok=False, error="search provider unavailable", error_code="UNAVAILABLE"),
    )
    assert obs.is_error is True
    assert obs.source is ObservationSource.TOOL
    assert obs.error_code == "UNAVAILABLE"
    assert obs.tool_name == "web_search"
    assert "failed" in obs.content_summary
    assert isinstance(obs.raw, dict) and obs.raw["error_code"] == "UNAVAILABLE"


def test_failed_tool_result_never_becomes_ok(observer) -> None:
    obs = observer.observe_tool_result(
        task_id="t1",
        tool_name="calculator",
        result=ToolResult(ok=False, error="boom", error_code="TOOL_FAILURE"),
    )
    assert obs.is_error is True
    assert obs.error_code == "TOOL_FAILURE"


def test_long_tool_error_is_truncated_to_summary_bound(observer) -> None:
    obs = observer.observe_tool_result(
        task_id="t1",
        tool_name="calculator",
        result=ToolResult(ok=False, error="x" * 5000, error_code="TOOL_FAILURE"),
    )
    assert len(obs.content_summary) <= 500


def test_huge_tool_output_is_bounded_in_raw(observer) -> None:
    huge = {"blob": "y" * (MAX_RAW_BYTES * 2)}
    obs = observer.observe_tool_result(
        task_id="t1", tool_name="file_reader", result=ToolResult(ok=True, data=huge)
    )
    assert isinstance(obs.raw, dict)
    assert obs.raw["truncated"] is True
    assert "preview" in obs.raw
    assert len(str(obs.raw)) < MAX_RAW_BYTES


# ---------------------------------------------------------------------------
# Reasoning results
# ---------------------------------------------------------------------------


def test_reasoning_result_becomes_observation(observer) -> None:
    obs = observer.observe_reasoning_result(
        task_id="t2",
        result=TaskResult(reason_summary="Day 1 covers South Mumbai.", data={"days": 3}),
    )
    assert obs.task_id == "t2"
    assert obs.is_error is False
    assert obs.source is ObservationSource.REASONING
    assert obs.content_summary == "Day 1 covers South Mumbai."
    assert obs.raw == {"days": 3}
    assert obs.tool_name is None
    assert obs.error_code is None


def test_reasoning_result_without_data_keeps_raw_none(observer) -> None:
    obs = observer.observe_reasoning_result(
        task_id="t2", result=TaskResult(reason_summary="Summary only.")
    )
    assert obs.raw is None


# ---------------------------------------------------------------------------
# Rejected decisions and typed errors
# ---------------------------------------------------------------------------


def test_rejected_decision_observation(observer) -> None:
    obs = observer.observe_rejected_decision(
        task_id="t1",
        tool_name="ghost_tool",
        detail="unknown tool: ghost_tool",
        error_code="TOOL_NOT_FOUND",
    )
    assert obs.is_error is True
    assert obs.source is ObservationSource.DECISION
    assert obs.tool_name == "ghost_tool"
    assert obs.error_code == "TOOL_NOT_FOUND"
    assert "rejected" in obs.content_summary


def test_system_error_observation(observer) -> None:
    obs = observer.observe_error(
        task_id="t1", detail="provider timed out", error_code="LLM_TIMEOUT"
    )
    assert obs.is_error is True
    assert obs.source is ObservationSource.SYSTEM
    assert obs.error_code == "LLM_TIMEOUT"
    assert obs.content_summary == "provider timed out"


# ---------------------------------------------------------------------------
# Schema / hygiene
# ---------------------------------------------------------------------------


def test_observation_schema_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Observation(task_id="t1", content_summary="x" * 501)
    with pytest.raises(ValidationError):
        Observation(task_id="", content_summary="ok")
    with pytest.raises(ValidationError):
        Observation(task_id="t1", content_summary="")


def test_observation_fields_are_exactly_the_documented_set() -> None:
    assert set(Observation.model_fields) == {
        "task_id",
        "content_summary",
        "raw",
        "is_error",
        "source",
        "tool_name",
        "error_code",
        "task_fingerprint",
        "created_at",
    }


def test_no_prompt_or_chain_of_thought_fields_exist() -> None:
    fields = set(Observation.model_fields)
    for banned in ("prompt", "reasoning", "chain_of_thought", "thoughts", "log"):
        assert banned not in fields


def test_observation_carries_a_timestamp(observer) -> None:
    obs = observer.observe_error(task_id="t1", detail="x", error_code="E")
    assert obs.created_at is not None


def test_observer_is_pure_no_state_mutation(observer) -> None:
    obs = observer.observe_error(task_id="t1", detail="x", error_code="E")
    assert isinstance(obs, Observation)
    assert not hasattr(observer, "_store")
    assert not hasattr(observer, "_controller")
