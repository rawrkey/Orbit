"""Tool decision flow tests: proposal -> validation -> execution -> ToolResult.

Covers the Phase 2 contract for the future flow:

    LLM -> ToolDecision -> validation -> registry lookup -> input validation
        -> execution -> ToolResult

Orchestration (Executor) is NOT implemented yet; these tests exercise the
deterministic primitives it will use.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.config.settings import load_settings
from app.llm.schemas import ToolDecision
from app.state.models import ToolCall
from app.tools.base import (
    Tool,
    ToolExecutionError,
    ToolSpec,
)
from app.tools.builtin import BUILTIN_TOOL_NAMES, create_default_registry
from app.tools.calculator import CalculatorTool
from app.tools.decision import resolve_tool_decision
from app.tools.execution import execute_tool_call
from app.tools.registry import DuplicateToolError, ToolNotFoundError, ToolRegistry


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = load_settings(
        {"ORBIT_WORKSPACE": str(workspace), "WEB_SEARCH_PROVIDER": "none"}
    )
    return create_default_registry(settings)


# ---------------------------------------------------------------------------
# Resolution: proposal -> approved ToolCall
# ---------------------------------------------------------------------------


def test_resolve_valid_decision(registry: ToolRegistry) -> None:
    decision = ToolDecision(
        tool_name="calculator",
        arguments={"expression": "2+2"},
        reason_summary="Recompute the total.",
    )
    call = resolve_tool_decision(registry, decision)
    assert call.tool_name == "calculator"
    assert call.arguments == {"expression": "2+2"}
    assert call.reason_summary == "Recompute the total."


def test_resolve_unknown_tool_raises(registry: ToolRegistry) -> None:
    decision = ToolDecision(tool_name="no_such_tool", reason_summary="x")
    with pytest.raises(ToolNotFoundError):
        resolve_tool_decision(registry, decision)


def test_resolve_malformed_arguments_raises(registry: ToolRegistry) -> None:
    decision = ToolDecision(
        tool_name="calculator",
        arguments={"wrong": "field"},
        reason_summary="x",
    )
    from app.tools.base import MalformedToolInputError

    with pytest.raises(MalformedToolInputError):
        resolve_tool_decision(registry, decision)


def test_resolve_does_not_execute(registry: ToolRegistry) -> None:
    """Resolving must not have side effects (no files written)."""
    decision = ToolDecision(
        tool_name="file_writer",
        arguments={"path": "should-not-exist.txt", "content": "x"},
        reason_summary="propose only",
    )
    resolve_tool_decision(registry, decision)
    workspace = registry.get("file_writer").workspace  # type: ignore[attr-defined]
    assert not (workspace / "should-not-exist.txt").exists()


# ---------------------------------------------------------------------------
# Execution: approved call -> ToolResult
# ---------------------------------------------------------------------------


def test_execute_calculator_success(registry: ToolRegistry) -> None:
    result = execute_tool_call(
        registry,
        ToolCall(tool_name="calculator", arguments={"expression": "6*7"}),
    )
    assert result.ok is True
    assert result.data == {"expression": "6*7", "result": 42.0}
    assert result.duration_ms >= 0
    assert result.error is None


def test_execute_unknown_tool_returns_failure_result(registry: ToolRegistry) -> None:
    result = execute_tool_call(registry, ToolCall(tool_name="ghost", arguments={}))
    assert result.ok is False
    assert result.error_code == "TOOL_NOT_FOUND"


def test_execute_unsafe_input_is_rejected(registry: ToolRegistry) -> None:
    result = execute_tool_call(
        registry,
        ToolCall(tool_name="calculator", arguments={"expression": "os.system('ls')"}),
    )
    assert result.ok is False
    assert result.error_code == "MALFORMED_INPUT"


def test_execute_missing_required_argument_is_rejected(
    registry: ToolRegistry,
) -> None:
    result = execute_tool_call(registry, ToolCall(tool_name="calculator", arguments={}))
    assert result.ok is False
    assert result.error_code == "MALFORMED_INPUT"


def test_execute_path_traversal_is_rejected(registry: ToolRegistry) -> None:
    result = execute_tool_call(
        registry,
        ToolCall(tool_name="file_reader", arguments={"path": "../../etc/passwd"}),
    )
    assert result.ok is False
    assert result.error_code == "PATH_TRAVERSAL"


def test_execute_unconfigured_web_search_is_unavailable(
    registry: ToolRegistry,
) -> None:
    result = execute_tool_call(
        registry, ToolCall(tool_name="web_search", arguments={"query": "x"})
    )
    assert result.ok is False
    assert result.error_code == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Retry / timeout policy
# ---------------------------------------------------------------------------


class FlagInput(BaseModel):
    value: bool = True


class FlakyTool(Tool):
    """Fails twice, then succeeds. Counts its attempts."""

    name = "flaky"
    description = "fails until attempt 3"
    input_schema = FlagInput
    output_schema = FlagInput
    timeout_s = 2.0
    max_retries = 2
    retry_backoff_s = 0.0

    def __init__(self) -> None:
        self.attempts = 0

    def execute(self, arguments: BaseModel) -> BaseModel:
        assert isinstance(arguments, FlagInput)
        self.attempts += 1
        if self.attempts < 3:
            raise ToolExecutionError("transient failure", tool_name=self.name)
        return arguments


class AlwaysFailsTool(Tool):
    name = "always_fails"
    description = "always raises"
    input_schema = FlagInput
    output_schema = FlagInput
    timeout_s = 1.0
    max_retries = 1
    retry_backoff_s = 0.0

    def __init__(self) -> None:
        self.attempts = 0

    def execute(self, arguments: BaseModel) -> BaseModel:
        self.attempts += 1
        raise ToolExecutionError("permanent failure", tool_name=self.name)


class SlowTool(Tool):
    name = "slow"
    description = "sleeps longer than its timeout"
    input_schema = FlagInput
    output_schema = FlagInput
    timeout_s = 0.05
    max_retries = 0
    retry_backoff_s = 0.0

    def execute(self, arguments: BaseModel) -> BaseModel:
        time.sleep(0.4)
        return arguments


def test_retries_are_bounded_and_can_succeed() -> None:
    registry = ToolRegistry()
    tool = FlakyTool()
    registry.register(tool)
    result = execute_tool_call(registry, ToolCall(tool_name="flaky", arguments={}))
    assert result.ok is True
    assert tool.attempts == 3  # 2 failures + 1 success within max_retries=2


def test_retries_exhausted_returns_failure() -> None:
    registry = ToolRegistry()
    tool = AlwaysFailsTool()
    registry.register(tool)
    result = execute_tool_call(
        registry, ToolCall(tool_name="always_fails", arguments={})
    )
    assert result.ok is False
    assert result.error_code == "TOOL_FAILURE"
    assert tool.attempts == 2  # 1 + max_retries=1


def test_tool_timeout_returns_timeout_code() -> None:
    registry = ToolRegistry()
    registry.register(SlowTool())
    result = execute_tool_call(registry, ToolCall(tool_name="slow", arguments={}))
    assert result.ok is False
    assert result.error_code == "TOOL_TIMEOUT"


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def test_all_builtin_tools_registered(registry: ToolRegistry) -> None:
    assert registry.names() == sorted(BUILTIN_TOOL_NAMES)


def test_registry_lookup_and_specs(registry: ToolRegistry) -> None:
    for name in BUILTIN_TOOL_NAMES:
        tool = registry.get(name)
        assert tool.name == name
        spec = tool.spec()
        assert isinstance(spec, ToolSpec)
        assert spec.timeout_s > 0
        assert spec.max_retries >= 0


def test_registry_unknown_and_duplicate_still_typed() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("nope")
    registry.register(CalculatorTool())
    with pytest.raises(DuplicateToolError):
        registry.register(CalculatorTool())


def test_end_to_end_decision_to_result(registry: ToolRegistry) -> None:
    """The full Phase 2 pipeline without any LLM involvement."""
    decision = ToolDecision(
        tool_name="calculator",
        arguments={"expression": "(8000 - 2000) / 2"},
        reason_summary="Per-person daily budget.",
    )
    call = resolve_tool_decision(registry, decision)
    result = execute_tool_call(registry, call)
    assert result.ok is True
    assert result.data is not None
    assert result.data["result"] == 3000.0  # type: ignore[index]
