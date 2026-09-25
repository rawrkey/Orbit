"""Tool contract and registry tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from app.tools.base import (
    MalformedToolInputError,
    Tool,
    ToolExecutionError,
    ToolSpec,
)
from app.tools.registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
)

ROOT = Path(__file__).resolve().parents[1]


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


class EchoTool(Tool):
    name = "echo"
    description = "Echoes the input text back."
    input_schema = EchoInput
    output_schema = EchoOutput
    timeout_s = 1.5
    max_retries = 2

    def execute(self, arguments: BaseModel) -> BaseModel:
        assert isinstance(arguments, EchoInput)
        return EchoOutput(text=arguments.text)


class BrokenOutputTool(Tool):
    """Violates its own output contract on purpose."""

    name = "broken"
    description = "Returns something that is not EchoOutput."
    input_schema = EchoInput
    output_schema = EchoOutput

    def execute(self, arguments: BaseModel) -> BaseModel:
        return "not a model"  # type: ignore[return-value]


class UnnamedTool(Tool):
    name = ""  # invalid on purpose
    description = "No name."
    input_schema = EchoInput
    output_schema = EchoOutput

    def execute(self, arguments: BaseModel) -> BaseModel:
        return EchoOutput(text="")


# ---------------------------------------------------------------------------
# Contract behavior
# ---------------------------------------------------------------------------


def test_tool_cannot_be_instantiated_without_execute() -> None:
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_tool_validates_input_before_executing() -> None:
    tool = EchoTool()
    result = tool.run({"text": "hello"})
    assert isinstance(result, EchoOutput)
    assert result.text == "hello"


def test_malformed_input_raises_typed_error() -> None:
    tool = EchoTool()
    with pytest.raises(MalformedToolInputError) as exc_info:
        tool.run({"wrong_field": "x"})
    assert exc_info.value.tool_name == "echo"


def test_output_contract_violation_raises_typed_error() -> None:
    tool = BrokenOutputTool()
    with pytest.raises(ToolExecutionError):
        tool.run({"text": "hi"})


def test_tool_spec_describes_the_contract() -> None:
    spec = EchoTool().spec()
    assert isinstance(spec, ToolSpec)
    assert spec.name == "echo"
    assert spec.timeout_s == 1.5
    assert spec.max_retries == 2
    assert "text" in spec.input_schema.get("properties", {})
    assert "text" in spec.output_schema.get("properties", {})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_and_lookup() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    assert registry.has("echo")
    assert registry.get("echo") is tool
    assert registry.names() == ["echo"]
    assert len(registry) == 1


def test_unknown_tool_raises_typed_error() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.get("web_search")
    assert exc_info.value.name == "web_search"
    assert registry.has("web_search") is False


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(DuplicateToolError) as exc_info:
        registry.register(EchoTool())
    assert exc_info.value.name == "echo"
    assert len(registry) == 1


def test_invalid_tool_contracts_are_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolRegistrationError):
        registry.register(UnnamedTool())


def test_list_tools_returns_sorted_serializable_specs() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    specs = registry.list_tools()
    assert [s.name for s in specs] == ["echo"]
    assert all(isinstance(s, ToolSpec) for s in specs)


def test_registry_requires_no_llm() -> None:
    """The tools layer must never import the LLM layer (Phase 2: every file)."""
    tools_dir = ROOT / "app" / "tools"
    files = sorted(tools_dir.glob("*.py"))
    assert files, "expected tool layer source files"
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "app.llm" not in source, f"{path.name} must not import app.llm"
        assert "import llm" not in source, f"{path.name} must not import llm"
