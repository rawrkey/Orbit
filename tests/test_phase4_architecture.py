"""Phase 4 architectural guardrails.

Proves the dependency direction and ownership rules hold:
- Executor/Observer do not own state transitions (Controller does)
- Executor never modifies the Plan or invokes later-stage components
- ToolRegistry stays the source of truth for tools
- LLM output can only reach execution through deterministic validation
- No circular imports
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXECUTOR = ROOT / "app" / "agent" / "executor.py"
OBSERVER = ROOT / "app" / "agent" / "observer.py"
CONTROLLER = ROOT / "app" / "agent" / "controller.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(_source(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


# ---------------------------------------------------------------------------
# Ownership: Controller mutates, Executor/Observer do not
# ---------------------------------------------------------------------------


def test_executor_and_observer_do_not_import_the_state_machine() -> None:
    for path in (EXECUTOR, OBSERVER):
        for name in _imports(path):
            assert "state_machine" not in name, (
                f"{path.name} must not own transitions via the state machine"
            )


def test_executor_and_observer_never_assign_workflow_state() -> None:
    for path in (EXECUTOR, OBSERVER):
        assert not re.search(r"\.state\s*=(?!=)", _source(path)), (
            f"{path.name} must not assign workflow state directly"
        )


def test_executor_and_observer_do_not_touch_the_state_store() -> None:
    for path in (EXECUTOR, OBSERVER):
        for name in _imports(path):
            assert "state.store" not in name, (
                f"{path.name} must go through the Controller for persistence"
            )


def test_executor_delegates_every_mutation_to_the_controller() -> None:
    source = _source(EXECUTOR)
    for method in (
        "begin_execution",
        "begin_task",
        "record_observation",
        "complete_task",
        "finish_execution",
        "record_execution_failure",
        "emit_event",
    ):
        assert f"self._controller.{method}" in source, (
            f"executor must use controller.{method}"
        )


# ---------------------------------------------------------------------------
# Later-stage components are not invoked
# ---------------------------------------------------------------------------


def test_executor_does_not_import_replanning_or_validation_components() -> None:
    banned = ("replanner", "validator", "synthesizer")
    for name in _imports(EXECUTOR) + _imports(OBSERVER):
        for bad in banned:
            assert bad not in name.lower(), f"unexpected import: {name}"


def test_executor_does_not_reference_plan_mutation_operations() -> None:
    source = _source(EXECUTOR)
    for pattern in (
        r"plan\.tasks\.append",
        r"plan\.tasks\.remove",
        r"plan\.tasks\.pop",
        r"plan\.tasks\[.*\]\s*=",
        r"\.plan\s*=(?!=)",
    ):
        assert re.search(pattern, source) is None, (
            f"executor must not modify the plan: {pattern}"
        )


# ---------------------------------------------------------------------------
# Tool source of truth + no direct code execution from LLM output
# ---------------------------------------------------------------------------


def test_executor_contains_no_hardcoded_tool_names() -> None:
    source = _source(EXECUTOR)
    for tool_name in ("calculator", "file_reader", "file_writer", "web_search"):
        assert f'"{tool_name}"' not in source
        assert f"'{tool_name}'" not in source


def test_llm_decisions_pass_validation_before_execution() -> None:
    source = _source(EXECUTOR)
    validation = source.find("resolve_tool_decision")
    execution = source.find("execute_tool_call")
    assert validation != -1
    assert execution != -1
    assert validation < execution, (
        "the decision must be validated before any tool executes"
    )


def test_llm_output_types_never_execute_directly() -> None:
    """A ToolDecision is only ever turned into a ToolCall via the registry."""
    source = _source(EXECUTOR)
    # execute_tool_call consumes a resolved ToolCall, never a raw decision.
    assert "execute_tool_call(self._registry, call)" in source
    assert "call = resolve_tool_decision" in source


# ---------------------------------------------------------------------------
# No circular imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "app.state.models",
        "app.state.events",
        "app.state.store",
        "app.llm.schemas",
        "app.tools.registry",
        "app.tools.decision",
        "app.tools.execution",
        "app.agent.state_machine",
        "app.agent.controller",
        "app.agent.observer",
        "app.agent.executor",
    ],
)
def test_module_imports_resolve_in_isolation(module: str) -> None:
    importlib.import_module(module)


def test_controller_still_never_imports_llm_or_tools() -> None:
    for path in (CONTROLLER, ROOT / "app" / "agent" / "state_machine.py"):
        source = _source(path)
        assert "app.llm" not in source
        assert "app.tools" not in source
