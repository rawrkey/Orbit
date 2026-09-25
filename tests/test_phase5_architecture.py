"""Phase 5 architecture tests: invariants the code must keep forever.

Static/structural checks (source + AST), complementing the behavioral tests:
- Controller is the ONLY state-mutation owner (stages never write state)
- No eval/exec anywhere in the agent; no API-key or chain-of-thought fields
- COMPLETED is reachable only through VALIDATING (PASS validation)
- Layering: stages stay in their lanes; no forbidden imports
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.agent.state_machine import LEGAL_TRANSITIONS
from app.state.models import WorkflowState

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "app" / "agent"

# Stage modules: pure propose/validate logic, never allowed to mutate state.
STAGE_MODULES = [
    "validator.py",
    "replanner.py",
    "adaptive_loop.py",
    "synthesizer.py",
    "workflow.py",
]

# Writing to workflow state without going through the controller.
FORBIDDEN_MUTATIONS = [
    r"\.state\s*=",
    r"\.plan\s*=",
    r"\.analysis\s*=",
    r"\.validation\s*=",
    r"\.task_states\s*\[",
    r"\.task_states\s*=",
    r"\.observations\.append",
    r"\.observations\s*=",
    r"\.replan_count\s*[+=]",
    r"\.final_answer\s*=",
    r"\.failure\s*=",
    r"\.plan_history\s*\[",
    r"\.plan_history\s*=",
    r"save_workflow",
]

BANNED_CALLS = ("eval", "exec", "compile", "__import__")
BANNED_IMPORT_FRAGMENTS = ("subprocess", "pickle", "shelve")


def read(name: str) -> str:
    return (AGENT / name).read_text(encoding="utf-8")


def source_files() -> list[Path]:
    return sorted(AGENT.glob("*.py"))


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


# ---------------------------------------------------------------------------
# Controller owns all mutations
# ---------------------------------------------------------------------------


def test_stage_modules_never_write_workflow_state() -> None:
    import re

    for name in STAGE_MODULES:
        source = read(name)
        for pattern in FORBIDDEN_MUTATIONS:
            match = re.search(pattern, source)
            assert match is None, (
                f"{name} contains a state mutation matching {pattern!r}: "
                f"{match.group(0)!r}"
            )


def test_only_controller_persists_workflows() -> None:
    writers = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "app").rglob("*.py")
        if "save_workflow" in path.read_text(encoding="utf-8")
    }
    # Only the Controller writes workflows, via a StateStore implementation
    # (in-memory and SQLite alike) - never from a stage/agent module.
    assert writers == {
        "app/agent/controller.py",
        "app/state/store.py",
        "app/state/persistent_store.py",
    }


# ---------------------------------------------------------------------------
# Banned capabilities
# ---------------------------------------------------------------------------


def test_no_dynamic_code_execution_in_agent_or_state() -> None:
    for path in source_files():
        names = called_names(path)
        banned = names & set(BANNED_CALLS)
        assert not banned, f"{path.name} calls banned builtins: {banned}"
        source = path.read_text(encoding="utf-8")
        for fragment in BANNED_IMPORT_FRAGMENTS:
            assert fragment not in source, f"{path.name} mentions {fragment}"


# A13 hardening: process launch / FFI banned across the ENTIRE app tree.
BANNED_APP_IMPORTS = frozenset(
    {"subprocess", "pty", "ctypes", "commands", "pipes", "popen2"}
)
BANNED_APP_CALLS = frozenset(
    {
        "system",
        "popen",
        "execl",
        "execle",
        "execlp",
        "execv",
        "execve",
        "execvp",
        "spawnl",
        "spawnv",
        "fork",
        "forkpty",
        "posix_spawn",
    }
)
BANNED_APP_FRAGMENTS = (
    "shell=True",
    "os.system(",
    "os.popen(",
    "os.exec",
    "os.spawn",
)


def test_no_process_launch_or_ffi_anywhere_in_app() -> None:
    for path in sorted((ROOT / "app").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        modules = imported_modules(path)
        banned_imports = modules & BANNED_APP_IMPORTS
        assert not banned_imports, (
            f"{path.relative_to(ROOT)} imports banned modules: {banned_imports}"
        )
        names = called_names(path)
        banned_names = names & BANNED_APP_CALLS
        assert not banned_names, (
            f"{path.relative_to(ROOT)} calls banned functions: {banned_names}"
        )
        for fragment in BANNED_APP_FRAGMENTS:
            assert fragment not in source, (
                f"{path.relative_to(ROOT)} contains banned fragment {fragment!r}"
            )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    assert keyword.arg != "shell", (
                        f"{path.relative_to(ROOT)} passes shell= to a call"
                    )


def test_no_api_keys_or_thought_fields_in_state() -> None:
    models = (ROOT / "app" / "state" / "models.py").read_text(encoding="utf-8")
    assert "api_key" not in models
    assert "chain_of_thought" not in models
    assert "thought" not in models.replace("chain-of-thought", "")
    events = (ROOT / "app" / "state" / "events.py").read_text(encoding="utf-8")
    assert "api_key" not in events


def test_no_api_key_references_in_agent_modules() -> None:
    for path in source_files():
        assert "api_key" not in path.read_text(encoding="utf-8"), path.name


def test_stage_system_prompts_forbid_chain_of_thought() -> None:
    for name in ("validator.py", "replanner.py", "synthesizer.py"):
        source = read(name)
        assert "chain-of-thought" in source, f"{name} must ban CoT in prompts"


# ---------------------------------------------------------------------------
# Completion requires a PASS validation
# ---------------------------------------------------------------------------


def test_completed_reachable_only_from_validating() -> None:
    sources = {
        state
        for state, targets in LEGAL_TRANSITIONS.items()
        if WorkflowState.COMPLETED in targets
    }
    assert sources == {WorkflowState.VALIDATING}


def test_execution_stages_cannot_complete_a_workflow() -> None:
    for state in (
        WorkflowState.READY,
        WorkflowState.EXECUTING,
        WorkflowState.AWAITING_VALIDATION,
        WorkflowState.REPLANNING,
        WorkflowState.PLANNING,
    ):
        assert WorkflowState.COMPLETED not in LEGAL_TRANSITIONS[state]


def test_validator_enforces_its_consistency_checker() -> None:
    source = read("validator.py")
    assert "validate_result_consistency(" in source
    assert "InconsistentValidationError" in source


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def test_execution_pipeline_does_not_import_adaptive_stage() -> None:
    banned = ("adaptive_loop", "validator", "replanner", "synthesizer")
    for name in ("executor.py", "observer.py", "planner.py", "goal_analyzer.py"):
        modules = imported_modules(AGENT / name)
        for fragment in banned:
            offenders = [m for m in modules if fragment in m]
            assert not offenders, f"{name} imports {offenders}"


def test_validator_and_replanner_stay_pure() -> None:
    """No tools, no controller, no store: they only read state they're given."""
    for name in ("validator.py", "replanner.py"):
        modules = imported_modules(AGENT / name)
        for fragment in ("app.tools", "app.agent.controller", "app.state.store"):
            assert not any(
                m == fragment or m.startswith(fragment + ".") for m in modules
            ), f"{name} must not import {fragment}"


def test_adaptive_loop_only_mutates_through_controller() -> None:
    modules = imported_modules(AGENT / "adaptive_loop.py")
    assert "app.agent.controller" in modules
    # It never touches the store directly.
    assert not any(m.startswith("app.state.store") for m in modules)


def test_all_agent_modules_import_cleanly() -> None:
    import importlib

    for path in source_files():
        if path.name.startswith("_"):
            continue
        importlib.import_module("app.agent." + path.stem)


def test_state_machine_table_is_complete() -> None:
    for state in WorkflowState:
        assert state in LEGAL_TRANSITIONS, state
