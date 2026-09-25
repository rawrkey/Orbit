"""Phase 6 architecture tests: integration invariants that must hold forever.

Static/structural checks complementing the behavioral suites:
- the orchestrator dispatches but never mutates state itself
- the synthesizer is presentation-only (no tools, no controller, no store)
- clarification/synthesis are first-class states/events with bounded schemas
- persistence stores only structured state - no prompts, no keys
- API response schemas can never leak raw observations or prompts
- the demo/offline path stays keyless and offline
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.agent.state_machine import LEGAL_TRANSITIONS
from app.state.events import EventType
from app.state.models import SynthesisResult, WorkflowState

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "app" / "agent"

S = WorkflowState


def imported_modules(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# State machine and events
# ---------------------------------------------------------------------------


def test_clarification_transitions_are_exactly_the_legal_ones() -> None:
    assert S.AWAITING_CLARIFICATION in LEGAL_TRANSITIONS[S.ANALYZING]
    assert LEGAL_TRANSITIONS[S.AWAITING_CLARIFICATION] == {
        S.PLANNING,
        S.FAILED,
    }
    # clarification can never short-circuit to completion
    assert S.COMPLETED not in LEGAL_TRANSITIONS[S.AWAITING_CLARIFICATION]


def test_clarification_and_synthesis_event_types_exist() -> None:
    assert EventType.CLARIFICATION_REQUESTED.value == "clarification.requested"
    assert EventType.CLARIFICATION_ANSWERED.value == "clarification.answered"
    assert EventType.SYNTHESIS_COMPLETED.value == "synthesis.completed"


def test_synthesis_result_has_no_reasoning_fields() -> None:
    assert set(SynthesisResult.model_fields) == {
        "final_answer",
        "key_results",
        "constraints_satisfied",
        "caveats",
        "next_actions",
    }
    forbidden = ("thought", "reasoning", "chain", "cot", "trace")
    for field in SynthesisResult.model_fields:
        assert not any(bad in field for bad in forbidden), field


# ---------------------------------------------------------------------------
# Layering: orchestrator and synthesizer stay in their lanes
# ---------------------------------------------------------------------------


def test_orchestrator_mutates_only_through_the_controller() -> None:
    modules = imported_modules("app/agent/workflow.py")
    assert "app.agent.controller" in modules
    assert not any(
        m.startswith("app.state.store") for m in modules
    ), "orchestrator must not touch the store"
    assert not any(
        m.startswith("app.state.persistent_store") for m in modules
    )
    assert not any(
        m.startswith("app.tools") for m in modules
    ), "orchestrator never calls tools directly"


def test_synthesizer_is_presentation_only() -> None:
    modules = imported_modules("app/agent/synthesizer.py")
    for banned in ("app.tools", "app.agent.controller", "app.state.store"):
        assert not any(
            m == banned or m.startswith(banned + ".") for m in modules
        ), f"synthesizer must not import {banned}"


def test_synthesizer_constructor_takes_only_a_provider() -> None:
    tree = ast.parse(source("app/agent/synthesizer.py"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and node.args.args
            and node.args.args[0].arg == "self"
        ):
            params = [a.arg for a in node.args.args[1:]]
            assert params == ["provider"], params
            return
    raise AssertionError("Synthesizer.__init__ not found")


def test_adaptive_loop_still_works_without_a_synthesizer() -> None:
    """Backward compatibility: synthesis is optional in the loop."""
    import inspect

    from app.agent.adaptive_loop import AdaptiveLoop

    signature = inspect.signature(AdaptiveLoop.__init__)
    synthesizer = signature.parameters["synthesizer"]
    assert synthesizer.default is None  # optional, not required


# ---------------------------------------------------------------------------
# Persistence: structured state only
# ---------------------------------------------------------------------------


def test_persistent_store_schema_stores_only_structured_state() -> None:
    from app.state.persistent_store import _SCHEMA

    # columns are (id, data) and (workflow_id, seq, data) - nothing richer
    assert _SCHEMA.count("TEXT NOT NULL") >= 3
    for banned in ("api_key", "prompt", "thought", "reasoning"):
        assert banned not in _SCHEMA, banned


def test_persistent_store_source_carries_no_key_material() -> None:
    schema = source("app/state/persistent_store.py")
    assert "api_key" not in schema
    assert "system_prompt" not in schema


# ---------------------------------------------------------------------------
# API: response schemas cannot leak
# ---------------------------------------------------------------------------


def test_api_response_schemas_expose_no_raw_or_prompt_fields() -> None:
    from app.api.schemas import (
        EventsResponse,
        RunWorkflowResponse,
        WorkflowStatusResponse,
    )

    banned = ("observations", "raw", "prompt", "reasoning", "api_key")
    for model in (WorkflowStatusResponse, RunWorkflowResponse, EventsResponse):
        for field in model.model_fields:
            assert not any(bad in field for bad in banned), (
                f"{model.__name__}.{field}"
            )


def test_api_source_never_formats_key_or_prompt_material() -> None:
    for path in sorted((ROOT / "app" / "api").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        # the only key reference allowed is reading it from settings
        without_settings = text.replace("openrouter_api_key", "")
        assert "api_key" not in without_settings, path.name
        assert "system_prompt" not in text, path.name


# ---------------------------------------------------------------------------
# Offline / keyless guarantees
# ---------------------------------------------------------------------------


def test_demo_module_is_offline_and_keyless() -> None:
    text = source("app/demo.py")
    for banned in ("httpx", "requests", "urllib", "socket", "api_key"):
        assert banned not in text, banned


def test_build_provider_falls_back_to_the_offline_demo() -> None:
    text = source("app/api/app.py")
    assert "create_demo_provider()" in text
    assert "OpenRouterProvider.from_settings" in text
    # key material is read from settings, never from request input
    assert "settings.openrouter_api_key" in text


def test_demo_provider_answers_are_mock_models() -> None:
    from app.demo import create_demo_provider

    provider = create_demo_provider()
    assert provider.model_id.startswith("mock/")
