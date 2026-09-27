"""Provider-selection tests: `.env` loading + the demo-fallback gate.

Root cause these tests pin down (frontend run with a real goal):

1. The documented project-root `.env` file was never loaded - `load_settings()`
   read `os.environ` only - so `WEB_SEARCH_PROVIDER=langsearch` and
   `LANGSEARCH_API_KEY` were invisible to the backend.
2. `build_provider()` fell back to the offline scripted provider on a missing
   `OPENROUTER_API_KEY` alone, ignoring web-search configuration, so ANY goal
   was silently replayed as the scripted budget scenario
   (`calculator 9000+0` -> FAIL -> `7500+0` -> PASS) and LangSearch never ran.

Everything here is offline: `httpx.MockTransport` only, no credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import app.config.settings as settings_module
from app.api.app import build_provider, create_app
from app.config.settings import load_settings
from app.llm.mock import RoutedMockProvider
from app.llm.openrouter import OpenRouterProvider
from app.llm.schemas import ToolDecision, ValidationProposal
from app.state.models import GoalAnalysis, Plan, SynthesisResult, Task
from app.state.store import InMemoryStateStore
from app.tools.base import WebSearchUnavailableError
from app.tools.builtin import create_search_provider
from app.tools.search_providers import LangSearchProvider

API_KEY = "langsearch-selection-key-do-not-leak"
GOAL = "Compare 3 gaming laptops under 80,000 INR with 16GB RAM and RTX 4050."
SEARCH_URL = "https://shop.example/asus-f15"

LANGSEARCH_ENV = (
    "WEB_SEARCH_PROVIDER=langsearch\n"
    f"LANGSEARCH_API_KEY={API_KEY}\n"
)


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the process environment and point `.env` at a temp file."""
    path = tmp_path / ".env"
    monkeypatch.setattr(settings_module, "_project_env_file", lambda: path)
    for name in (
        "OPENROUTER_API_KEY",
        "WEB_SEARCH_PROVIDER",
        "LANGSEARCH_API_KEY",
        "WEB_SEARCH_API_KEY",
        "WEB_SEARCH_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    return path


# ---------------------------------------------------------------------------
# 1. `.env` is actually loaded (documented configuration path)
# ---------------------------------------------------------------------------


def test_env_file_is_read_when_loading_the_real_environment(env_file: Path) -> None:
    env_file.write_text("# local config\n" + LANGSEARCH_ENV, encoding="utf-8")

    settings = load_settings()

    assert settings.web_search_provider == "langsearch"
    assert settings.web_search_api_key == API_KEY


def test_real_environment_wins_over_the_env_file(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "none")

    assert load_settings().web_search_provider == "none"


def test_explicit_env_mapping_never_reads_the_env_file(env_file: Path) -> None:
    """Test hermeticity: an explicit mapping must ignore `.env` entirely."""
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")

    settings = load_settings({})

    assert settings.web_search_provider == "none"
    assert settings.web_search_api_key is None


# ---------------------------------------------------------------------------
# 2. langsearch + key -> LangSearchProvider; + missing key -> unavailable
# ---------------------------------------------------------------------------


def test_langsearch_with_key_selects_the_langsearch_provider(
    env_file: Path,
) -> None:
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")

    provider = create_search_provider(load_settings())

    assert isinstance(provider, LangSearchProvider)
    assert provider.name == "langsearch"


def test_langsearch_without_key_is_explicitly_unavailable(
    env_file: Path,
) -> None:
    env_file.write_text("WEB_SEARCH_PROVIDER=langsearch\n", encoding="utf-8")

    settings = load_settings()
    assert create_search_provider(settings) is None

    from app.tools.web_search import WebSearchTool

    tool = WebSearchTool(provider=create_search_provider(settings))
    with pytest.raises(WebSearchUnavailableError) as exc_info:
        tool.run({"query": "anything"})
    assert "WEB_SEARCH_PROVIDER" in str(exc_info.value)
    assert "not configured" in str(exc_info.value)
    assert "LANGSEARCH_API_KEY" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. build_provider(): the scripted demo only when NOTHING is configured
# ---------------------------------------------------------------------------


def test_demo_provider_is_selected_only_when_nothing_is_configured(
    env_file: Path,
) -> None:
    env_file.write_text("# empty on purpose\n", encoding="utf-8")

    provider = build_provider()

    assert provider.model_id.startswith("mock/")


def test_scripted_demo_is_refused_when_langsearch_is_explicitly_configured(
    env_file: Path,
) -> None:
    """The core regression: never silently replay the budget scenario."""
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY") as exc_info:
        build_provider()

    message = str(exc_info.value)
    assert "langsearch" in message
    assert "9000" not in message  # no scripted content involved


def test_app_factory_refuses_to_wire_the_demo_when_search_is_configured(
    env_file: Path,
) -> None:
    """uvicorn startup path: explicit error instead of a scripted workflow."""
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_app()


def test_openrouter_key_selects_the_real_llm_next_to_langsearch(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-only")

    provider = build_provider()

    assert isinstance(provider, OpenRouterProvider)


def test_env_file_can_carry_the_llm_key_too(env_file: Path) -> None:
    env_file.write_text(
        LANGSEARCH_ENV + "OPENROUTER_API_KEY=sk-or-test-only\n", encoding="utf-8"
    )

    assert isinstance(build_provider(), OpenRouterProvider)


# ---------------------------------------------------------------------------
# 4. A normal workflow exercises LangSearch - never the budget scenario
# ---------------------------------------------------------------------------


def scripted_laptop_llm() -> RoutedMockProvider:
    """A real-shaped LLM script for the laptop goal (NOT the demo scenario)."""
    analysis = GoalAnalysis(
        objective="Compare gaming laptops under 80000 INR with 16GB RAM.",
        reason_summary="Budget and specs are explicit.",
        clarification_required=False,
        is_feasible=True,
    )
    plan = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Search",
                description="Search the web for RTX 4050 laptops under 80000 INR",
                requires_tool=True,
                tool_hint="web_search",
                success_criteria="candidate laptops found",
            )
        ],
    )
    return RoutedMockProvider(
        GoalAnalysis=[analysis],
        Plan=[plan],
        ToolDecision=[
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "RTX 4050 16GB laptop under 80000", "max_results": 3},
                reason_summary="Find candidates within budget.",
            )
        ],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="Candidate observed within budget.",
                constraint_verdicts=[],
                goal_outcome="satisfied",
            )
        ],
        SynthesisResult=[
            SynthesisResult(
                final_answer=(
                    "Best match: ASUS TUF F15 at ₹74,990 with 16GB RAM and "
                    f"RTX 4050 - source: {SEARCH_URL}"
                ),
                key_results=["₹74,990", "16GB RAM", "RTX 4050"],
                constraints_satisfied=[],
                caveats=["Prices come from retrieved search snippets."],
                next_actions=["Confirm the price on the source page."],
            )
        ],
    )


async def test_normal_workflow_uses_langsearch_and_never_the_budget_scenario(
    env_file: Path,
) -> None:
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")
    settings = load_settings()

    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        body = json.loads(request.content)
        seen_queries.append(body["query"])
        return httpx.Response(
            200,
            json={
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "ASUS TUF F15 RTX 4050",
                                "url": SEARCH_URL,
                                "snippet": "i5, RTX 4050, 16GB RAM - ₹74,990",
                            }
                        ]
                    }
                }
            },
        )

    # Only the external LangSearch HTTP call is mocked, and only while the
    # real registry constructs the provider (the sanctioned injection seam).
    real_client = httpx.Client
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(handler)
            ),
        )
        app = create_app(
            store=InMemoryStateStore(),
            provider=scripted_laptop_llm(),
            settings=settings,
        )

    controller = app.state.controller
    workflow_id = controller.create_workflow(GOAL).id
    report = await app.state.orchestrator.run(workflow_id)

    # The real search provider ran (mocked HTTP only) ...
    assert report.success is True
    assert seen_queries == ["RTX 4050 16GB laptop under 80000"]
    final = controller.get_workflow(workflow_id)
    tool_observations = [
        obs
        for obs in final.observations
        if obs.source.value == "tool"
    ]
    assert tool_observations, "the search tool must have produced an observation"
    assert tool_observations[0].raw["provider"] == "langsearch"
    assert [r["url"] for r in tool_observations[0].raw["results"]] == [SEARCH_URL]

    # ... and the scripted budget scenario was never replayed.
    events = controller.events(workflow_id)
    blob = json.dumps([event.model_dump(mode="json") for event in events])
    assert "calculator" not in blob
    assert "9000" not in blob
    assert "7500" not in blob
    assert "7,500" not in blob
    called_tools = [
        event.payload.get("tool_name")
        for event in events
        if event.type.value == "tool.called"
    ]
    assert called_tools == ["web_search"]
    assert final.final_answer is not None
    assert SEARCH_URL in final.final_answer
