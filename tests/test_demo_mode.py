"""Phase 8.6: explicit offline demo path (``POST /workflows`` with ``demo``).

Root cause these tests pin down: the app wires ONE provider at startup and
``build_provider()`` returns OpenRouter whenever ``OPENROUTER_API_KEY`` is
set, while the Try Demo button created its workflow through the ordinary
``POST /workflows``. The canonical demo goal therefore ran on the real LLM
and failed with HTTP 429 under contest load.

The fix is an explicit, opt-in request flag (``demo: true``). Only that
workflow is routed to the existing scripted demo provider; normal creation
and the real provider-selection rules are untouched - there is no silent
fallback in either direction.

Everything is offline: OpenRouter traffic is intercepted with
``httpx.MockTransport``, LangSearch stays disabled, and the fake key never
leaves this file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import app.config.settings as settings_module
from app.api.app import build_provider, create_app
from app.config.settings import load_settings
from app.demo import DEMO_GOAL
from app.llm.openrouter import OpenRouterProvider
from app.llm.schemas import TaskResult, ValidationProposal
from app.state.models import GoalAnalysis, Plan, SynthesisResult, Task
from app.state.store import InMemoryStateStore

FAKE_KEY = "sk-or-demo-mode-test-do-not-use"
GOAL = "Find options under a budget of 8000 INR"

NORMAL_ANALYSIS = GoalAnalysis(
    objective="Find options under 8000 INR",
    reason_summary="Budget cap is explicit.",
    constraints=[],
)
NORMAL_PLAN = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Draft list",
            description="Draft the options list",
            success_criteria="list drafted",
        )
    ],
)
NORMAL_TASK_RESULT = TaskResult(reason_summary="Options drafted.", data={})
NORMAL_PROPOSAL = ValidationProposal(
    reason_summary="All options within budget.",
    constraint_verdicts=[],
    goal_outcome="satisfied",
)
NORMAL_SYNTHESIS = SynthesisResult(
    final_answer="Three options fit the budget.",
    key_results=["3 options"],
    constraints_satisfied=[],
)


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the environment: OpenRouter configured, search disabled."""
    path = tmp_path / ".env"
    monkeypatch.setattr(settings_module, "_project_env_file", lambda: path)
    for name in (
        "OPENROUTER_API_KEY",
        "WEB_SEARCH_PROVIDER",
        "LANGSEARCH_API_KEY",
        "WEB_SEARCH_API_KEY",
        "LLM_MAX_RETRIES",
        "LLM_BACKOFF_S",
    ):
        monkeypatch.delenv(name, raising=False)
    path.write_text(
        f"OPENROUTER_API_KEY={FAKE_KEY}\n"
        "LLM_MAX_RETRIES=0\n"
        "LLM_BACKOFF_S=0\n",
        encoding="utf-8",
    )
    return path


def install_openrouter_transport(
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Record every httpx.AsyncClient call (i.e. every OpenRouter attempt).

    Returns the live request list: an empty list after a successful run
    proves the configured OpenRouter provider was never contacted.
    """
    hits: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        def handle(request: httpx.Request) -> httpx.Response:
            hits.append(request)
            return respond(request)

        return real_client(transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return hits


def make_client() -> TestClient:
    # provider=None -> build_provider(): with the key configured above this
    # app's real path is OpenRouter, exactly like production.
    app = create_app(store=InMemoryStateStore(), settings=load_settings())
    return TestClient(app)


def create(client: TestClient, payload: dict) -> str:
    response = client.post("/workflows", json=payload)
    assert response.status_code == 201
    return response.json()["workflow_id"]


def rate_limited(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, json={"error": {"message": "rate limited"}})


def openrouter_script() -> dict[str, list[Mapping[str, object]]]:
    """One canned completion per schema for a plain reasoning workflow."""
    return {
        "GoalAnalysis": [NORMAL_ANALYSIS.model_dump(mode="json")],
        "Plan": [NORMAL_PLAN.model_dump(mode="json")],
        "TaskResult": [NORMAL_TASK_RESULT.model_dump(mode="json")],
        "ValidationProposal": [NORMAL_PROPOSAL.model_dump(mode="json")],
        "SynthesisResult": [NORMAL_SYNTHESIS.model_dump(mode="json")],
    }


def scripted_responder(
    script: dict[str, list[Mapping[str, object]]],
) -> Callable[[httpx.Request], httpx.Response]:
    """Emulate OpenRouter chat-completions by routing on the schema name."""

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        routed = (
            payload.get("response_format", {}).get("json_schema", {}).get("name")
        )
        queue = script.get(str(routed))
        if not queue:
            return httpx.Response(
                429,
                json={"error": {"message": f"no scripted response for {routed}"}},
            )
        content = queue.pop(0)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    return respond


# ---------------------------------------------------------------------------
# 1-3. Try Demo: scripted provider selected, no OpenRouter call, full story
# ---------------------------------------------------------------------------


def test_try_demo_selects_the_scripted_provider_with_a_key_configured(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the environment WOULD select OpenRouter for an ordinary workflow ...
    assert isinstance(build_provider(), OpenRouterProvider)

    # ... and any attempt to reach it would be recorded and rejected.
    hits = install_openrouter_transport(monkeypatch, rate_limited)
    client = make_client()

    created = client.post("/workflows", json={"goal": DEMO_GOAL, "demo": True})
    assert created.status_code == 201
    # backward-compatible response shape: no provider details exposed
    assert set(created.json()) == {"workflow_id", "state"}
    workflow_id = created.json()["workflow_id"]

    run = client.post(f"/workflows/{workflow_id}/run", json={})
    assert run.status_code == 200
    body = run.json()

    # the existing scripted adaptive scenario, end to end
    assert body["success"] is True
    assert (body["executions"], body["validations"], body["replans"]) == (2, 2, 1)
    assert body["final_state"] == "COMPLETED"
    assert body["termination"] == "validated"

    # Try Demo performed NO OpenRouter request (configured but unused).
    assert hits == []


def test_try_demo_completes_the_9000_to_7500_adaptive_scenario(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = install_openrouter_transport(monkeypatch, rate_limited)
    client = make_client()
    workflow_id = create(client, {"goal": DEMO_GOAL, "demo": True})

    body = client.post(f"/workflows/{workflow_id}/run", json={}).json()
    assert body["success"] is True

    # 9000 -> validation failure -> corrective replan -> 7500 -> pass
    events = client.get(f"/workflows/{workflow_id}/events").json()["events"]
    types = [event["type"] for event in events]
    assert "validation.failed" in types
    assert "replan.accepted" in types
    assert "validation.passed" in types
    called = [
        event["payload"].get("tool_name")
        for event in events
        if event["type"] == "tool.called"
    ]
    assert called == ["calculator", "calculator"]

    status = client.get(f"/workflows/{workflow_id}").json()
    assert "7,500" in status["final_answer"]
    assert status["replan_count"] == 1
    assert hits == []


# ---------------------------------------------------------------------------
# 4 + 10. Normal creation is unchanged: OpenRouter, ordinary contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [{"goal": GOAL}, {"goal": GOAL, "demo": False}])
def test_normal_workflow_still_uses_openrouter_when_key_configured(
    env_file: Path, monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    hits = install_openrouter_transport(
        monkeypatch, scripted_responder(openrouter_script())
    )
    client = make_client()

    workflow_id = create(client, payload)
    body = client.post(f"/workflows/{workflow_id}/run", json={}).json()

    # a real-LLM workflow completes on the configured provider ...
    assert body["success"] is True
    assert body["termination"] == "validated"
    # ... and never the scripted 9000 -> 7500 story
    assert body["replans"] == 0
    status = client.get(f"/workflows/{workflow_id}").json()
    assert status["final_answer"] == NORMAL_SYNTHESIS.final_answer

    # OpenRouter was selected and called with the configured key
    assert len(hits) == 5
    for request in hits:
        assert str(request.url).endswith("/chat/completions")
        assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


def test_create_keeps_the_ordinary_contract_with_the_flag_present(
    env_file: Path,
) -> None:
    client = make_client()
    response = client.post("/workflows", json={"goal": GOAL, "demo": False})
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"workflow_id", "state"}
    assert len(body["workflow_id"]) == 32
    assert body["state"] == "RECEIVED"


def test_demo_flag_never_bypasses_goal_validation(env_file: Path) -> None:
    client = make_client()
    response = client.post("/workflows", json={"goal": " ", "demo": True})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. OpenRouter unavailable: explicit failure, never a silent demo fallback
# ---------------------------------------------------------------------------


def test_normal_workflow_fails_explicitly_when_openrouter_is_unavailable(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = install_openrouter_transport(monkeypatch, rate_limited)
    client = make_client()
    workflow_id = create(client, {"goal": GOAL})

    body = client.post(f"/workflows/{workflow_id}/run", json={}).json()

    # explicit failure - the scripted budget scenario is never replayed
    assert body["success"] is False
    assert body["termination"] == "analysis_error"
    assert body["final_state"] == "FAILED"
    assert body["replans"] == 0
    assert hits, "OpenRouter was attempted, not bypassed"

    events = client.get(f"/workflows/{workflow_id}/events").text
    assert "calculator" not in events
    assert "9000" not in events
    assert "7500" not in events
