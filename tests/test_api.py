"""API tests: the four endpoints, error mapping, and leak-proof responses.

Runs through FastAPI's TestClient against an in-memory store and the
offline demo/scripted providers - no network, no API key. Responses are
asserted to never contain raw observation payloads, prompts, or keys.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.config.settings import load_settings
from app.demo import create_demo_provider
from app.llm.mock import RoutedMockProvider
from app.llm.schemas import TaskResult, ValidationProposal
from app.state.models import (
    GoalAnalysis,
    Plan,
    SynthesisResult,
    Task,
)
from app.state.store import InMemoryStateStore

GOAL = "Find options under a budget of 8000 INR"
TRAVEL_GOAL = "Plan a trip."

CLARIFY_ANALYSIS = GoalAnalysis(
    objective="Plan a trip with the user's dates",
    reason_summary="Travel dates are missing.",
    required_inputs=["travel dates"],
    clarification_required=True,
)

ANSWER_KEY = "Please provide: travel dates"

ANALYSIS = GoalAnalysis(
    objective="Find options under 8000 INR",
    reason_summary="Budget cap explicit.",
    constraints=[],
)
PLAN = Plan(
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
REASONING = TaskResult(reason_summary="Options drafted.", data={})
PASS_PROPOSAL = ValidationProposal(
    reason_summary="All options within budget.",
    constraint_verdicts=[],
    goal_outcome="satisfied",
)
SYNTH = SynthesisResult(
    final_answer="Three options fit the budget.",
    key_results=["3 options"],
    constraints_satisfied=[],
)


def make_client(provider=None) -> TestClient:
    app = create_app(
        store=InMemoryStateStore(),
        provider=provider or create_demo_provider(),
        settings=load_settings({}),
    )
    return TestClient(app)


def create(client: TestClient, goal: str = GOAL) -> str:
    response = client.post("/workflows", json={"goal": goal})
    assert response.status_code == 201
    return response.json()["workflow_id"]


# ---------------------------------------------------------------------------
# POST /workflows
# ---------------------------------------------------------------------------


def test_create_returns_201_with_id_and_state() -> None:
    client = make_client()
    response = client.post("/workflows", json={"goal": GOAL})
    assert response.status_code == 201
    body = response.json()
    assert len(body["workflow_id"]) == 32
    assert body["state"] == "RECEIVED"


@pytest.mark.parametrize(
    "payload",
    [
        {"goal": ""},
        {"goal": "   "},
        {"goal": 123},
        {},
        {"goal": "x" * 4001},
    ],
)
def test_create_rejects_invalid_goals(payload) -> None:
    client = make_client()
    response = client.post("/workflows", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /workflows/{id}
# ---------------------------------------------------------------------------


def test_status_returns_structured_snapshot() -> None:
    client = make_client()
    workflow_id = create(client)
    response = client.get(f"/workflows/{workflow_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"] == workflow_id
    assert body["state"] == "RECEIVED"
    assert body["task_count"] == 0
    assert body["observation_count"] == 0
    assert body["event_count"] == 1
    assert body["final_answer"] is None
    assert body["clarification_questions"] == []


def test_status_of_unknown_workflow_is_404() -> None:
    client = make_client()
    assert client.get("/workflows/doesnotexist").status_code == 404


def test_status_never_leaks_prompts_or_keys() -> None:
    client = make_client()
    workflow_id = create(client)
    client.post(f"/workflows/{workflow_id}/run", json={})
    text = client.get(f"/workflows/{workflow_id}").text.lower()
    assert "api_key" not in text
    assert "openrouter" not in text
    assert '"raw"' not in text
    assert "observation.raw" not in text
    assert "system prompt" not in text


# ---------------------------------------------------------------------------
# POST /workflows/{id}/run
# ---------------------------------------------------------------------------


def test_run_demo_completes_with_a_replan() -> None:
    client = make_client()
    workflow_id = create(client, "Find options under a budget of 8000.")
    response = client.post(f"/workflows/{workflow_id}/run", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["final_state"] == "COMPLETED"
    assert body["termination"] == "validated"
    assert body["executions"] == 2
    assert body["validations"] == 2
    assert body["replans"] == 1

    status = client.get(f"/workflows/{workflow_id}").json()
    assert status["state"] == "COMPLETED"
    assert status["replan_count"] == 1
    assert status["plan_version"] == 2
    assert "7,500" in status["final_answer"]


def test_run_without_body_is_accepted() -> None:
    client = make_client()
    workflow_id = create(client)
    response = client.post(f"/workflows/{workflow_id}/run")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_run_unknown_workflow_is_404() -> None:
    client = make_client()
    response = client.post("/workflows/nope/run", json={})
    assert response.status_code == 404


def test_rerunning_a_completed_workflow_short_circuits() -> None:
    client = make_client()
    workflow_id = create(client, "Find options under a budget of 8000.")
    client.post(f"/workflows/{workflow_id}/run", json={})
    response = client.post(f"/workflows/{workflow_id}/run", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["termination"] == "already_terminal"
    assert body["success"] is True


def test_run_rejects_unbounded_answers() -> None:
    client = make_client()
    workflow_id = create(client)
    response = client.post(
        f"/workflows/{workflow_id}/run",
        json={"answers": {"q": "x" * 501}},
    )
    assert response.status_code == 422
    response = client.post(
        f"/workflows/{workflow_id}/run",
        json={"answers": {f"q{i}": "a" for i in range(21)}},
    )
    assert response.status_code == 422
    response = client.post(
        f"/workflows/{workflow_id}/run",
        json={"answers": {"": "a"}},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Clarification over HTTP
# ---------------------------------------------------------------------------


def clarifying_client() -> TestClient:
    provider = RoutedMockProvider(
        GoalAnalysis=[CLARIFY_ANALYSIS],
        Plan=[PLAN],
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
        SynthesisResult=[SYNTH],
    )
    return make_client(provider)


def test_run_reports_awaiting_and_status_lists_questions() -> None:
    client = clarifying_client()
    workflow_id = create(client, TRAVEL_GOAL)

    body = client.post(f"/workflows/{workflow_id}/run", json={}).json()
    assert body["termination"] == "awaiting_clarification"
    assert body["final_state"] == "AWAITING_CLARIFICATION"

    status = client.get(f"/workflows/{workflow_id}").json()
    assert status["state"] == "AWAITING_CLARIFICATION"
    assert status["clarification_questions"] == [ANSWER_KEY]


def test_partial_answers_conflict_with_409() -> None:
    client = clarifying_client()
    workflow_id = create(client, TRAVEL_GOAL)
    client.post(f"/workflows/{workflow_id}/run", json={})

    response = client.post(
        f"/workflows/{workflow_id}/run",
        json={"answers": {"wrong question": "Dec 1-4"}},
    )
    assert response.status_code == 409
    assert "missing answers" in response.json()["detail"]


def test_full_answers_complete_the_workflow() -> None:
    client = clarifying_client()
    workflow_id = create(client, TRAVEL_GOAL)
    client.post(f"/workflows/{workflow_id}/run", json={})

    response = client.post(
        f"/workflows/{workflow_id}/run",
        json={"answers": {ANSWER_KEY: "Dec 1-4"}},
    )
    assert response.status_code == 200
    assert response.json()["termination"] == "validated"

    status = client.get(f"/workflows/{workflow_id}").json()
    assert status["state"] == "COMPLETED"
    assert status["final_answer"] == SYNTH.final_answer


# ---------------------------------------------------------------------------
# GET /workflows/{id}/events
# ---------------------------------------------------------------------------


def test_events_endpoint_streams_the_log() -> None:
    client = make_client()
    workflow_id = create(client, "Find options under a budget of 8000.")
    client.post(f"/workflows/{workflow_id}/run", json={})

    response = client.get(f"/workflows/{workflow_id}/events")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["events"])
    assert body["count"] > 10
    assert [e["seq"] for e in body["events"]] == list(range(body["count"]))
    types = [e["type"] for e in body["events"]]
    assert "workflow.started" in types
    assert "replan.accepted" in types
    assert "validation.passed" in types
    assert "synthesis.completed" in types
    assert types[-1] == "workflow.completed"


def test_events_after_returns_only_newer_events() -> None:
    client = make_client()
    workflow_id = create(client, "Find options under a budget of 8000.")
    client.post(f"/workflows/{workflow_id}/run", json={})

    full = client.get(f"/workflows/{workflow_id}/events").json()["count"]
    tail = client.get(
        f"/workflows/{workflow_id}/events?after={full - 4}"
    ).json()
    assert tail["count"] == 3
    assert [e["seq"] for e in tail["events"]] == [full - 3, full - 2, full - 1]


def test_events_rejects_bad_after_and_unknown_workflow() -> None:
    client = make_client()
    workflow_id = create(client)
    assert (
        client.get(f"/workflows/{workflow_id}/events?after=-1").status_code
        == 422
    )
    assert client.get("/workflows/nope/events").status_code == 404


def test_events_payloads_carry_no_raw_observation_data() -> None:
    client = make_client()
    workflow_id = create(client, "Find options under a budget of 8000.")
    client.post(f"/workflows/{workflow_id}/run", json={})
    text = client.get(f"/workflows/{workflow_id}/events").text
    assert '"raw"' not in text
    lowered = text.lower()
    assert "api_key" not in lowered
    assert "system prompt" not in lowered
    # decision summaries exist but not full reasoning traces
    assert "chain of thought" not in lowered


# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------


def test_openapi_document_is_served() -> None:
    client = make_client()
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/workflows" in paths
    assert "/workflows/{workflow_id}" in paths
    assert "/workflows/{workflow_id}/run" in paths
    assert "/workflows/{workflow_id}/events" in paths


def test_responses_serialise_to_pure_json() -> None:
    client = make_client()
    workflow_id = create(client, "Find options under a budget of 8000.")
    client.post(f"/workflows/{workflow_id}/run", json={})
    for path in (
        f"/workflows/{workflow_id}",
        f"/workflows/{workflow_id}/events",
    ):
        payload = json.loads(client.get(path).text)
        assert isinstance(payload, dict)
