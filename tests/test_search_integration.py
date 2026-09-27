"""Real web-search integration: the full deterministic tool path + research.

Only the EXTERNAL HTTP search is mocked (``httpx.MockTransport``). Everything
else is the real ORBIT machinery: the scripted mock LLM proposes decisions,
Python validates them through the registry, the registered ``WebSearchTool``
runs the real ``LangSearchProvider``, the Observer converts the ``ToolResult``
into workflow state, and the controller records the append-only events.

Covers:
- Step 9: decision -> validation -> registry -> tool -> provider -> results
  -> ToolResult -> Observer -> workflow state/events
- Step 10: an agentic research run ("Compare 3 gaming laptops under 80,000
  with 16GB RAM and RTX 4050") that searches, validates, fails, adapts,
  searches again, then synthesizes - all through the real stages
- Step 17: the offline scripted demo still runs with no search credentials
  and never touches the network

Fully offline: no credentials, no internet, no fabricated search results.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.agent.controller import WorkflowController
from app.agent.executor import Executor
from app.agent.goal_analyzer import GoalAnalyzer
from app.agent.planner import Planner
from app.agent.replanner import Replanner
from app.agent.synthesizer import Synthesizer
from app.agent.validator import Validator
from app.agent.workflow import WorkflowOrchestrator
from app.config.settings import load_settings
from app.demo import DEMO_GOAL, DEMO_SYNTHESIS, create_demo_provider
from app.llm.mock import MockProvider, RoutedMockProvider
from app.llm.schemas import TaskResult, ToolDecision, ValidationProposal
from app.state.events import EventType
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    GoalAnalysis,
    ObservationSource,
    Plan,
    SynthesisResult,
    Task,
    WorkflowState,
)
from app.state.store import InMemoryStateStore
from app.tools.builtin import create_default_registry

S = WorkflowState
E = EventType
API_KEY = "langsearch-integration-key-ABC123-DO-NOT-LEAK"

# ---------------------------------------------------------------------------
# Realistic provider responses (what the mocked LangSearch HTTP endpoint
# returns under data.webPages.value)
# ---------------------------------------------------------------------------

OVER_BUDGET_RESULTS = [
    {
        "name": "Acer Nitro V15 RTX 4050 gaming laptop",
        "url": "https://shop.example/nitro-v15",
        "snippet": "Core i5, RTX 4050, 16GB RAM, 512GB SSD, 15.6in 144Hz - ₹84,999",
    },
    {
        "name": "Lenovo LOQ 15 RTX 4050",
        "url": "https://store.example/loq-15",
        "snippet": "Ryzen 7, RTX 4050, 16GB RAM, 1TB SSD - ₹86,490",
    },
    {
        "name": "HP Victus 16 RTX 4050",
        "url": "https://market.example/victus-16",
        "snippet": "Core i5, RTX 4050, 16GB RAM, 512GB SSD - ₹82,990",
    },
]

UNDER_BUDGET_RESULTS = [
    {
        "name": "ASUS TUF F15 RTX 4050 under 80000",
        "url": "https://shop.example/asus-f15",
        "snippet": "Core i5, RTX 4050, 16GB RAM, 512GB SSD, 144Hz - ₹74,990",
    },
    {
        "name": "Acer Nitro V14 RTX 4050 deal",
        "url": "https://store.example/nitro-v14",
        "snippet": "Ryzen 5, RTX 4050, 16GB RAM, 512GB SSD - ₹77,499",
    },
    {
        "name": "Lenovo IdeaPad Gaming RTX 4050",
        "url": "https://market.example/ideapad-rtx4050",
        "snippet": "Core i5, RTX 4050, 16GB RAM, 1TB SSD - ₹79,500",
    },
]


def make_search_client(seen_queries: list[str]) -> httpx.Client:
    """Mocked LangSearch endpoint: refined queries get under-budget results."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        body = json.loads(request.content)
        query = body["query"]
        seen_queries.append(query)
        results = UNDER_BUDGET_RESULTS if "India" in query else OVER_BUDGET_RESULTS
        return httpx.Response(200, json={"data": {"webPages": {"value": results}}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def langsearch_settings(workspace: Path, **extra: str):
    env = {
        "ORBIT_WORKSPACE": str(workspace),
        "WEB_SEARCH_PROVIDER": "langsearch",
        "LANGSEARCH_API_KEY": API_KEY,
    }
    env.update(extra)
    return load_settings(env)


def ready_workflow(controller: WorkflowController, plan: Plan, analysis) -> str:
    workflow = controller.create_workflow("Find gaming laptops under 80000 INR.")
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, analysis)
    controller.record_plan(workflow.id, plan)
    return workflow.id


SEARCH_ANALYSIS = GoalAnalysis(
    objective="Find gaming laptops under 80000 INR.",
    reason_summary="Budget cap is explicit.",
    constraints=[
        Constraint(
            id="c1",
            name="budget",
            type=ConstraintType.BUDGET,
            description="Price under 80000 INR",
            rule=ConstraintRule(
                operator=ConstraintOperator.LTE, value=80000.0, unit="INR"
            ),
        )
    ],
)


# ---------------------------------------------------------------------------
# Step 9: LLM decision -> deterministic validation -> real tool path
# ---------------------------------------------------------------------------


async def test_llm_decision_flows_through_the_real_search_tool_path(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    seen_queries: list[str] = []
    settings = langsearch_settings(workspace)
    registry = create_default_registry(
        settings, search_client=make_search_client(seen_queries)
    )

    controller = WorkflowController(InMemoryStateStore(), max_replans=2)
    plan = Plan(
        tasks=[
            Task(
                id="t1",
                title="Search the web",
                description="Search the web for RTX 4050 laptops under 80000 INR",
                requires_tool=True,
                tool_hint="web_search",
                constraint_ids=["c1"],
                success_criteria="search results observed",
            )
        ]
    )
    workflow_id = ready_workflow(controller, plan, SEARCH_ANALYSIS)

    query = "RTX 4050 gaming laptop 16GB under 80000"
    llm = MockProvider(
        [
            ToolDecision(
                tool_name="web_search",
                arguments={"query": query, "max_results": 3},
                reason_summary="Find candidate laptops within budget.",
            )
        ]
    )
    report = await Executor(controller, llm, registry).execute(workflow_id)

    # Execution -> observation -> AWAITING_VALIDATION (never straight to done)
    assert report.success is True
    final = controller.get_workflow(workflow_id)
    assert final.state is S.AWAITING_VALIDATION
    assert final.task_states["t1"].value == "COMPLETED"

    # Observer converted the ToolResult into normalized workflow state
    observation = final.observations[0]
    assert observation.source is ObservationSource.TOOL
    assert observation.tool_name == "web_search"
    assert observation.is_error is False
    assert observation.raw is not None
    raw = observation.raw
    assert raw["provider"] == "langsearch"
    assert raw["query"] == query
    assert [r["url"] for r in raw["results"]] == [
        r["url"] for r in OVER_BUDGET_RESULTS
    ]
    for result in raw["results"]:
        assert set(result) == {"title", "url", "snippet", "source"}
        assert result["source"] in ("shop.example", "store.example", "market.example")

    # The provider really was called (mocked HTTP), exactly once, with the query
    assert seen_queries == [query]

    # Event log tells the deterministic story
    events = controller.events(workflow_id)
    types = [event.type for event in events]
    for expected in (
        E.TOOL_DECISION_PROPOSED,
        E.TOOL_CALLED,
        E.TOOL_COMPLETED,
        E.TASK_COMPLETED,
    ):
        assert expected in types, expected

    called = next(event for event in events if event.type is E.TOOL_CALLED)
    assert called.payload["tool_name"] == "web_search"
    assert query in str(called.payload["arguments_preview"])

    # Credentials never reach events or state
    blob = json.dumps([event.model_dump(mode="json") for event in events])
    assert API_KEY not in blob
    assert API_KEY not in json.dumps(final.model_dump(mode="json"), default=str)


async def test_provider_failure_becomes_a_typed_observation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="search backend unavailable")

    settings = langsearch_settings(workspace)
    registry = create_default_registry(
        settings, search_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    controller = WorkflowController(InMemoryStateStore(), max_replans=2)
    plan = Plan(
        tasks=[
            Task(
                id="t1",
                description="Search the web",
                requires_tool=True,
                tool_hint="web_search",
                success_criteria="results observed",
            )
        ]
    )
    workflow_id = ready_workflow(controller, plan, SEARCH_ANALYSIS)
    llm = MockProvider(
        [
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "anything"},
                reason_summary="Search.",
            )
        ]
    )

    report = await Executor(controller, llm, registry).execute(workflow_id)

    assert report.success is False
    assert report.failure is not None
    assert report.failure.error_code == "SEARCH_PROVIDER_ERROR"
    final = controller.get_workflow(workflow_id)
    assert final.state is S.FAILED
    observation = final.observations[0]
    assert observation.is_error is True
    assert observation.error_code == "SEARCH_PROVIDER_ERROR"
    assert "HTTP 503" in observation.content_summary
    assert API_KEY not in observation.content_summary
    # bounded retries at both layers (1 tool retry x 2 provider attempts)
    assert 2 <= len(calls) <= 4
    assert E.TOOL_FAILED in [event.type for event in controller.events(workflow_id)]


# ---------------------------------------------------------------------------
# Step 10: agentic research - search, fail validation, adapt, search again
# ---------------------------------------------------------------------------

RESEARCH_GOAL = (
    "Compare 3 gaming laptops under ₹80,000 with 16GB RAM and RTX 4050."
)

RESEARCH_ANALYSIS = GoalAnalysis(
    objective=(
        "Compare 3 gaming laptops under 80000 INR with 16GB RAM and RTX 4050."
    ),
    reason_summary="Budget, RAM and GPU requirements are explicit.",
    constraints=[
        Constraint(
            id="c1",
            name="budget",
            type=ConstraintType.BUDGET,
            description="Price under 80000 INR",
            rule=ConstraintRule(
                operator=ConstraintOperator.LTE, value=80000.0, unit="INR"
            ),
        ),
        Constraint(
            id="c2",
            name="ram",
            type=ConstraintType.CUSTOM,
            description="At least 16 GB RAM",
            rule=ConstraintRule(
                operator=ConstraintOperator.GTE, value=16.0, unit="GB"
            ),
        ),
        Constraint(
            id="c3",
            name="gpu",
            type=ConstraintType.PREFERENCE,
            description="NVIDIA RTX 4050 GPU",
        ),
    ],
    clarification_required=False,
    is_feasible=True,
)

PLAN_A = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Search",
            description=(
                "Search the web for RTX 4050 gaming laptops with 16GB RAM "
                "priced under 80000 INR"
            ),
            requires_tool=True,
            tool_hint="web_search",
            constraint_ids=["c1", "c2", "c3"],
            success_criteria="candidate laptops found",
        ),
        Task(
            id="t2",
            title="Extract",
            description="Extract price, RAM and GPU from the observed results",
            constraint_ids=["c1", "c2", "c3"],
            depends_on=["t1"],
            success_criteria="specs extracted from results",
        ),
    ],
)

PLAN_B = Plan(
    version=1,
    tasks=[
        Task(
            id="t1",
            title="Search",
            description=(
                "Search again with a refined query that filters RTX 4050 "
                "16GB laptops in India to under 80000 INR"
            ),
            requires_tool=True,
            tool_hint="web_search",
            constraint_ids=["c1", "c2", "c3"],
            success_criteria="refined candidates found",
        ),
        Task(
            id="t2",
            title="Extract",
            description=(
                "Re-extract price, RAM and GPU from the refined search results"
            ),
            constraint_ids=["c1", "c2", "c3"],
            depends_on=["t1"],
            success_criteria="refined specs extracted from results",
        ),
    ],
)

FIRST_QUERY = "RTX 4050 gaming laptop 16GB under 80000"
REVISED_QUERY = "RTX 4050 16GB gaming laptop India under 80000 price"


def search_decision(query: str) -> ToolDecision:
    return ToolDecision(
        tool_name="web_search",
        arguments={"query": query, "max_results": 3},
        reason_summary=f"Search the web for {query}.",
    )


def extraction(lowest_price: float, note: str) -> TaskResult:
    return TaskResult(
        reason_summary=note,
        data={
            "lowest_price": lowest_price,
            "ram_gb": 16,
            "gpu": "RTX 4050",
            "sources": 3,
        },
    )


def budget_verdict(outcome: str, note: str) -> dict:
    return {
        "constraint_id": "c1",
        "outcome": outcome,
        "evidence": [{"task_id": "t2", "note": note}],
        "evidence_field": "lowest_price",
    }


def ram_verdict() -> dict:
    return {
        "constraint_id": "c2",
        "outcome": "satisfied",
        "evidence": [{"task_id": "t2", "note": "16 GB RAM observed"}],
        "evidence_field": "ram_gb",
    }


def gpu_verdict() -> dict:
    return {
        "constraint_id": "c3",
        "outcome": "satisfied",
        "evidence": [{"task_id": "t2", "note": "RTX 4050 observed in results"}],
    }


def research_provider() -> RoutedMockProvider:
    return RoutedMockProvider(
        GoalAnalysis=[RESEARCH_ANALYSIS],
        Plan=[PLAN_A, PLAN_B],
        ToolDecision=[search_decision(FIRST_QUERY), search_decision(REVISED_QUERY)],
        TaskResult=[
            extraction(84999.0, "Lowest observed price is 84999 INR."),
            extraction(74999.0, "Lowest observed price is 74999 INR."),
        ],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="Observed price exceeds the 80000 INR budget.",
                constraint_verdicts=[
                    budget_verdict("violated", "lowest observed price 84999 INR"),
                    ram_verdict(),
                    gpu_verdict(),
                ],
                goal_outcome="not_satisfied",
            ),
            ValidationProposal(
                reason_summary="All three requirements are met by the results.",
                constraint_verdicts=[
                    budget_verdict("satisfied", "lowest observed price 74999 INR"),
                    ram_verdict(),
                    gpu_verdict(),
                ],
                goal_outcome="satisfied",
            ),
        ],
        SynthesisResult=[
            SynthesisResult(
                final_answer=(
                    "3 gaming laptops with RTX 4050 and 16GB RAM under ₹80,000: "
                    "ASUS TUF F15 at ₹74,990 (https://shop.example/asus-f15), "
                    "Acer Nitro V14 at ₹77,499 (https://store.example/nitro-v14) "
                    "and Lenovo IdeaPad Gaming at ₹79,500 "
                    "(https://market.example/ideapad-rtx4050). Prices and specs "
                    "come from the retrieved search results."
                ),
                key_results=[
                    "Lowest price: ₹74,990",
                    "RAM: 16GB on all three",
                    "GPU: RTX 4050 on all three",
                ],
                constraints_satisfied=["c1", "c2", "c3"],
                caveats=[
                    "Prices and specifications come from retrieved search "
                    "result snippets - confirm them on the source pages."
                ],
                next_actions=[
                    "Open the source links to confirm the current price",
                    "Compare warranty and display refresh rates before buying",
                ],
            )
        ],
    )


def research_orchestrator(controller, provider, registry):
    return WorkflowOrchestrator(
        controller,
        GoalAnalyzer(provider),
        Planner(provider),
        Executor(controller, provider, registry),
        Validator(provider),
        Replanner(provider),
        synthesizer=Synthesizer(provider),
    )


async def test_research_goal_searches_validates_adapts_and_completes(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    seen_queries: list[str] = []
    registry = create_default_registry(
        langsearch_settings(workspace), search_client=make_search_client(seen_queries)
    )
    controller = WorkflowController(InMemoryStateStore(), max_replans=3)
    provider = research_provider()
    orchestrator = research_orchestrator(controller, provider, registry)

    workflow = controller.create_workflow(RESEARCH_GOAL)
    report = await orchestrator.run(workflow.id)

    # --- 9: reached synthesis/completion after one adaptation -------------
    assert report.success is True
    assert report.termination == "validated"
    assert (report.executions, report.validations, report.replans) == (2, 2, 1)
    final = controller.get_workflow(workflow.id)
    assert final.state is S.COMPLETED
    assert final.plan.version == 2
    assert final.validation is not None
    assert final.validation.status.value == "PASS"
    assert final.synthesis is not None
    assert final.final_answer == final.synthesis.final_answer

    # --- 4 + 8: two real (mocked-HTTP) searches, with a revised strategy --
    assert seen_queries == [FIRST_QUERY, REVISED_QUERY]
    assert seen_queries[0] != seen_queries[1]

    # --- 5: observations carry the normalized provider results ------------
    tool_observations = [
        observation
        for observation in final.observations
        if observation.source is ObservationSource.TOOL
    ]
    assert len(tool_observations) == 2
    first_urls = [r["url"] for r in tool_observations[0].raw["results"]]  # type: ignore[index]
    second_urls = [r["url"] for r in tool_observations[1].raw["results"]]  # type: ignore[index]
    assert first_urls == [r["url"] for r in OVER_BUDGET_RESULTS]
    assert second_urls == [r["url"] for r in UNDER_BUDGET_RESULTS]

    reasoning_observations = [
        observation
        for observation in final.observations
        if observation.source is ObservationSource.REASONING
    ]
    assert [obs.raw["lowest_price"] for obs in reasoning_observations] == [  # type: ignore[index]
        84999.0,
        74999.0,
    ]

    # --- 6 + 7: validation failed deterministically, then passed ----------
    assert final.plan_history, "the first plan must be preserved in history"
    first_validation = final.plan_history[0].validation
    assert first_validation is not None
    assert first_validation.status.value == "FAIL"
    assert first_validation.violated_constraints == ["c1"]
    assert first_validation.satisfied_constraints == ["c2", "c3"]
    assert final.validation.satisfied_constraints == ["c1", "c2", "c3"]

    # --- event story: search -> validate -> adapt -> search -> synthesize --
    events = controller.events(workflow.id)
    types = [event.type for event in events]
    for expected in (
        E.GOAL_ANALYZED,
        E.WORKFLOW_PLANNED,
        E.TOOL_DECISION_PROPOSED,
        E.TOOL_CALLED,
        E.TOOL_COMPLETED,
        E.VALIDATION_FAILED,
        E.REPLAN_ACCEPTED,
        E.VALIDATION_PASSED,
        E.SYNTHESIS_COMPLETED,
        E.WORKFLOW_COMPLETED,
    ):
        assert expected in types, expected

    first_search = types.index(E.TOOL_CALLED)
    failed = types.index(E.VALIDATION_FAILED)
    adapted = types.index(E.REPLAN_ACCEPTED)
    second_search = len(types) - 1 - types[::-1].index(E.TOOL_CALLED)
    passed = types.index(E.VALIDATION_PASSED)
    assert first_search < failed < adapted < second_search < passed
    assert types[-1] is E.WORKFLOW_COMPLETED

    # The trace shows the queries (no prompts, no keys, no raw payloads)
    search_summaries = [
        event.decision_summary
        for event in events
        if event.type in (E.TOOL_DECISION_PROPOSED, E.TOOL_CALLED)
    ]
    assert any(FIRST_QUERY in (summary or "") for summary in search_summaries)
    assert any(REVISED_QUERY in (summary or "") for summary in search_summaries)

    # --- security: search data only, credentials never leak ---------------
    blob = json.dumps([event.model_dump(mode="json") for event in events])
    assert API_KEY not in blob
    assert json.dumps(final.model_dump(mode="json"), default=str).count(API_KEY) == 0
    # the final answer cites URLs that came from the search results
    assert "https://shop.example/asus-f15" in final.final_answer
    assert "https://shop.example/asus-f15" in second_urls
    assert "https://shop.example/nitro-v15" not in final.final_answer


# ---------------------------------------------------------------------------
# Step 17: offline demo regression - no credentials, no network
# ---------------------------------------------------------------------------


async def test_offline_demo_still_runs_without_search_credentials(tmp_path) -> None:
    network_calls: list[int] = []

    def forbidden_handler(request: httpx.Request) -> httpx.Response:
        network_calls.append(1)
        raise AssertionError("the offline demo must never touch the network")

    settings = load_settings({"ORBIT_WORKSPACE": str(tmp_path)})
    registry = create_default_registry(
        settings, search_client=httpx.Client(transport=httpx.MockTransport(forbidden_handler))
    )
    assert registry.get("web_search").available is False

    controller = WorkflowController(InMemoryStateStore(), max_replans=3)
    provider = create_demo_provider()
    orchestrator = WorkflowOrchestrator(
        controller,
        GoalAnalyzer(provider),
        Planner(provider),
        Executor(controller, provider, registry),
        Validator(provider),
        Replanner(provider),
        synthesizer=Synthesizer(provider),
    )

    workflow = controller.create_workflow(DEMO_GOAL)
    report = await orchestrator.run(workflow.id)

    assert report.success is True
    assert report.termination == "validated"
    assert report.replans == 1
    final = controller.get_workflow(workflow.id)
    assert final.state is S.COMPLETED
    assert final.replan_count == 1
    assert final.validation is not None
    assert final.validation.status.value == "PASS"
    assert final.final_answer == DEMO_SYNTHESIS.final_answer

    # The scripted story: 9000 FAIL -> adapt -> 7500 PASS
    tool_results = [
        observation.raw.get("result")
        for observation in final.observations
        if observation.source is ObservationSource.TOOL
    ]
    assert tool_results == [9000, 7500]
    assert network_calls == []  # no HTTP happened at all
    assert provider.model_id.startswith("mock/")
