"""Evidence handoff: web-search results reach downstream tasks as bounded,
dependency-scoped structured evidence.

Root cause fixed here: ``ExecutionContext.previous_observations`` reduced
every predecessor to a one-line digest, so the structured WebSearchOutput
(title/url/snippet/source) the Observer stored in ``Observation.raw`` never
reached the LLM of the tasks that needed it. Downstream extraction and
comparison tasks had no data to reason from, validation could never find
evidence, and the replan budget was exhausted on a workflow that had in
fact already searched successfully.

The architectural fix (no new provider, no crawling, no prompt-only patch):
- ``EvidenceRecord`` + ``dependency_evidence`` on the execution context
- dependency-scoped: only the transitive ``depends_on`` closure contributes
- freshness shared with the Validator via ``observations_by_task`` - evidence
  survives replans that preserve the producer, goes stale when redefined
- explicit bounds: record count, chars per record, total chars, omitted count
- REASONING_SYSTEM / TaskResult require an explicit insufficient-evidence
  marker when the facts are absent - absence never becomes a fabrication

Everything runs offline: only the external LangSearch HTTP call is mocked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.controller import WorkflowController
from app.agent.executor import (
    MAX_EVIDENCE_CHARS_PER_RECORD,
    MAX_EVIDENCE_RECORDS,
    MAX_EVIDENCE_TOTAL_CHARS,
    Executor,
    build_execution_context,
    build_dependency_evidence,
)
from app.agent.goal_analyzer import GoalAnalyzer
from app.agent.observer import MAX_RAW_BYTES
from app.agent.planner import Planner
from app.agent.replanner import Replanner
from app.agent.synthesizer import Synthesizer
from app.agent.validator import Validator
from app.agent.workflow import WorkflowOrchestrator
from app.api.app import create_app
from app.config.settings import load_settings
from app.llm.mock import MockProvider, RoutedMockProvider
from app.llm.schemas import TaskResult, ToolDecision, ValidationProposal
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    GoalAnalysis,
    Observation,
    ObservationSource,
    Plan,
    SynthesisResult,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    WorkflowState,
)
from app.state.store import InMemoryStateStore
from app.tools.builtin import create_default_registry

S = WorkflowState

GOAL = "Find gaming laptops under 80000 INR with 16GB RAM and RTX 4050."
QUERY = "RTX 4050 gaming laptop 16GB under 80000 INR"
SEARCH_KEY = "langsearch-evidence-key-XYZ789-DO-NOT-LEAK"
OPENROUTER_KEY = "sk-or-evidence-key-FAKE-DO-NOT-LEAK"
PRICE = 74990.0

# What the mocked LangSearch HTTP endpoint returns under data.webPages.value.
# Six realistic candidates: model, Indian rupee price, CPU, GPU, RAM,
# storage and display all live in the snippet (the only free-text field
# the provider preserves). Three qualify (16GB + RTX 4050 + <= 80000 INR).
SEARCH_RESULTS = [
    {
        "name": "ASUS TUF F15",
        "url": "https://shop.example/asus-f15",
        "snippet": (
            "ASUS TUF F15: Ryzen 7 7435H, RTX 4050, 16GB RAM, 512GB SSD, "
            "15.6in 144Hz display - \u20b974,990"
        ),
    },
    {
        "name": "Acer Nitro V14",
        "url": "https://store.example/nitro-v14",
        "snippet": (
            "Acer Nitro V14: Ryzen 5 7535HS, RTX 4050, 16GB RAM, 512GB SSD, "
            "15.6in 144Hz display - \u20b977,499"
        ),
    },
    {
        "name": "Lenovo IdeaPad Gaming 3",
        "url": "https://market.example/ideapad-rtx4050",
        "snippet": (
            "Lenovo IdeaPad Gaming 3: Core i5-12450H, RTX 4050, 16GB RAM, "
            "1TB SSD, 15.6in 120Hz display - \u20b979,500"
        ),
    },
    {
        "name": "HP Victus 16",
        "url": "https://shop.example/victus-16",
        "snippet": (
            "HP Victus 16: Core i5-13420H, RTX 4050, 8GB RAM, 512GB SSD, "
            "16.1in 144Hz display - \u20b976,990"
        ),
    },
    {
        "name": "MSI Katana 15",
        "url": "https://store.example/katana-15",
        "snippet": (
            "MSI Katana 15: Core i7-13620H, RTX 4050, 16GB RAM, 1TB SSD, "
            "15.6in 144Hz display - \u20b989,990"
        ),
    },
    {
        "name": "Dell G15 3530",
        "url": "https://market.example/dell-g15",
        "snippet": (
            "Dell G15 3530: Core i5-13450HX, RTX 3050, 16GB RAM, 512GB SSD, "
            "15.6in 120Hz display - \u20b972,990"
        ),
    },
]
SOURCE_URL = SEARCH_RESULTS[0]["url"]

IRRELEVANT_RESULTS = [
    {
        "name": "How to choose a gaming laptop",
        "url": "https://blog.example/laptop-guide",
        "snippet": "Tips for picking a laptop in 2026.",
    },
]


ANALYSIS = GoalAnalysis(
    objective="Find gaming laptops under 80000 INR with 16GB RAM and RTX 4050.",
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
        )
    ],
    clarification_required=False,
    is_feasible=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalized_results() -> list[dict]:
    """The shape the real provider stores: title/url/snippet/source only."""
    return [
        {
            "title": item["name"],
            "url": item["url"],
            "snippet": item["snippet"],
            "source": item["url"].split("/")[2],
        }
        for item in SEARCH_RESULTS
    ]


def search_raw(query: str = QUERY) -> dict:
    return {"query": query, "provider": "langsearch", "results": normalized_results()}


def search_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        title="Search",
        description="Search the web for RTX 4050 laptops under 80000 INR",
        requires_tool=True,
        tool_hint="web_search",
        success_criteria="search results observed",
    )


def reasoning_task(
    task_id: str, description: str, depends_on: list[str] | None = None
) -> Task:
    return Task(
        id=task_id,
        title=task_id,
        description=description,
        depends_on=depends_on or [],
        success_criteria=f"{task_id} completed",
    )


def search_decision(query: str = QUERY) -> ToolDecision:
    return ToolDecision(
        tool_name="web_search",
        # ask for enough candidates that every fixture result flows through
        arguments={"query": query, "max_results": 10},
        reason_summary="Find candidate laptops within budget.",
    )


def langsearch_settings(workspace: Path, **extra: str):
    env = {
        "ORBIT_WORKSPACE": str(workspace),
        "WEB_SEARCH_PROVIDER": "langsearch",
        "LANGSEARCH_API_KEY": SEARCH_KEY,
    }
    env.update(extra)
    return load_settings(env)


def make_search_client(
    seen_queries: list[str], results: list[dict] | None = None
) -> httpx.Client:
    """Mocked LangSearch endpoint: only the external HTTP call is faked."""
    payload = results if results is not None else SEARCH_RESULTS

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers["Authorization"] == f"Bearer {SEARCH_KEY}"
        body = json.loads(request.content)
        seen_queries.append(body["query"])
        return httpx.Response(200, json={"data": {"webPages": {"value": payload}}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def ready_workflow(controller: WorkflowController, plan: Plan) -> str:
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, plan)
    return workflow.id


def fail_validation() -> ValidationResult:
    return ValidationResult(
        status=ValidationStatus.FAIL,
        reason_summary="Observed price exceeds the 80000 INR budget.",
        violated_constraints=["c1"],
        replanning_required=True,
    )


def append_raw_observation(
    workflow, task_id: str, raw, *, summary: str = "recorded"
) -> None:
    workflow.observations.append(
        Observation(
            task_id=task_id,
            content_summary=summary,
            raw=raw,
            source=ObservationSource.REASONING,
        )
    )


@pytest.fixture
def controller() -> WorkflowController:
    return WorkflowController(InMemoryStateStore(), max_replans=3)


@pytest.fixture
def seen_queries() -> list[str]:
    return []


@pytest.fixture
def registry(tmp_path: Path, seen_queries: list[str]):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return create_default_registry(
        langsearch_settings(workspace),
        search_client=make_search_client(seen_queries),
    )


async def run_chain(
    controller, provider, registry, plan: Plan
) -> tuple[object, str]:
    workflow_id = ready_workflow(controller, plan)
    report = await Executor(controller, provider, registry).execute(workflow_id)
    return report, workflow_id


# ---------------------------------------------------------------------------
# Steps 1-3: structured evidence exists, is stored, and reaches the prompt
# ---------------------------------------------------------------------------


async def test_web_search_result_is_stored_as_structured_evidence(
    controller, registry, seen_queries
) -> None:
    plan = Plan(tasks=[search_task("t1"), reasoning_task("t2", "extract", ["t1"])])
    provider = MockProvider(
        [
            search_decision(),
            TaskResult(
                reason_summary="Lowest observed price extracted.",
                data={"max_price_inr": PRICE},
            ),
        ]
    )
    report, workflow_id = await run_chain(controller, provider, registry, plan)

    assert report.success is True
    workflow = controller.get_workflow(workflow_id)
    observation = workflow.observations[0]
    assert observation.source is ObservationSource.TOOL
    assert observation.tool_name == "web_search"
    assert observation.is_error is False

    raw = observation.raw
    assert raw is not None
    assert raw["query"] == QUERY
    assert raw["provider"] == "langsearch"
    assert [r["url"] for r in raw["results"]] == [
        item["url"] for item in SEARCH_RESULTS
    ]
    for result in raw["results"]:
        assert set(result) == {"title", "url", "snippet", "source"}

    # bounded in workflow state, round-trippable, keyless
    serialized = json.dumps(raw, ensure_ascii=False)
    assert len(serialized.encode("utf-8")) <= MAX_RAW_BYTES
    blob = json.dumps(workflow.model_dump(mode="json"), default=str)
    assert SOURCE_URL in blob
    assert SEARCH_KEY not in blob


async def test_downstream_task_receives_predecessor_evidence_in_context(
    controller, registry, seen_queries
) -> None:
    plan = Plan(tasks=[search_task("t1"), reasoning_task("t2", "extract", ["t1"])])
    provider = MockProvider(
        [
            search_decision(),
            TaskResult(
                reason_summary="Lowest observed price extracted.",
                data={"max_price_inr": PRICE},
            ),
        ]
    )
    report, workflow_id = await run_chain(controller, provider, registry, plan)
    assert report.success is True

    workflow = controller.get_workflow(workflow_id)
    extract = workflow.plan.tasks[1]
    context = build_execution_context(workflow, extract, registry)

    assert [e.task_id for e in context.dependency_evidence] == ["t1"]
    record = context.dependency_evidence[0]
    assert record.source is ObservationSource.TOOL
    assert record.tool_name == "web_search"
    assert record.truncated is False
    assert record.data["query"] == QUERY
    assert record.data["results"][0]["url"] == SOURCE_URL
    assert record.data["results"][0]["snippet"].endswith("\u20b974,990")
    assert context.evidence_omitted == 0

    # the actual LLM prompt for t2 carries the structured search results
    prompt = provider.calls[1].prompt
    assert "dependency_evidence" in prompt
    assert SOURCE_URL in prompt
    assert "langsearch" in prompt
    assert "\u20b974,990" in prompt
    assert "chain_of_thought" not in prompt


async def test_unrelated_task_evidence_is_scoped_out_of_the_context(
    controller, registry, seen_queries
) -> None:
    plan = Plan(
        tasks=[
            search_task("t_search"),
            reasoning_task("t_other", "Record a side note"),
            reasoning_task("t_extract", "Extract from the search", ["t_search"]),
        ]
    )
    provider = MockProvider(
        [
            search_decision(),
            TaskResult(
                reason_summary="Side note recorded.",
                data={"marker": "UNRELATED-MARKER-77"},
            ),
            TaskResult(
                reason_summary="Extracted from search evidence.",
                data={"max_price_inr": PRICE},
            ),
        ]
    )
    report, workflow_id = await run_chain(controller, provider, registry, plan)
    assert report.success is True

    workflow = controller.get_workflow(workflow_id)
    extract = workflow.plan.tasks[2]
    context = build_execution_context(workflow, extract, registry)

    # only the dependency contributes structured evidence ...
    assert [e.task_id for e in context.dependency_evidence] == ["t_search"]
    # ... and the unrelated task's data never appears anywhere in the prompt
    prompt = provider.calls[2].prompt
    assert "UNRELATED-MARKER-77" not in prompt
    assert SOURCE_URL in prompt


async def test_evidence_flows_transitively_along_the_dependency_chain(
    controller, registry, seen_queries
) -> None:
    plan = Plan(
        tasks=[
            search_task("t1"),
            reasoning_task("t2", "Extract specs", ["t1"]),
            reasoning_task("t3", "Compare candidates", ["t2"]),
        ]
    )
    provider = MockProvider(
        [
            search_decision(),
            TaskResult(
                reason_summary="Lowest observed price extracted.",
                data={"max_price_inr": PRICE, "ram_gb": 16},
            ),
            TaskResult(
                reason_summary="Cheapest candidate is within budget.",
                data={"max_price_inr": PRICE, "within_budget": True},
            ),
        ]
    )
    report, workflow_id = await run_chain(controller, provider, registry, plan)
    assert report.success is True

    workflow = controller.get_workflow(workflow_id)
    compare = workflow.plan.tasks[2]
    context = build_execution_context(workflow, compare, registry)

    assert [e.task_id for e in context.dependency_evidence] == ["t1", "t2"]
    assert context.dependency_evidence[0].data["results"][0]["url"] == SOURCE_URL
    assert context.dependency_evidence[1].data == {
        "max_price_inr": PRICE,
        "ram_gb": 16,
    }

    # both hops are visible to the LLM that produced t3's result
    prompt = provider.calls[2].prompt
    assert SOURCE_URL in prompt  # hop 1: raw search results
    assert "max_price_inr" in prompt  # hop 2: t2's structured extraction


# ---------------------------------------------------------------------------
# Step 4: evidence survives replans; goes stale when the producer is redefined
# ---------------------------------------------------------------------------


def run_t1_with_search_evidence(controller, plan: Plan) -> str:
    """Drive t1 through execution with a real-shaped search observation."""
    workflow_id = ready_workflow(controller, plan)
    controller.begin_execution(workflow_id)
    controller.begin_task(workflow_id, "t1")
    controller.record_observation(
        workflow_id,
        Observation(
            task_id="t1",
            content_summary="Tool 'web_search' returned: provider, query, results",
            raw=search_raw(),
            source=ObservationSource.TOOL,
            tool_name="web_search",
        ),
    )
    controller.complete_task(workflow_id, "t1")
    controller.finish_execution(workflow_id)
    controller.begin_validation(workflow_id)
    controller.record_validation(workflow_id, fail_validation())
    controller.begin_replanning(workflow_id, reason="budget exceeded")
    return workflow_id


def test_evidence_survives_replan_for_the_preserved_producer(
    controller, registry
) -> None:
    workflow_id = run_t1_with_search_evidence(
        controller, Plan(tasks=[search_task("t1")])
    )

    # replan adds a downstream consumer; t1's definition is untouched
    updated = controller.replace_plan(
        workflow_id,
        Plan(tasks=[search_task("t1"), reasoning_task("t2", "extract", ["t1"])]),
        reason="add extraction step",
    )
    assert updated.task_states["t1"] is TaskStatus.COMPLETED

    context = build_execution_context(updated, updated.plan.tasks[1], registry)
    assert [e.task_id for e in context.dependency_evidence] == ["t1"]
    record = context.dependency_evidence[0]
    # complete, unbounded search evidence survives the replan untouched
    assert record.truncated is False
    assert len(record.data["results"]) == len(SEARCH_RESULTS)
    assert all(
        set(result) == {"title", "url", "snippet", "source"}
        for result in record.data["results"]
    )
    assert record.data["results"][0]["url"] == SOURCE_URL


def test_evidence_goes_stale_when_the_producer_is_redefined(
    controller, registry
) -> None:
    workflow_id = run_t1_with_search_evidence(
        controller, Plan(tasks=[search_task("t1")])
    )

    redefined_search = search_task("t1")
    redefined_search.description = (
        "Search again for RTX 4050 laptops under 80000 INR in India"
    )
    updated = controller.replace_plan(
        workflow_id,
        Plan(tasks=[redefined_search, reasoning_task("t2", "extract", ["t1"])]),
        reason="refine the search step",
    )
    # redefined producer resets: its old observation is stale evidence
    assert updated.task_states["t1"] is TaskStatus.PENDING
    assert updated.observations, "the stale observation is kept in history"

    context = build_execution_context(updated, updated.plan.tasks[1], registry)
    assert context.dependency_evidence == []


# ---------------------------------------------------------------------------
# Step 7: explicit bounds
# ---------------------------------------------------------------------------


def six_dependency_plan(ids: list[str]) -> Plan:
    return Plan(
        tasks=[
            *[reasoning_task(task_id, f"step {task_id}") for task_id in ids],
            reasoning_task("target", "consume every dependency", ids),
        ]
    )


def test_evidence_record_cap_bounds_how_many_dependencies_appear(
    controller, registry
) -> None:
    ids = [f"d{i}" for i in range(6)]
    workflow_id = ready_workflow(controller, six_dependency_plan(ids))
    workflow = controller.get_workflow(workflow_id)
    for task_id in ids:
        append_raw_observation(workflow, task_id, {"blob": f"{task_id}-payload"})

    records, omitted = build_dependency_evidence(workflow, workflow.plan.tasks[-1])

    assert len(records) == MAX_EVIDENCE_RECORDS
    assert omitted == 1
    assert [r.task_id for r in records] == ids[:MAX_EVIDENCE_RECORDS]


def test_evidence_char_budget_truncates_records_and_caps_the_total(
    controller, registry
) -> None:
    ids = [f"d{i}" for i in range(6)]
    workflow_id = ready_workflow(controller, six_dependency_plan(ids))
    workflow = controller.get_workflow(workflow_id)
    for index, task_id in enumerate(ids):
        # d0 exceeds the per-record cap; the rest share the total budget
        payload = "y" * 6000 if index == 0 else "x" * 2000
        append_raw_observation(workflow, task_id, {"blob": payload})

    records, omitted = build_dependency_evidence(workflow, workflow.plan.tasks[-1])

    assert len(records) + omitted == len(ids)
    assert 1 <= len(records) < len(ids)
    serialized = [
        json.dumps(record.data, ensure_ascii=False) for record in records
    ]
    assert all(len(text) <= MAX_EVIDENCE_CHARS_PER_RECORD for text in serialized)
    assert sum(len(text) for text in serialized) <= MAX_EVIDENCE_TOTAL_CHARS
    # deterministic truncation marker instead of silent data loss
    oversized = records[0]
    assert oversized.task_id == "d0"
    assert oversized.truncated is True
    assert oversized.data["truncated"] is True
    assert oversized.data["original_chars"] > MAX_EVIDENCE_CHARS_PER_RECORD


# ---------------------------------------------------------------------------
# Evidence quality: complete search-result records under the same bounds
# ---------------------------------------------------------------------------


def raw_search_for(results: list[dict]) -> dict:
    return {"query": QUERY, "provider": "langsearch", "results": results}


def normalized_item(index: int, snippet: str) -> dict:
    return {
        "title": f"Candidate {index}",
        "url": f"https://shop.example/candidate-{index}",
        "snippet": snippet,
        "source": "shop.example",
    }


def complete_specs(count: int) -> list[dict]:
    big = "Detailed specification block " * 12  # ~350 chars per snippet
    return [normalized_item(index, f"{big}#{index}") for index in range(count)]


def evidence_for_raw(controller, raw: dict):
    """Attach a TOOL search observation to t1 and build t2's evidence."""
    plan = Plan(tasks=[search_task("t1"), reasoning_task("t2", "use it", ["t1"])])
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    workflow.observations.append(
        Observation(
            task_id="t1",
            content_summary="Tool 'web_search' returned: provider, query, results",
            raw=raw,
            source=ObservationSource.TOOL,
            tool_name="web_search",
        )
    )
    return build_dependency_evidence(workflow, workflow.plan.tasks[1])


def test_search_evidence_keeps_every_result_complete_within_the_record_budget(
    controller,
) -> None:
    results = [
        normalized_item(index, f"RTX 4050 candidate {index} with full specs")
        for index in range(5)
    ]
    records, omitted = evidence_for_raw(controller, raw_search_for(results))

    assert omitted == 0
    assert len(records) == 1
    record = records[0]
    assert record.truncated is False
    assert "results_omitted" not in record.data
    # identical to what the tool stored: nothing merged, nothing cut
    assert record.data["results"] == results


def test_oversized_search_evidence_keeps_a_complete_head_never_a_mid_record_cut(
    controller,
) -> None:
    results = complete_specs(12)
    raw = raw_search_for(results)
    assert len(json.dumps(raw, ensure_ascii=False)) > MAX_EVIDENCE_CHARS_PER_RECORD

    records, omitted = evidence_for_raw(controller, raw)

    assert omitted == 0
    assert len(records) == 1
    record = records[0]
    data = record.data
    # never the old character preview that sliced a result in half
    assert "preview" not in data
    assert "original_chars" not in data
    kept = data["results"]
    assert 1 <= len(kept) < len(results)
    # deterministic head, in order, each record byte-identical to the input
    assert kept == results[: len(kept)]
    assert data["results_omitted"] == len(results) - len(kept)
    assert record.truncated is True
    assert len(json.dumps(data, ensure_ascii=False)) <= MAX_EVIDENCE_CHARS_PER_RECORD


def test_search_evidence_truncation_is_deterministic(controller) -> None:
    raw = raw_search_for(complete_specs(12))
    first, first_omitted = evidence_for_raw(controller, raw)
    second, second_omitted = evidence_for_raw(controller, raw)

    assert first_omitted == second_omitted
    assert [record.model_dump() for record in first] == [
        record.model_dump() for record in second
    ]


def test_second_search_dependency_keeps_complete_results_under_the_total_budget(
    controller,
) -> None:
    # s1 nearly fills the total budget; s2 must still contribute COMPLETE
    # results (a head) instead of being dropped wholesale or preview-cut.
    big = "Verified specification block " * 12
    raw1 = raw_search_for(
        [normalized_item(index, f"{big}#a{index}") for index in range(9)]
    )
    raw2 = raw_search_for(
        [normalized_item(100 + index, f"{big}#b{index}") for index in range(9)]
    )
    assert len(json.dumps(raw1, ensure_ascii=False)) > MAX_EVIDENCE_CHARS_PER_RECORD
    assert len(json.dumps(raw2, ensure_ascii=False)) > MAX_EVIDENCE_CHARS_PER_RECORD

    plan = Plan(
        tasks=[
            search_task("s1"),
            search_task("s2"),
            reasoning_task("target", "consume both searches", ["s1", "s2"]),
        ]
    )
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    for task_id, raw in (("s1", raw1), ("s2", raw2)):
        workflow.observations.append(
            Observation(
                task_id=task_id,
                content_summary="Tool 'web_search' returned: provider, query, results",
                raw=raw,
                source=ObservationSource.TOOL,
                tool_name="web_search",
            )
        )

    records, omitted = build_dependency_evidence(
        workflow, workflow.plan.tasks[-1]
    )

    # both dependencies appear (the old code dropped the second whole)
    assert len(records) == 2
    assert omitted == 0
    serialized = [json.dumps(r.data, ensure_ascii=False) for r in records]
    assert all(len(text) <= MAX_EVIDENCE_CHARS_PER_RECORD for text in serialized)
    assert sum(len(text) for text in serialized) <= MAX_EVIDENCE_TOTAL_CHARS
    for record, source in zip(records, (raw1, raw2)):
        assert "preview" not in record.data
        kept = record.data["results"]
        assert kept == source["results"][: len(kept)]
        assert record.data["results_omitted"] == len(source["results"]) - len(kept)
        assert record.truncated is True


async def test_downstream_prompt_receives_every_complete_search_result(
    controller, registry, seen_queries
) -> None:
    plan = Plan(tasks=[search_task("t1"), reasoning_task("t2", "extract", ["t1"])])
    provider = MockProvider(
        [
            search_decision(),
            TaskResult(
                reason_summary="Qualifying candidates extracted.",
                data={"candidate_count": 3},
            ),
        ]
    )
    report, workflow_id = await run_chain(controller, provider, registry, plan)
    assert report.success is True

    # every fixture result arrives whole - the cut is never inside a record
    prompt = provider.calls[1].prompt
    for item in SEARCH_RESULTS:
        assert item["name"] in prompt
        assert item["url"] in prompt
        assert item["snippet"] in prompt
    assert "results_omitted" not in prompt
    assert "preview" not in prompt

    workflow = controller.get_workflow(workflow_id)
    context = build_execution_context(workflow, workflow.plan.tasks[1], registry)
    record = context.dependency_evidence[0]
    assert record.truncated is False
    assert len(record.data["results"]) == len(SEARCH_RESULTS)


def test_unrelated_search_evidence_stays_out_of_the_dependency_context(
    controller, registry
) -> None:
    plan = Plan(
        tasks=[
            search_task("s1"),
            search_task("s2"),
            reasoning_task("target", "consume s1 only", ["s1"]),
        ]
    )
    workflow_id = ready_workflow(controller, plan)
    workflow = controller.get_workflow(workflow_id)
    for task_id, url, snippet in (
        ("s1", "https://shop.example/candidate-1", "Kept specifications"),
        ("s2", "https://rival.example/deal-of-the-day", "Unrelated specifications"),
    ):
        workflow.observations.append(
            Observation(
                task_id=task_id,
                content_summary="Tool 'web_search' returned: provider, query, results",
                raw=raw_search_for(
                    [
                        {
                            "title": f"Result for {task_id}",
                            "url": url,
                            "snippet": snippet,
                            "source": url.split("/")[2],
                        }
                    ]
                ),
                source=ObservationSource.TOOL,
                tool_name="web_search",
            )
        )

    context = build_execution_context(workflow, workflow.plan.tasks[-1], registry)
    assert [e.task_id for e in context.dependency_evidence] == ["s1"]
    evidence_json = json.dumps(
        [record.model_dump() for record in context.dependency_evidence],
        ensure_ascii=False,
    )
    assert "https://shop.example/candidate-1" in evidence_json
    assert "https://rival.example/deal-of-the-day" not in evidence_json


# ---------------------------------------------------------------------------
# Steps 6 + 8: insufficient evidence surfaces as insufficient, never faked
# ---------------------------------------------------------------------------


async def test_reasoning_without_usable_evidence_gets_an_empty_evidence_list(
    controller, registry, seen_queries
) -> None:
    plan = Plan(
        tasks=[
            reasoning_task("t1", "note something"),
            reasoning_task("t2", "derive from t1", ["t1"]),
        ]
    )
    provider = MockProvider(
        [
            TaskResult(reason_summary="Nothing to record.", data=None),
            TaskResult(reason_summary="No data to build on.", data={}),
        ]
    )
    report, workflow_id = await run_chain(controller, provider, registry, plan)
    assert report.success is True

    workflow = controller.get_workflow(workflow_id)
    context = build_execution_context(workflow, workflow.plan.tasks[1], registry)
    assert context.dependency_evidence == []  # explicit emptiness, not absence

    prompt = provider.calls[1].prompt
    assert '"dependency_evidence":[]' in prompt
    system = provider.calls[1].system
    assert system is not None
    assert "insufficient_evidence" in system
    assert "invented number, spec, quote, or source" in system


async def test_irrelevant_results_yield_insufficient_evidence_not_fabrication(
    controller, tmp_path: Path, seen_queries
) -> None:
    """The opposite behavior: no specs in the results -> no invented specs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    registry = create_default_registry(
        langsearch_settings(workspace),
        search_client=make_search_client(seen_queries, IRRELEVANT_RESULTS),
    )
    plan = Plan(tasks=[search_task("t1"), reasoning_task("t2", "extract", ["t1"])])
    insufficient = TaskResult(
        reason_summary="The retrieved pages contain no laptop specs.",
        data={
            "insufficient_evidence": True,
            "reason": "snippets contain no prices or specifications",
        },
    )
    provider = RoutedMockProvider(
        ToolDecision=[search_decision("gaming laptop buying guide")],
        TaskResult=[insufficient],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="No price evidence was observed.",
                constraint_verdicts=[
                    {
                        "constraint_id": "c1",
                        "outcome": "insufficient",
                        "evidence": [],
                        "evidence_field": None,
                    }
                ],
                goal_outcome="insufficient",
            )
        ],
    )

    workflow_id = ready_workflow(controller, plan)
    report = await Executor(controller, provider, registry).execute(workflow_id)
    assert report.success is True

    workflow = controller.get_workflow(workflow_id)
    # t2 saw the (irrelevant) evidence and the contract forbade inventing
    task_calls = [c for c in provider.calls if c.schema_name == "TaskResult"]
    assert len(task_calls) == 1
    assert "https://blog.example/laptop-guide" in task_calls[0].prompt
    assert task_calls[0].system is not None
    assert "insufficient_evidence" in task_calls[0].system

    # the stored extraction is exactly the insufficient marker - no specs
    extraction = workflow.observations[-1]
    assert extraction.raw == insufficient.data
    assert extraction.raw.get("max_price_inr") is None
    assert extraction.raw.get("within_budget") is None

    controller.begin_validation(workflow_id)
    result = await Validator(provider).validate(controller.get_workflow(workflow_id))
    assert result.status is ValidationStatus.INSUFFICIENT
    assert result.satisfied_constraints == []
    assert result.evidence == []


# ---------------------------------------------------------------------------
# Steps 8-9: full offline e2e - search -> extract -> compare -> PASS -> answer
# ---------------------------------------------------------------------------


PRICE_TAG = re.compile(r"\u20b9([\d,]+)")


def snippet_price(snippet: str) -> int:
    match = PRICE_TAG.search(snippet)
    assert match is not None, f"no rupee price in snippet: {snippet!r}"
    return int(match.group(1).replace(",", ""))


def cpu_of(snippet: str) -> str:
    return snippet.split(", RTX", 1)[0].split(": ", 1)[1]


def storage_of(snippet: str) -> str:
    match = re.search(r"(\d+(?:GB|TB) SSD)", snippet)
    assert match is not None, snippet
    return match.group(1)


def display_of(snippet: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?in \d+Hz)", snippet)
    assert match is not None, snippet
    return match.group(1)


def qualifying_results() -> list[dict]:
    """Fixture-derived truth: 16GB + RTX 4050 + price <= 80000 INR."""
    return [
        item
        for item in SEARCH_RESULTS
        if "16GB RAM" in item["snippet"]
        and "RTX 4050" in item["snippet"]
        and snippet_price(item["snippet"]) <= 80000
    ]


def exclusion_reason(item: dict) -> str:
    snippet = item["snippet"]
    if "16GB RAM" not in snippet:
        return f"{item['name']}: less than 16GB RAM"
    if "RTX 4050" not in snippet:
        return f"{item['name']}: GPU is not RTX 4050"
    return f"{item['name']}: price {snippet_price(snippet)} over budget"


def candidate_rows() -> list[dict]:
    """Candidate facts parsed out of the qualifying snippets only."""
    return [
        {
            "model": item["name"],
            "price_inr": snippet_price(item["snippet"]),
            "cpu": cpu_of(item["snippet"]),
            "ram_gb": 16,
            "storage": storage_of(item["snippet"]),
            "display": display_of(item["snippet"]),
            "url": item["url"],
        }
        for item in qualifying_results()
    ]


def extraction_result() -> TaskResult:
    excluded = [
        exclusion_reason(item)
        for item in SEARCH_RESULTS
        if item not in qualifying_results()
    ]
    return TaskResult(
        reason_summary=f"{len(candidate_rows())} qualifying candidates extracted.",
        data={"candidates": candidate_rows(), "excluded": excluded},
    )


def comparison_result() -> TaskResult:
    rows = candidate_rows()
    cheapest = min(rows, key=lambda row: row["price_inr"])
    return TaskResult(
        reason_summary="Cheapest qualifying candidate is within budget.",
        data={
            "max_price_inr": cheapest["price_inr"],
            "within_budget": cheapest["price_inr"] <= 80000,
            "cheapest": cheapest["model"],
            "candidate_count": len(rows),
        },
    )


def evidence_chain_provider() -> RoutedMockProvider:
    plan = Plan(
        version=1,
        tasks=[
            search_task("t1"),
            reasoning_task(
                "t2",
                "Extract model, price, CPU, RAM, storage and display for "
                "qualifying candidates from the observed search results",
                ["t1"],
            ),
            reasoning_task(
                "t3",
                "Compare the extracted candidates and check the budget",
                ["t2"],
            ),
        ],
    )
    rows = candidate_rows()
    cheapest = min(rows, key=lambda row: row["price_inr"])
    sources = ", ".join(f"{row['model']} {row['url']}" for row in rows)
    return RoutedMockProvider(
        GoalAnalysis=[ANALYSIS],
        Plan=[plan],
        ToolDecision=[search_decision()],
        TaskResult=[extraction_result(), comparison_result()],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="Observed price respects the 80000 INR budget.",
                constraint_verdicts=[
                    {
                        "constraint_id": "c1",
                        "outcome": "satisfied",
                        "evidence": [
                            {
                                "task_id": "t3",
                                "note": f"max observed price {cheapest['price_inr']}",
                            }
                        ],
                        "evidence_field": "max_price_inr",
                    }
                ],
                goal_outcome="satisfied",
            )
        ],
        SynthesisResult=[
            SynthesisResult(
                final_answer=(
                    f"Cheapest qualifying candidate: {cheapest['model']} at "
                    f"\u20b9{cheapest['price_inr']:,} ({cheapest['cpu']}, "
                    f"RTX 4050, 16GB RAM, {cheapest['storage']}, "
                    f"{cheapest['display']}) - under the \u20b980,000 budget. "
                    "Also qualifying: "
                    + ", ".join(
                        f"{row['model']} at \u20b9{row['price_inr']:,}"
                        for row in rows
                        if row is not cheapest
                    )
                    + f". Sources: {sources}"
                ),
                key_results=[
                    f"Lowest price: \u20b9{cheapest['price_inr']:,}",
                    f"Qualifying candidates: {len(rows)} of {len(SEARCH_RESULTS)}",
                    "GPU: RTX 4050, RAM: 16GB",
                ],
                constraints_satisfied=["c1"],
                caveats=[
                    "Prices come from retrieved search snippets - confirm them "
                    "on the source page."
                ],
                next_actions=["Open the source link to confirm the price."],
            )
        ],
    )


def evidence_chain_orchestrator(controller, provider, registry):
    return WorkflowOrchestrator(
        controller,
        GoalAnalyzer(provider),
        Planner(provider),
        Executor(controller, provider, registry),
        Validator(provider),
        Replanner(provider),
        synthesizer=Synthesizer(provider),
    )


async def test_search_extract_compare_chain_validates_on_real_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    seen_queries: list[str] = []
    registry = create_default_registry(
        langsearch_settings(workspace),
        search_client=make_search_client(seen_queries),
    )
    controller = WorkflowController(InMemoryStateStore(), max_replans=3)
    provider = evidence_chain_provider()
    orchestrator = evidence_chain_orchestrator(controller, provider, registry)

    workflow = controller.create_workflow(GOAL)
    report = await orchestrator.run(workflow.id)

    # single-pass completion: evidence replaced the replan churn entirely
    assert report.success is True
    assert report.termination == "validated"
    assert (report.executions, report.validations, report.replans) == (1, 1, 0)

    final = controller.get_workflow(workflow.id)
    assert final.state is S.COMPLETED
    assert final.validation is not None
    assert final.validation.status is ValidationStatus.PASS
    assert final.validation.satisfied_constraints == ["c1"]
    assert final.validation.evidence[0].task_id == "t3"
    assert final.synthesis is not None

    # the real (mocked-HTTP) search ran exactly once
    assert seen_queries == [QUERY]

    qualifying = qualifying_results()
    rows = candidate_rows()
    assert len(qualifying) == 3
    assert len(rows) == 3

    # hop 1: t2's prompt carries every COMPLETE search result
    task_calls = [c for c in provider.calls if c.schema_name == "TaskResult"]
    assert len(task_calls) == 2
    t2_prompt = task_calls[0].prompt
    for item in SEARCH_RESULTS:
        assert item["name"] in t2_prompt
        assert item["url"] in t2_prompt
        assert item["snippet"] in t2_prompt
    assert "results_omitted" not in t2_prompt
    assert "preview" not in t2_prompt
    assert "within_budget" not in t2_prompt  # not produced yet

    # hop 2: t3 saw t2's structured extraction
    t3_prompt = task_calls[1].prompt
    assert "candidates" in t3_prompt
    assert "price_inr" in t3_prompt
    assert "Ryzen 7 7435H" in t3_prompt  # a fact extracted from the snippets

    # context level: one complete, bounded evidence record
    context = build_execution_context(final, final.plan.tasks[1], registry)
    record = context.dependency_evidence[0]
    assert record.truncated is False
    assert len(record.data["results"]) == len(SEARCH_RESULTS)
    assert (
        len(json.dumps(record.data, ensure_ascii=False))
        <= MAX_EVIDENCE_CHARS_PER_RECORD
    )

    # the extraction traces back to the retrieved data - nothing fabricated
    extraction, comparison = final.observations[-2], final.observations[-1]
    candidates = extraction.raw["candidates"]
    assert [c["model"] for c in candidates] == [q["name"] for q in qualifying]
    assert [c["price_inr"] for c in candidates] == [
        snippet_price(q["snippet"]) for q in qualifying
    ]
    assert [c["url"] for c in candidates] == [q["url"] for q in qualifying]
    for candidate, source in zip(candidates, qualifying):
        assert candidate["cpu"] in source["snippet"]
        assert candidate["storage"] in source["snippet"]
        assert candidate["display"] in source["snippet"]
        assert candidate["ram_gb"] == 16
    assert len(extraction.raw["excluded"]) == len(SEARCH_RESULTS) - len(qualifying)

    # the comparison re-derives the budget result from those candidates
    assert comparison.raw["max_price_inr"] == min(
        row["price_inr"] for row in rows
    )
    assert comparison.raw["candidate_count"] == len(rows)
    assert comparison.raw["cheapest"] == rows[0]["model"]
    assert comparison.raw["within_budget"] is True
    # Python independently re-checks the constraint rule from the evidence
    assert comparison.raw["max_price_inr"] <= ANALYSIS.constraints[0].rule.value

    # the final answer cites the qualifying sources and nothing else
    assert final.final_answer == final.synthesis.final_answer
    for item in qualifying:
        assert item["url"] in final.final_answer
    for item in SEARCH_RESULTS:
        if item not in qualifying:
            assert item["url"] not in final.final_answer


# ---------------------------------------------------------------------------
# Step 8: credentials never reach API responses, events, or state
# ---------------------------------------------------------------------------


def scripted_search_llm() -> RoutedMockProvider:
    # Constraint-free analysis: this run only proves credential hygiene, so
    # validation needs no extractable rule to reach PASS.
    analysis = GoalAnalysis(
        objective="Find gaming laptops under 80000 INR with 16GB RAM.",
        reason_summary="Budget and specs are explicit.",
        constraints=[],
        clarification_required=False,
        is_feasible=True,
    )
    plan = Plan(
        version=1,
        tasks=[search_task("t1")],
    )
    return RoutedMockProvider(
        GoalAnalysis=[analysis],
        Plan=[plan],
        ToolDecision=[search_decision()],
        ValidationProposal=[
            ValidationProposal(
                reason_summary="Candidate observed within budget.",
                constraint_verdicts=[],
                goal_outcome="satisfied",
            )
        ],
        SynthesisResult=[
            SynthesisResult(
                final_answer=f"Best match at 74990 INR - source: {SOURCE_URL}",
                key_results=["74990 INR"],
                constraints_satisfied=[],
                caveats=["Prices come from retrieved search snippets."],
                next_actions=["Confirm the price on the source page."],
            )
        ],
    )


def test_api_responses_events_and_state_never_expose_credentials(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    settings = load_settings(
        {
            "ORBIT_WORKSPACE": str(workspace),
            "WEB_SEARCH_PROVIDER": "langsearch",
            "LANGSEARCH_API_KEY": SEARCH_KEY,
            "OPENROUTER_API_KEY": OPENROUTER_KEY,
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {SEARCH_KEY}"
        return httpx.Response(
            200, json={"data": {"webPages": {"value": SEARCH_RESULTS}}}
        )

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
            provider=scripted_search_llm(),
            settings=settings,
        )

    client = TestClient(app)
    created = client.post("/workflows", json={"goal": GOAL})
    assert created.status_code == 201
    workflow_id = created.json()["workflow_id"]

    run = client.post(f"/workflows/{workflow_id}/run", json={})
    assert run.status_code == 200
    assert run.json()["success"] is True
    status = client.get(f"/workflows/{workflow_id}")
    events = client.get(f"/workflows/{workflow_id}/events")
    assert status.status_code == 200
    assert events.status_code == 200
    assert events.json()["count"] > 5

    for response in (run, status, events):
        assert SEARCH_KEY not in response.text
        assert OPENROUTER_KEY not in response.text
        assert "Bearer " not in response.text

    workflow = app.state.controller.get_workflow(workflow_id)
    dump = json.dumps(workflow.model_dump(mode="json"), default=str)
    assert SEARCH_KEY not in dump
    assert OPENROUTER_KEY not in dump
    # structured search evidence IS in state (that is the handoff working)
    assert SOURCE_URL in dump
