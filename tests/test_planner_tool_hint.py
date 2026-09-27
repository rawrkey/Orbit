"""The requires_tool / tool_hint contract for real (LLM) planning.

Root cause these tests pin down (frontend run with a real goal and a real
LLM): the planner prompt only said when to OMIT tool_hint and never listed
the registered tools, and the JSON schema marked the field optional with no
description. A real LLM therefore emitted ``{"requires_tool": true}`` with
``tool_hint`` null; deterministic validation correctly rejected it
("requires a tool but has no tool_hint") and the bounded repair repeated
the same omission -> ``PlanValidationError``.

The contract is now explicit in the prompt AND the schema, the registered
tool names travel from the registry into the planner/replanner, and
``validate_plan`` rejects any hint that is not a registered tool.

Everything here is offline: no network, no API key; baseline tests untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import app.config.settings as settings_module
from app.agent.planner import (
    PLANNER_SYSTEM_PROMPT,
    PlanValidationError,
    Planner,
    validate_plan,
)
from app.agent.replanner import REPLANNER_SYSTEM_PROMPT, ReplanValidationError, Replanner
from app.api.app import create_app
from app.config.settings import load_settings
from app.llm.mock import MockProvider, RoutedMockProvider
from app.llm.schemas import ToolDecision, ValidationProposal
from app.state.models import (
    Constraint,
    ConstraintOperator,
    ConstraintRule,
    ConstraintType,
    EvidenceRef,
    GoalAnalysis,
    Plan,
    SynthesisResult,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
    Workflow,
)
from app.state.store import InMemoryStateStore
from app.tools.builtin import BUILTIN_TOOL_NAMES

GOAL = "Compare 3 gaming laptops under 80,000 INR with 16GB RAM and RTX 4050."
SEARCH_URL = "https://shop.example/asus-f15"
API_KEY = "langsearch-hint-key-do-not-leak"

ANALYSIS = GoalAnalysis(
    objective="Compare gaming laptops under 80000 INR with 16GB RAM.",
    reason_summary="Budget and specs are explicit.",
    clarification_required=False,
    is_feasible=True,
)


def make_task(**overrides: object) -> Task:
    base: dict = {
        "id": "t1",
        "title": "Find candidates",
        "description": "Search the web for RTX 4050 laptops under 80000 INR",
        "requires_tool": True,
        "success_criteria": "candidate laptops found",
    }
    base.update(overrides)
    return Task(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. The prompt and schema state the contract explicitly, in both directions
# ---------------------------------------------------------------------------


def test_prompt_states_the_contract_in_both_directions() -> None:
    prompt = PLANNER_SYSTEM_PROMPT.format(
        max_tasks=7, available_tools="'web_search'"
    )

    assert "requires_tool=false -> tool_hint must be null or omitted" in prompt
    assert "requires_tool=true -> tool_hint is REQUIRED" in prompt
    assert 'tool_hint="web_search"' in prompt


def test_planner_system_prompt_lists_the_registered_tools() -> None:
    planner = Planner(MockProvider(), available_tools=BUILTIN_TOOL_NAMES)

    prompt = planner._system_prompt()

    for name in BUILTIN_TOOL_NAMES:
        assert repr(name) in prompt
    assert "any tool registered" not in prompt


def test_planner_prompt_falls_back_cleanly_without_a_tool_list() -> None:
    planner = Planner(MockProvider())

    prompt = planner._system_prompt()

    assert "requires_tool=true -> tool_hint is REQUIRED" in prompt
    assert "any tool registered in this runtime" in prompt


def test_task_schema_describes_the_tool_hint_contract() -> None:
    schema = Task.model_json_schema()
    hint = schema["properties"]["tool_hint"]["description"].lower()
    requires = schema["properties"]["requires_tool"]["description"].lower()

    assert "required" in hint and "requires_tool" in hint
    assert "tool_hint is required" in requires


# ---------------------------------------------------------------------------
# 2. Deterministic validation: unregistered hints rejected, rules intact
# ---------------------------------------------------------------------------


def test_validate_plan_rejects_a_tool_hint_that_is_not_registered() -> None:
    plan = Plan(tasks=[make_task(tool_hint="search")])

    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan, available_tools=("calculator", "web_search"))

    issues = excinfo.value.issues
    assert any(
        "is not a registered tool" in issue and "'search'" in issue
        for issue in issues
    )
    assert any("calculator" in issue for issue in issues)


def test_validate_plan_without_a_tool_list_keeps_baseline_behaviour() -> None:
    plan = Plan(tasks=[make_task(tool_hint="file")])

    validate_plan(plan)


def test_missing_tool_hint_is_still_rejected_with_a_tool_list() -> None:
    plan = Plan(tasks=[make_task()])

    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan, available_tools=BUILTIN_TOOL_NAMES)

    assert any(
        "no tool_hint" in issue for issue in excinfo.value.issues
    )


def test_hint_without_requires_tool_is_still_rejected_with_a_tool_list() -> None:
    plan = Plan(
        tasks=[make_task(requires_tool=False, tool_hint="web_search")]
    )

    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan, available_tools=BUILTIN_TOOL_NAMES)

    assert any(
        "requires_tool is false" in issue
        for issue in excinfo.value.issues
    )


# ---------------------------------------------------------------------------
# 3. The bounded repair actually fixes a real-shaped omission
# ---------------------------------------------------------------------------


async def test_planner_repairs_a_missing_tool_hint_in_one_round() -> None:
    omitted = Plan(tasks=[make_task()])
    corrected = Plan(tasks=[make_task(tool_hint="web_search")])
    provider = MockProvider([omitted, corrected])
    planner = Planner(provider, available_tools=BUILTIN_TOOL_NAMES)

    plan = await planner.plan(GOAL, ANALYSIS)

    assert plan.tasks[0].tool_hint == "web_search"
    assert len(provider.calls) == 2

    # The repair prompt carries the exact issue ...
    assert "requires a tool but has no tool_hint" in provider.calls[1].prompt
    # ... and both calls carry the contract plus the registered tool list.
    for call in provider.calls:
        assert call.system is not None
        assert "requires_tool=true -> tool_hint is REQUIRED" in call.system
        assert repr("web_search") in call.system
        assert "Registered tools" in call.prompt


async def test_planner_still_fails_after_the_bounded_repair() -> None:
    unregistered = Plan(tasks=[make_task(tool_hint="search")])
    provider = MockProvider([unregistered, unregistered])
    planner = Planner(provider, available_tools=("calculator", "web_search"))

    with pytest.raises(PlanValidationError) as excinfo:
        await planner.plan(GOAL, ANALYSIS)

    issues = excinfo.value.issues
    assert any("is not a registered tool" in issue for issue in issues)
    assert len(provider.calls) == 2  # exactly one bounded repair


# ---------------------------------------------------------------------------
# 4. The replanner carries the same contract
# ---------------------------------------------------------------------------

REPLAN_ANALYSIS = GoalAnalysis(
    objective="Plan a trip under 8000 INR",
    reason_summary="Duration and budget are explicit.",
    constraints=[
        Constraint(
            id="c1",
            name="budget",
            type=ConstraintType.BUDGET,
            description="Total stay under 8000 INR",
            rule=ConstraintRule(
                operator=ConstraintOperator.LTE, value=8000.0, unit="INR"
            ),
        )
    ],
)


def make_replan_workflow() -> Workflow:
    plan = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Compute total",
                description="Compute the trip total from published prices",
                requires_tool=True,
                tool_hint="calculator",
                constraint_ids=["c1"],
                success_criteria="total computed",
            )
        ],
    )
    return Workflow(
        id="wf1",
        goal="Plan a trip under 8000 INR",
        analysis=REPLAN_ANALYSIS,
        plan=plan,
        task_states={"t1": TaskStatus.COMPLETED},
        replan_count=1,
    )


FAILED_VALIDATION = ValidationResult(
    status=ValidationStatus.FAIL,
    reason_summary="Trip total 9000 INR exceeds the 8000 INR budget.",
    satisfied_constraints=[],
    violated_constraints=["c1"],
    evidence=[EvidenceRef(task_id="t1", note="total 9000")],
    replanning_required=True,
)


async def test_replanner_rejects_an_unregistered_hint_and_lists_the_tools() -> None:
    proposal = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Find cheaper total",
                description="Recompute the total using cheaper stay options",
                requires_tool=True,
                tool_hint="search",
                constraint_ids=["c1"],
                success_criteria="revised total computed",
            )
        ],
    )
    provider = MockProvider([proposal, proposal])
    replanner = Replanner(
        provider, available_tools=("calculator", "web_search")
    )

    with pytest.raises(ReplanValidationError) as excinfo:
        await replanner.propose(make_replan_workflow(), FAILED_VALIDATION)

    assert any(
        "is not a registered tool" in issue
        for issue in excinfo.value.issues
    )
    assert len(provider.calls) == 2
    system = provider.calls[0].system
    assert system is not None
    assert "EXACTLY one registered tool" in system
    assert repr("calculator") in system
    assert "Constraints are FIXED by the user" in REPLANNER_SYSTEM_PROMPT


async def test_replanner_accepts_a_registered_hint() -> None:
    proposal = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Cheaper calculator run",
                description="Recompute the total using cheaper stays",
                requires_tool=True,
                tool_hint="calculator",
                constraint_ids=["c1"],
                success_criteria="revised total computed",
            )
        ],
    )
    provider = MockProvider([proposal])
    replanner = Replanner(
        provider, available_tools=("calculator", "web_search")
    )

    accepted = await replanner.propose(
        make_replan_workflow(), FAILED_VALIDATION
    )

    assert accepted.tasks[0].tool_hint == "calculator"


# ---------------------------------------------------------------------------
# 5. End to end: the laptop goal repairs its hint and really searches
# ---------------------------------------------------------------------------

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


def scripted_laptop_llm_with_omitted_hint() -> RoutedMockProvider:
    """A real-shaped script reproducing the observed failure."""
    omitted_hint = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Search",
                description="Search for RTX 4050 laptops under 80000 INR",
                requires_tool=True,
                success_criteria="candidate laptops found",
            )
        ],
    )
    corrected = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Search",
                description="Search for RTX 4050 laptops under 80000 INR",
                requires_tool=True,
                tool_hint="web_search",
                success_criteria="candidate laptops found",
            )
        ],
    )
    return RoutedMockProvider(
        GoalAnalysis=[ANALYSIS],
        Plan=[omitted_hint, corrected],
        ToolDecision=[
            ToolDecision(
                tool_name="web_search",
                arguments={
                    "query": "RTX 4050 16GB laptop under 80000",
                    "max_results": 3,
                },
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


async def test_laptop_goal_repairs_the_hint_and_runs_the_real_search(
    env_file: Path,
) -> None:
    env_file.write_text(LANGSEARCH_ENV, encoding="utf-8")
    settings = load_settings()
    provider = scripted_laptop_llm_with_omitted_hint()

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
            provider=provider,
            settings=settings,
        )

    controller = app.state.controller
    workflow_id = controller.create_workflow(GOAL).id
    report = await app.state.orchestrator.run(workflow_id)

    # The planner asked twice: first attempt omitted tool_hint and failed
    # deterministic validation, the repair succeeded with the explicit
    # contract and the registered tool list in both prompts.
    plan_calls = [c for c in provider.calls if c.schema_name == "Plan"]
    assert len(plan_calls) == 2
    assert "requires a tool but has no tool_hint" in plan_calls[1].prompt
    assert "Registered tools" in plan_calls[1].prompt
    assert plan_calls[1].system is not None
    assert "requires_tool=true -> tool_hint is REQUIRED" in plan_calls[1].system

    # The corrected plan names a registered tool and the search really ran.
    assert report.success is True
    final = controller.get_workflow(workflow_id)
    assert final.plan is not None
    assert final.plan.tasks[0].tool_hint == "web_search"
    assert seen_queries == ["RTX 4050 16GB laptop under 80000"]
    called_tools = [
        event.payload.get("tool_name")
        for event in controller.events(workflow_id)
        if event.type.value == "tool.called"
    ]
    assert called_tools == ["web_search"]

    # The scripted budget scenario was never replayed.
    blob = json.dumps(
        [event.model_dump(mode="json") for event in controller.events(workflow_id)]
    )
    assert "9000" not in blob
    assert "7500" not in blob


# ---------------------------------------------------------------------------
# 6. The ACTUAL live Nemotron failure mode (captured request/response path)
# ---------------------------------------------------------------------------

# Exact shape of the raw ``message.content`` Nemotron returned for the live
# failing planning request (captured via temporary instrumentation against
# the real configured endpoint): every task with requires_tool=true came
# back WITHOUT the tool_hint key at all - the decoder emits ``required``
# keys but drops optional nullable fields, no matter what the prose prompt
# says. Both legs (initial + bounded repair) looked like this.
NEMOTRON_RAW_FAILING_PLAN = json.dumps(
    {
        "tasks": [
            {
                "id": "t1",
                "title": "Search for gaming laptops",
                "description": "Search the market for RTX 4050 laptops under 80000 INR",
                "requires_tool": True,
                "success_criteria": "candidate models found",
                "depends_on": [],
                "constraint_ids": [],
                "status": "PENDING",
                "attempts": 0,
                # tool_hint / input_hint / expected_output ABSENT - live behavior
            },
            {
                "id": "t2",
                "title": "Compare candidates",
                "description": "Compare the retrieved candidates",
                "requires_tool": False,
                "success_criteria": "comparison written",
                "depends_on": ["t1"],
                "constraint_ids": [],
                "status": "PENDING",
                "attempts": 0,
            },
            {
                "id": "t3",
                "title": "Fetch detailed specs",
                "description": "Fetch detailed specs for the selected models",
                "requires_tool": True,
                "success_criteria": "spec sheets collected",
                "depends_on": ["t2"],
                "constraint_ids": [],
                "status": "PENDING",
                "attempts": 0,
                # tool_hint ABSENT - exactly what the live trace rejected
            },
        ],
        "version": 1,
    }
)

# Shape returned AFTER the schema fix on the same endpoint/goal: every task
# carries tool_hint (string when a tool is needed, explicit null otherwise).
NEMOTRON_RAW_CORRECTED_PLAN = json.dumps(
    {
        "tasks": [
            {
                "id": "t1",
                "title": "Search for gaming laptops",
                "description": "Search the market for RTX 4050 laptops under 80000 INR",
                "requires_tool": True,
                "tool_hint": "web_search",
                "input_hint": "gaming laptop under 80000 INR RTX 4050 16GB RAM",
                "success_criteria": "candidate models found",
                "depends_on": [],
                "constraint_ids": [],
                "status": "PENDING",
                "attempts": 0,
            },
            {
                "id": "t2",
                "title": "Compare candidates",
                "description": "Compare the retrieved candidates",
                "requires_tool": False,
                "tool_hint": None,
                "success_criteria": "comparison written",
                "depends_on": ["t1"],
                "constraint_ids": [],
                "status": "PENDING",
                "attempts": 0,
            },
        ],
        "version": 1,
    }
)


def test_task_schema_structurally_requires_tool_hint() -> None:
    """The schema actually sent to the LLM carries the contract (D/E)."""
    task_schema = Plan.model_json_schema()["$defs"]["Task"]

    assert "tool_hint" in task_schema["required"]
    rule = task_schema["allOf"][0]
    assert rule["if"]["properties"]["requires_tool"] == {"const": True}
    assert rule["if"]["required"] == ["requires_tool"]
    assert rule["then"]["required"] == ["tool_hint"]
    assert rule["then"]["properties"]["tool_hint"] == {"type": "string"}


async def test_payload_sent_to_the_llm_carries_the_structural_contract() -> None:
    """The real complete() path: response_format AND system message (H)."""
    import httpx

    from app.llm.openrouter import OpenRouterProvider

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": NEMOTRON_RAW_CORRECTED_PLAN}}
                ]
            },
        )

    provider = OpenRouterProvider(
        api_key="test-key-not-real",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await provider.complete(
        "plan it", Plan, system=PLANNER_SYSTEM_PROMPT.format(
            max_tasks=7, available_tools="'web_search'"
        )
    )

    body = json.loads(captured[0].content)
    sent = body["response_format"]["json_schema"]["schema"]
    assert "tool_hint" in sent["$defs"]["Task"]["required"]
    assert "allOf" in sent["$defs"]["Task"]

    system_text = body["messages"][0]["content"]
    assert '"allOf"' in system_text
    assert '"tool_hint"' in system_text
    assert "Required JSON schema" in system_text
    await provider.aclose()


def test_captured_failing_response_reaches_the_live_validation_error() -> None:
    """Reproduce the actual failure: raw Nemotron content -> live issue."""
    from app.llm.openrouter import OpenRouterProvider

    plan = OpenRouterProvider._validate(NEMOTRON_RAW_FAILING_PLAN, Plan)

    # The schema-level parse accepts it (tool_hint is nullable at the
    # Python layer) - deterministic validation is the guard, unchanged.
    assert plan.tasks[0].tool_hint is None
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan, available_tools=BUILTIN_TOOL_NAMES)

    issues = excinfo.value.issues
    assert "task 't1' requires a tool but has no tool_hint" in issues
    assert "task 't3' requires a tool but has no tool_hint" in issues


def test_captured_corrected_response_passes_planning_validation() -> None:
    """The post-fix live shape (verified on the real endpoint) validates."""
    from app.llm.openrouter import OpenRouterProvider

    plan = OpenRouterProvider._validate(NEMOTRON_RAW_CORRECTED_PLAN, Plan)

    validate_plan(plan, available_tools=BUILTIN_TOOL_NAMES)
    assert plan.tasks[0].tool_hint == "web_search"
    assert plan.tasks[1].tool_hint is None
