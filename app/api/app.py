"""FastAPI application factory: goal in, structured workflow out.

Endpoints (all JSON, all bounded):

- ``POST /workflows``                create a workflow from a goal
- ``GET  /workflows/{id}``           structured status snapshot
- ``POST /workflows/{id}/run``       drive the workflow (optional answers)
- ``GET  /workflows/{id}/events``    append-only decision log (``?after=``)

Error mapping: unknown workflow -> 404, controller rejections (e.g. a
clarification answer that misses a required question) -> 409, anything else
is a programming error -> 500. Responses never contain raw observation
payloads, prompts, API keys, or chain-of-thought.

Provider policy: OpenRouter when ``OPENROUTER_API_KEY`` is configured,
otherwise the offline scripted demo provider - but only when NOTHING is
configured. With web search explicitly enabled, silently replaying the
scripted demo would hide a missing ``OPENROUTER_API_KEY``, so that is an
explicit configuration error instead.

Demo mode: ``POST /workflows`` accepts an optional ``demo`` flag. Only a
workflow created with ``demo: true`` runs on the scripted offline demo
provider; every other workflow uses the provider-selection rules above
with no silent fallback and no change to the real path.
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from app.agent.controller import ControllerError, WorkflowController
from app.agent.executor import Executor
from app.agent.goal_analyzer import GoalAnalyzer
from app.agent.planner import Planner
from app.agent.replanner import Replanner
from app.agent.synthesizer import Synthesizer
from app.agent.validator import Validator
from app.agent.workflow import WorkflowOrchestrator
from app.api.schemas import (
    CreateWorkflowRequest,
    CreateWorkflowResponse,
    EventView,
    EventsResponse,
    RunWorkflowRequest,
    RunWorkflowResponse,
    WorkflowStatusResponse,
)
from app.config.settings import Settings, load_settings
from app.demo import create_demo_provider
from app.llm.base import LLMProvider
from app.llm.openrouter import OpenRouterProvider
from app.state.models import Workflow
from app.state.store import InMemoryStateStore, StateStore, WorkflowNotFoundError
from app.tools.builtin import create_default_registry

__all__ = ["build_provider", "create_app"]


def build_provider() -> LLMProvider:
    """Real provider when a key is configured, else the offline demo scripts.

    The scripted demo is the OFFLINE fallback: it runs only when nothing is
    configured. If web search is explicitly configured, replaying the demo
    budget scenario would silently swallow real intent to search - a missing
    LLM key must then be reported explicitly instead of faking a workflow.
    """
    settings = load_settings()
    if settings.openrouter_api_key:
        return OpenRouterProvider.from_settings(settings)
    if (settings.web_search_provider or "none").strip().lower() not in ("", "none"):
        raise ValueError(
            "OPENROUTER_API_KEY is not configured, but web search is enabled "
            f"({settings.web_search_provider}): a real LLM is required to use "
            "it. Set OPENROUTER_API_KEY to run real workflows, or set "
            "WEB_SEARCH_PROVIDER=none to run the offline scripted demo."
        )
    return create_demo_provider()


def create_app(
    *,
    store: StateStore | None = None,
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the ASGI app with fully wired agent stages."""
    resolved_settings = settings or load_settings()
    state_store = store or InMemoryStateStore()
    llm = provider or build_provider()
    registry = create_default_registry(resolved_settings)
    controller = WorkflowController(
        state_store, max_replans=resolved_settings.max_replans
    )

    def build_orchestrator(stage_llm: LLMProvider) -> WorkflowOrchestrator:
        return WorkflowOrchestrator(
            controller,
            GoalAnalyzer(stage_llm),
            Planner(
                stage_llm,
                max_tasks=resolved_settings.max_plan_tasks,
                available_tools=registry.names(),
            ),
            Executor(controller, stage_llm, registry),
            Validator(stage_llm),
            Replanner(stage_llm, available_tools=registry.names()),
            synthesizer=Synthesizer(stage_llm),
        )

    # The real workflow path: the configured provider (OpenRouter when a
    # key exists, the offline fallback only when nothing is configured).
    orchestrator = build_orchestrator(llm)
    # Explicit demo path: the existing scripted demo provider, selected
    # ONLY for workflows created with demo=true. Same controller, same
    # tools, same pipeline - just the scripted LLM for that workflow.
    demo_orchestrator = build_orchestrator(create_demo_provider())
    demo_workflows: set[str] = set()

    app = FastAPI(
        title="ORBIT",
        version="6.0.0",
        description=(
            "Adaptive goal-to-execution workflow agent. LLM proposes, "
            "deterministic validation disposes; the controller owns state."
        ),
    )
    app.state.controller = controller
    app.state.orchestrator = orchestrator

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    @app.exception_handler(WorkflowNotFoundError)
    async def _not_found(request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ControllerError)
    async def _rejected(request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post(
        "/workflows", response_model=CreateWorkflowResponse, status_code=201
    )
    def create_workflow(body: CreateWorkflowRequest) -> CreateWorkflowResponse:
        workflow = controller.create_workflow(body.goal)
        if body.demo:
            # Explicit opt-in only: this workflow is routed to the scripted
            # demo provider at run time. Nothing else changes for it.
            demo_workflows.add(workflow.id)
        return CreateWorkflowResponse(
            workflow_id=workflow.id, state=workflow.state
        )

    @app.get("/workflows/{workflow_id}", response_model=WorkflowStatusResponse)
    def get_workflow(workflow_id: str) -> WorkflowStatusResponse:
        return _status(controller.get_workflow(workflow_id))

    @app.post(
        "/workflows/{workflow_id}/run", response_model=RunWorkflowResponse
    )
    async def run_workflow(
        workflow_id: str, body: RunWorkflowRequest | None = None
    ) -> RunWorkflowResponse:
        answers = body.answers if body is not None else None
        target = (
            demo_orchestrator
            if workflow_id in demo_workflows
            else orchestrator
        )
        report = await target.run(workflow_id, answers=answers)
        return RunWorkflowResponse(
            workflow_id=report.workflow_id,
            success=report.success,
            final_state=report.final_state,
            executions=report.executions,
            validations=report.validations,
            replans=report.replans,
            termination=report.termination,
            failure=report.failure,
        )

    @app.get(
        "/workflows/{workflow_id}/events", response_model=EventsResponse
    )
    def get_events(
        workflow_id: str, after: int | None = Query(default=None, ge=0)
    ) -> EventsResponse:
        controller.get_workflow(workflow_id)  # 404 for unknown ids
        events = (
            controller.events_after(workflow_id, after)
            if after is not None
            else controller.events(workflow_id)
        )
        views = [EventView.from_event(event) for event in events]
        return EventsResponse(
            workflow_id=workflow_id, events=views, count=len(views)
        )

    return app


def _status(workflow: Workflow) -> WorkflowStatusResponse:
    """Structured status view - no raw observations, no prompts, no keys."""
    return WorkflowStatusResponse(
        workflow_id=workflow.id,
        goal=workflow.goal,
        state=workflow.state,
        replan_count=workflow.replan_count,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        task_count=len(workflow.plan.tasks) if workflow.plan else 0,
        observation_count=len(workflow.observations),
        event_count=workflow.event_count,
        plan_version=workflow.plan.version if workflow.plan else None,
        clarification_questions=workflow.clarification_questions,
        final_answer=workflow.final_answer,
        failure=workflow.failure,
    )
