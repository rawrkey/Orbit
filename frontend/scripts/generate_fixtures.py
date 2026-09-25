"""Generate API fixtures from the REAL backend (offline, scripted providers).

Run from repo root:  python frontend/scripts/generate_fixtures.py
Writes JSON snapshots into frontend/src/api/fixtures/ so TypeScript types
can be checked (`satisfies`) and key-sets pinned against actual responses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.api.app import create_app  # noqa: E402
from app.config.settings import load_settings  # noqa: E402
from app.demo import create_demo_provider  # noqa: E402
from app.llm.mock import RoutedMockProvider  # noqa: E402
from app.state.models import GoalAnalysis  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "api" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

DEMO_GOAL = "Find options under a budget of 8000."

CLARIFY_ANALYSIS = GoalAnalysis(
    objective="Plan the trip once dates are known",
    reason_summary="Travel dates are missing.",
    required_inputs=["travel dates"],
    clarification_required=True,
)


def dump(name: str, payload: object) -> None:
    path = OUT / f"{name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    # --- 1. demo goal: full happy path -------------------------------
    demo = TestClient(
        create_app(provider=create_demo_provider(), settings=load_settings({}))
    )
    created = demo.post("/workflows", json={"goal": DEMO_GOAL})
    assert created.status_code == 201, created.text
    dump("create_201", created.json())
    wid = created.json()["workflow_id"]

    dump("status_received", demo.get(f"/workflows/{wid}").json())
    dump("events_received", demo.get(f"/workflows/{wid}/events").json())

    run = demo.post(f"/workflows/{wid}/run", json={})
    assert run.status_code == 200, run.text
    assert run.json()["termination"] == "validated", run.text
    dump("run_validated", run.json())

    dump("status_completed", demo.get(f"/workflows/{wid}").json())
    dump("events_demo", demo.get(f"/workflows/{wid}/events").json())

    rerun = demo.post(f"/workflows/{wid}/run", json={})
    assert rerun.json()["termination"] == "already_terminal"
    dump("run_already_terminal", rerun.json())

    # --- 2. error shapes ---------------------------------------------
    not_found = demo.get("/workflows/doesnotexist")
    assert not_found.status_code == 404
    dump("error_404", not_found.json())

    unprocessable = demo.post("/workflows", json={"goal": ""})
    assert unprocessable.status_code == 422
    dump("error_422", unprocessable.json())

    # --- 3. clarification round --------------------------------------
    clarifying = TestClient(
        create_app(
            provider=RoutedMockProvider(GoalAnalysis=[CLARIFY_ANALYSIS]),
            settings=load_settings({}),
        )
    )
    wid2 = clarifying.post(
        "/workflows", json={"goal": "Plan a trip."}
    ).json()["workflow_id"]
    await_run = clarifying.post(f"/workflows/{wid2}/run", json={})
    assert await_run.json()["termination"] == "awaiting_clarification"
    dump("run_awaiting", await_run.json())
    dump(
        "status_awaiting_clarification",
        clarifying.get(f"/workflows/{wid2}").json(),
    )
    conflict = clarifying.post(
        f"/workflows/{wid2}/run", json={"answers": {"wrong question": "Dec 1-4"}}
    )
    assert conflict.status_code == 409, conflict.text
    dump("error_409", conflict.json())

    # --- 4. failure path (no analysis scripted -> LLM error) ---------
    failing = TestClient(
        create_app(provider=RoutedMockProvider(), settings=load_settings({}))
    )
    wid3 = failing.post(
        "/workflows", json={"goal": DEMO_GOAL}
    ).json()["workflow_id"]
    failed_run = failing.post(f"/workflows/{wid3}/run", json={})
    assert failed_run.json()["success"] is False, failed_run.text
    dump("run_failed", failed_run.json())
    dump("status_failed", failing.get(f"/workflows/{wid3}").json())

    # --- 5. mid-run states driven through the REAL controller ---------
    _dump_midrun_statuses()

    print("all fixtures generated from the live backend")


def _dump_midrun_statuses() -> None:
    """Drive real workflows to EXECUTING / VALIDATING / REPLANNING."""
    from app.agent.controller import WorkflowController
    from app.state.models import (
        Constraint,
        ConstraintOperator,
        ConstraintRule,
        ConstraintType,
        GoalAnalysis,
        Observation,
        ObservationSource,
        Plan,
        Task,
        ValidationResult,
        ValidationStatus,
    )
    from app.state.persistent_store import PersistentStateStore

    analysis = GoalAnalysis(
        objective="Find options under 8000 INR",
        reason_summary="Budget cap explicit.",
        constraints=[
            Constraint(
                id="c1",
                name="budget",
                type=ConstraintType.BUDGET,
                description="Total under 8000 INR",
                rule=ConstraintRule(
                    operator=ConstraintOperator.LTE,
                    value=8000.0,
                    unit="INR",
                ),
            )
        ],
    )
    plan = Plan(
        version=1,
        tasks=[
            Task(
                id="t1",
                title="Compute total",
                description="Compute the option total",
                requires_tool=True,
                tool_hint="calculator",
                constraint_ids=["c1"],
                success_criteria="total computed",
            ),
            Task(
                id="t2",
                title="Summarize",
                description="Summarize the options",
                success_criteria="summarized",
            ),
        ],
    )
    failing_result = ValidationResult(
        status=ValidationStatus.FAIL,
        reason_summary="Total 9000 exceeds the 8000 cap.",
        violated_constraints=["c1"],
        evidence=[],
        replanning_required=True,
    )

    store = PersistentStateStore(":memory:")
    controller = WorkflowController(store, max_replans=3)

    def to_ready() -> str:
        workflow = controller.create_workflow(DEMO_GOAL)
        controller.begin_analysis(workflow.id)
        controller.record_analysis(workflow.id, analysis)
        controller.record_plan(workflow.id, plan)
        return workflow.id

    def finish_execution(workflow_id: str) -> None:
        controller.begin_execution(workflow_id)
        for task_id in ("t1", "t2"):
            controller.begin_task(workflow_id, task_id)
            controller.record_observation(
                workflow_id,
                Observation(
                    task_id=task_id,
                    content_summary=f"{task_id} done",
                    raw={"result": 9000 if task_id == "t1" else None},
                    source=ObservationSource.TOOL
                    if task_id == "t1"
                    else ObservationSource.REASONING,
                    tool_name="calculator" if task_id == "t1" else None,
                ),
            )
            controller.complete_task(workflow_id, task_id)
        controller.finish_execution(workflow_id)

    # EXECUTING
    wid_executing = to_ready()
    controller.begin_execution(wid_executing)
    controller.begin_task(wid_executing, "t1")
    dump("status_executing", _status_of(controller, wid_executing))

    # VALIDATING (execution finished, no verdict yet)
    wid_validating = to_ready()
    finish_execution(wid_validating)
    controller.begin_validation(wid_validating)
    dump("status_validating", _status_of(controller, wid_validating))

    # REPLANNING (validation FAILED, budget available)
    wid_replanning = to_ready()
    finish_execution(wid_replanning)
    controller.begin_validation(wid_replanning)
    controller.record_validation(wid_replanning, failing_result)
    controller.begin_replanning(wid_replanning)
    dump("status_replanning", _status_of(controller, wid_replanning))

    store.close()


def _status_of(controller, workflow_id: str) -> dict:
    workflow = controller.get_workflow(workflow_id)
    from app.api.app import _status

    return _status(workflow).model_dump(mode="json")


if __name__ == "__main__":
    main()
