"""Persistence tests: durable workflows, append-only event logs, recovery.

A fresh ``PersistentStateStore`` on the same SQLite file must reproduce the
exact workflow and event history the Controller had before the "restart",
and the orchestrator must finish workflows interrupted at READY or at
AWAITING_VALIDATION. Windows note: every file-backed store is closed
(``store.close()``) before the test ends so no handle leaks.
"""

from __future__ import annotations

import pytest

from app.agent.controller import WorkflowController
from app.agent.executor import Executor
from app.agent.goal_analyzer import GoalAnalyzer
from app.agent.planner import Planner
from app.agent.replanner import Replanner
from app.agent.validator import Validator
from app.agent.workflow import WorkflowOrchestrator
from app.config.settings import load_settings
from app.llm.mock import RoutedMockProvider
from app.llm.schemas import TaskResult, ValidationProposal
from app.state.events import Event, EventLog, EventType
from app.state.models import (
    GoalAnalysis,
    Plan,
    SynthesisResult,
    Task,
    WorkflowState,
)
from app.state.persistent_store import PersistentStateStore
from app.state.store import WorkflowNotFoundError
from app.tools.builtin import create_default_registry

S = WorkflowState
E = EventType
GOAL = "Find options under a budget of 8000 INR"

ANALYSIS = GoalAnalysis(
    objective="Find options under 8000",
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

REASONING = TaskResult(reason_summary="Three options under budget.", data={})
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


@pytest.fixture
def store(tmp_path):
    s = PersistentStateStore(tmp_path / "orbit.db")
    yield s
    s.close()


def test_workflow_roundtrip_preserves_exact_state(store) -> None:
    controller = WorkflowController(store, max_replans=2)
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    controller.record_analysis(workflow.id, ANALYSIS)
    controller.record_plan(workflow.id, PLAN)

    loaded = store.get_workflow(workflow.id)
    assert loaded == controller.get_workflow(workflow.id)
    assert loaded.state is S.READY
    assert loaded.plan.version == 1


def test_reads_return_independent_copies(store) -> None:
    controller = WorkflowController(store, max_replans=2)
    workflow = controller.create_workflow(GOAL)
    copy = store.get_workflow(workflow.id)
    copy.final_answer = "tampered"
    assert store.get_workflow(workflow.id).final_answer is None


def test_unknown_workflow_raises_not_found(store) -> None:
    with pytest.raises(WorkflowNotFoundError):
        store.get_workflow("missing")


def test_list_workflow_ids_is_sorted(store) -> None:
    controller = WorkflowController(store, max_replans=2)
    b = controller.create_workflow("beta goal").id
    a = controller.create_workflow("alpha goal").id
    assert store.list_workflow_ids() == sorted([a, b])


def test_event_log_roundtrip_preserves_seq_and_timestamps(store) -> None:
    controller = WorkflowController(store, max_replans=2)
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    original = [e for e in store.get_event_log(workflow.id).all()]
    assert len(original) == 2  # workflow.started, state.changed

    restored = store.get_event_log(workflow.id)
    assert [e.seq for e in restored.all()] == [e.seq for e in original]
    assert [e.timestamp for e in restored.all()] == [
        e.timestamp for e in original
    ]
    assert [e.type for e in restored.all()] == [e.type for e in original]


def test_appending_after_reload_continues_the_sequence(store) -> None:
    controller = WorkflowController(store, max_replans=2)
    workflow = controller.create_workflow(GOAL)
    controller.begin_analysis(workflow.id)
    log = store.get_event_log(workflow.id)
    assert len(log.all()) == 2

    log.append(
        event_type=E.GOAL_ANALYZED,
        state=S.ANALYZING,
    )  # seq 3 after restore
    store.save_event_log(workflow.id, log)
    reloaded = store.get_event_log(workflow.id)
    assert [e.seq for e in reloaded.all()] == [0, 1, 2]
    assert reloaded.all()[-1].type is E.GOAL_ANALYZED


def test_save_event_log_rejects_foreign_workflow_id(store) -> None:
    log = EventLog("log-workflow")
    with pytest.raises(ValueError, match="cannot be saved as"):
        store.save_event_log("other-workflow", log)


def test_event_log_load_refuses_a_non_empty_log() -> None:
    populated = EventLog("wf")
    populated.append(event_type=E.WORKFLOW_STARTED, state=S.RECEIVED)
    with pytest.raises(ValueError, match="non-empty"):
        populated.load([])


def test_event_log_load_rejects_foreign_and_non_contiguous_events() -> None:
    good = EventLog("wf-a")
    good.append(event_type=E.WORKFLOW_STARTED, state=S.RECEIVED)
    events = list(good.all())

    foreign = EventLog("wf-b")
    foreign.append(event_type=E.WORKFLOW_STARTED, state=S.RECEIVED)
    with pytest.raises(ValueError, match="does not belong"):
        EventLog("wf-a").load(list(foreign.all()))

    # seq gap: build a log, tamper with seq via model_copy
    gap = [
        events[0],
        events[0].model_copy(update={"seq": 2}),
    ]
    with pytest.raises(ValueError, match="contiguous"):
        EventLog("wf-a").load(gap)


def test_file_created_and_close_is_idempotent(tmp_path) -> None:
    db = tmp_path / "nested" / "orbit.db"
    store = PersistentStateStore(db)
    assert db.exists()
    controller = WorkflowController(store, max_replans=2)
    controller.create_workflow(GOAL)
    store.close()
    store.close()  # safe twice


# ---------------------------------------------------------------------------
# Restart recovery: same file, brand new objects
# ---------------------------------------------------------------------------


async def test_workflow_survives_a_restart_at_ready(tmp_path) -> None:
    db = tmp_path / "orbit.db"
    provider = RoutedMockProvider(
        GoalAnalysis=[ANALYSIS],
        Plan=[PLAN],
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
        SynthesisResult=[SYNTH],
    )
    registry = create_default_registry(load_settings({}))

    # First "process": drive the workflow to READY, then stop.
    store1 = PersistentStateStore(db)
    controller1 = WorkflowController(store1, max_replans=2)
    workflow = controller1.create_workflow(GOAL)
    controller1.begin_analysis(workflow.id)
    controller1.record_analysis(workflow.id, ANALYSIS)
    controller1.record_plan(workflow.id, PLAN)
    assert controller1.get_workflow(workflow.id).state is S.READY
    store1.close()

    # Second "process": reopen the file, resume and complete.
    store2 = PersistentStateStore(db)
    controller2 = WorkflowController(store2, max_replans=2)
    restored = controller2.get_workflow(workflow.id)
    assert restored.state is S.READY
    assert restored.plan.tasks[0].id == "t1"
    assert len(controller2.events(workflow.id)) == 6  # history intact

    orchestrator = WorkflowOrchestrator(
        controller2,
        GoalAnalyzer(provider),
        Planner(provider),
        Executor(controller2, provider, registry),
        Validator(provider),
        Replanner(provider),
    )
    report = await orchestrator.run(restored.id)
    assert report.termination == "validated"
    assert report.success is True
    store2.close()


async def test_workflow_survives_a_restart_at_awaiting_validation(
    tmp_path,
) -> None:
    db = tmp_path / "orbit.db"
    provider = RoutedMockProvider(
        GoalAnalysis=[ANALYSIS],
        Plan=[PLAN],
        TaskResult=[REASONING],
        ValidationProposal=[PASS_PROPOSAL],
        SynthesisResult=[SYNTH],
    )
    registry = create_default_registry(load_settings({}))

    # First "process": run exactly one executor pass, then "crash".
    store1 = PersistentStateStore(db)
    controller1 = WorkflowController(store1, max_replans=2)
    executor1 = Executor(controller1, provider, registry)
    workflow = controller1.create_workflow(GOAL)
    controller1.begin_analysis(workflow.id)
    controller1.record_analysis(workflow.id, ANALYSIS)
    controller1.record_plan(workflow.id, PLAN)
    execution = await executor1.execute(workflow.id)
    assert execution.success is True
    assert execution.final_state is S.AWAITING_VALIDATION
    assert controller1.get_workflow(workflow.id).state is S.AWAITING_VALIDATION
    store1.close()

    # Second "process": reopen, validate and finish.
    store2 = PersistentStateStore(db)
    controller2 = WorkflowController(store2, max_replans=2)
    restored = controller2.get_workflow(workflow.id)
    assert restored.state is S.AWAITING_VALIDATION
    assert restored.task_states["t1"] is not None

    orchestrator = WorkflowOrchestrator(
        controller2,
        GoalAnalyzer(provider),
        Planner(provider),
        Executor(controller2, provider, registry),
        Validator(provider),
        Replanner(provider),
    )
    report = await orchestrator.run(restored.id)
    assert report.termination == "validated"
    assert controller2.get_workflow(workflow.id).state is S.COMPLETED
    store2.close()


async def test_events_survive_restart_and_new_events_append(tmp_path) -> None:
    db = tmp_path / "orbit.db"
    store1 = PersistentStateStore(db)
    controller1 = WorkflowController(store1, max_replans=2)
    workflow = controller1.create_workflow(GOAL)
    controller1.begin_analysis(workflow.id)
    before = list(controller1.events(workflow.id))
    store1.close()

    store2 = PersistentStateStore(db)
    controller2 = WorkflowController(store2, max_replans=2)
    after = list(controller2.events(workflow.id))
    assert [(e.seq, e.type, e.timestamp) for e in after] == [
        (e.seq, e.type, e.timestamp) for e in before
    ]
    controller2.record_analysis(workflow.id, ANALYSIS)
    extended = controller2.events(workflow.id)
    assert [e.seq for e in extended] == list(range(len(extended)))
    assert [e.type for e in extended[: len(before)]] == [
        e.type for e in before
    ]
    store2.close()
