"""Event system tests: sequencing, append-only behavior, association."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.state.events import Event, EventLog, EventType
from app.state.models import WorkflowState

S = WorkflowState


def make_log(workflow_id: str = "wf-1") -> EventLog:
    return EventLog(workflow_id)


# ---------------------------------------------------------------------------
# Sequence numbering
# ---------------------------------------------------------------------------


def test_sequence_starts_at_zero_and_increments() -> None:
    log = make_log()
    events = [
        log.append(event_type=EventType.STATE_CHANGED, state=S.RECEIVED),
        log.append(event_type=EventType.STATE_CHANGED, state=S.ANALYZING),
        log.append(event_type=EventType.TASK_STARTED, state=S.EXECUTING, task_id="t1"),
    ]
    assert [e.seq for e in events] == [0, 1, 2]
    assert len(log) == 3


def test_sequence_numbers_never_change_after_reads() -> None:
    log = make_log()
    log.append(event_type=EventType.WORKFLOW_STARTED, state=S.RECEIVED)
    first_snapshot = log.all()
    log.append(event_type=EventType.WORKFLOW_PLANNED, state=S.READY)
    assert first_snapshot[0].seq == 0
    assert [e.seq for e in log.all()] == [0, 1]


def test_each_workflow_log_has_independent_sequences() -> None:
    log_a = EventLog("wf-a")
    log_b = EventLog("wf-b")
    log_a.append(event_type=EventType.WORKFLOW_STARTED, state=S.RECEIVED)
    log_a.append(event_type=EventType.STATE_CHANGED, state=S.ANALYZING)
    event_b = log_b.append(event_type=EventType.WORKFLOW_STARTED, state=S.RECEIVED)
    assert event_b.seq == 0
    assert event_b.workflow_id == "wf-b"
    assert len(log_b) == 1


# ---------------------------------------------------------------------------
# Workflow association and payload
# ---------------------------------------------------------------------------


def test_events_are_associated_with_their_workflow() -> None:
    log = EventLog("wf-42")
    event = log.append(event_type=EventType.TOOL_CALLED, state=S.EXECUTING)
    assert event.workflow_id == "wf-42"


def test_event_preserves_state_task_and_summary() -> None:
    log = make_log()
    event = log.append(
        event_type=EventType.TOOL_CALLED,
        state=S.EXECUTING,
        task_id="t1",
        decision_summary="Search accommodation because it is the largest uncertain budget item.",
        payload={"tool_name": "web_search"},
    )
    assert event.state == S.EXECUTING
    assert event.task_id == "t1"
    assert event.decision_summary is not None
    assert event.decision_summary.startswith("Search accommodation")
    assert event.payload == {"tool_name": "web_search"}


def test_event_serialization_round_trip() -> None:
    log = make_log()
    original = log.append(
        event_type=EventType.VALIDATION_FAILED,
        state=S.VALIDATING,
        decision_summary="Budget constraint violated.",
        payload={"failed_constraints": ["c1"], "amount": 9500},
    )
    restored = Event.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.type == EventType.VALIDATION_FAILED
    assert restored.state == S.VALIDATING


# ---------------------------------------------------------------------------
# Append-only behavior
# ---------------------------------------------------------------------------


def test_log_exposes_no_removing_or_mutating_api() -> None:
    log = make_log()
    for method in ("remove", "clear", "pop", "delete", "discard", "rewrite"):
        assert not hasattr(log, method), f"EventLog must not expose {method}()"


def test_after_returns_only_newer_events() -> None:
    log = make_log()
    for _ in range(4):
        log.append(event_type=EventType.STATE_CHANGED, state=S.EXECUTING)
    assert [e.seq for e in log.after(1)] == [2, 3]
    assert log.after(3) == []
    assert [e.seq for e in log.after(-1)] == [0, 1, 2, 3]


def test_iterating_yields_all_events_in_order() -> None:
    log = make_log()
    for _ in range(3):
        log.append(event_type=EventType.STATE_CHANGED, state=S.READY)
    assert [e.seq for e in log] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Invalid events are rejected
# ---------------------------------------------------------------------------


def test_event_with_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Event(seq=0, workflow_id="wf", type="workflow.exploded", state=S.RECEIVED)


def test_event_with_negative_seq_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Event(seq=-1, workflow_id="wf", type=EventType.STATE_CHANGED, state=S.RECEIVED)


def test_event_log_requires_workflow_id() -> None:
    with pytest.raises(ValueError):
        EventLog("")
