"""State store abstraction.

Phase 1 ships an in-memory implementation. The ``StateStore`` protocol keeps
the controller independent of persistence, so a JSON/SQLite backend can be
swapped in later without touching controller logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.state.events import EventLog
from app.state.models import Workflow


class WorkflowNotFoundError(Exception):
    """Raised when a workflow id is not present in the store."""

    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id
        super().__init__(f"workflow not found: {workflow_id}")


@runtime_checkable
class StateStore(Protocol):
    """Persistence interface used by the controller."""

    def save_workflow(self, workflow: Workflow) -> None: ...

    def get_workflow(self, workflow_id: str) -> Workflow: ...

    def get_event_log(self, workflow_id: str) -> EventLog: ...

    def save_event_log(self, workflow_id: str, log: EventLog) -> None: ...

    def list_workflow_ids(self) -> list[str]: ...


class InMemoryStateStore:
    """Dict-backed store. Returns deep copies so callers cannot mutate state
    behind the controller's back."""

    def __init__(self) -> None:
        self._workflows: dict[str, Workflow] = {}
        self._logs: dict[str, EventLog] = {}

    def save_workflow(self, workflow: Workflow) -> None:
        self._workflows[workflow.id] = workflow.model_copy(deep=True)

    def get_workflow(self, workflow_id: str) -> Workflow:
        stored = self._workflows.get(workflow_id)
        if stored is None:
            raise WorkflowNotFoundError(workflow_id)
        return stored.model_copy(deep=True)

    def get_event_log(self, workflow_id: str) -> EventLog:
        log = self._logs.get(workflow_id)
        if log is None:
            log = EventLog(workflow_id)
            self._logs[workflow_id] = log
        return log

    def save_event_log(self, workflow_id: str, log: EventLog) -> None:
        if log.workflow_id != workflow_id:
            raise ValueError(
                f"event log for {log.workflow_id!r} cannot be saved as {workflow_id!r}"
            )
        self._logs[workflow_id] = log

    def list_workflow_ids(self) -> list[str]:
        return sorted(self._workflows)
