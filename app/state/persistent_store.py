"""SQLite-backed StateStore: durable workflows + append-only event logs.

Swaps in behind the same ``StateStore`` protocol the Controller already
uses, so persistence needs no controller changes:

- ``workflows`` table: one row per workflow, canonical pydantic JSON
- ``events`` table: one row per event keyed by (workflow_id, seq),
  append-only by construction (seq is the primary-key component)
- one connection with a lock; every mutation commits immediately
- every read deserializes fresh objects, so callers can never mutate
  persisted state behind the Controller's back (mirrors the in-memory
  store's copy semantics)

No API keys, prompts, or raw reasoning live in these rows - workflows and
events carry only structured state and decision summaries.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from app.state.events import Event, EventLog
from app.state.models import Workflow
from app.state.store import WorkflowNotFoundError

__all__ = ["PersistentStateStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    workflow_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (workflow_id, seq)
);
"""


class PersistentStateStore:
    """SQLite implementation of the ``StateStore`` protocol."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the underlying connection (safe to call twice)."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> PersistentStateStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # StateStore protocol
    # ------------------------------------------------------------------

    def save_workflow(self, workflow: Workflow) -> None:
        payload = workflow.model_dump_json()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO workflows (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (workflow.id, payload),
            )

    def get_workflow(self, workflow_id: str) -> Workflow:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM workflows WHERE id = ?", (workflow_id,)
            ).fetchone()
        if row is None:
            raise WorkflowNotFoundError(workflow_id)
        return Workflow.model_validate_json(row[0])

    def get_event_log(self, workflow_id: str) -> EventLog:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM events WHERE workflow_id = ? ORDER BY seq",
                (workflow_id,),
            ).fetchall()
        log = EventLog(workflow_id)
        log.load([Event.model_validate_json(data) for (data,) in rows])
        return log

    def save_event_log(self, workflow_id: str, log: EventLog) -> None:
        if log.workflow_id != workflow_id:
            raise ValueError(
                f"event log for {log.workflow_id!r} cannot be saved as "
                f"{workflow_id!r}"
            )
        rows = [
            (workflow_id, event.seq, event.model_dump_json())
            for event in log.all()
        ]
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM events WHERE workflow_id = ?", (workflow_id,)
            )
            self._conn.executemany(
                "INSERT INTO events (workflow_id, seq, data) VALUES (?, ?, ?)",
                rows,
            )

    def list_workflow_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM workflows ORDER BY id"
            ).fetchall()
        return [workflow_id for (workflow_id,) in rows]
