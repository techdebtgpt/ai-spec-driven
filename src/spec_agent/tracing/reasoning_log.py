from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
from uuid import uuid4

from ..domain.models import LogEntry
from ..persistence.store import JsonStore


class ReasoningLog:
    """
    Structured audit log for every major step (clarification, plan, spec, etc.).
    """

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def record(self, task_id: str, entry_type: str, payload: Dict) -> LogEntry:
        entry = LogEntry(
            id=str(uuid4()),
            task_id=task_id,
            timestamp=datetime.now(timezone.utc),
            entry_type=entry_type,
            payload=payload,
        )
        self.store.append_log(entry)
        return entry


