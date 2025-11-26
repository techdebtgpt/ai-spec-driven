from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List

from ..domain.models import LogEntry, Task


class JsonStore:
    """
    Lightweight persistence for tasks and reasoning logs.

    The format matches the MVP requirement to keep structured JSON that can be
    exported or inspected outside the tool.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks_file = self.root / "tasks.json"
        self.logs_file = self.root / "logs.json"
        self._lock = Lock()

    # --- Task persistence -------------------------------------------------
    def load_tasks(self) -> List[Task]:
        if not self.tasks_file.exists():
            return []
        with self._lock:
            raw = json.loads(self.tasks_file.read_text())
        return [Task.from_dict(item) for item in raw]

    def save_tasks(self, tasks: Iterable[Task]) -> None:
        serializable = [task.to_dict() for task in tasks]
        with self._lock:
            self.tasks_file.write_text(json.dumps(serializable, indent=2))

    def upsert_task(self, task: Task) -> Task:
        tasks: Dict[str, Task] = {existing.id: existing for existing in self.load_tasks()}
        tasks[task.id] = task
        self.save_tasks(tasks.values())
        return task

    # --- Reasoning log persistence ---------------------------------------
    def append_log(self, entry: LogEntry) -> None:
        log_entries = self.load_logs()
        log_entries.append(entry)
        serializable = [
            {
                "id": log.id,
                "task_id": log.task_id,
                "timestamp": log.timestamp.isoformat(),
                "entry_type": log.entry_type,
                "payload": log.payload,
            }
            for log in log_entries
        ]
        with self._lock:
            self.logs_file.write_text(json.dumps(serializable, indent=2))

    def load_logs(self) -> List[LogEntry]:
        if not self.logs_file.exists():
            return []
        with self._lock:
            raw = json.loads(self.logs_file.read_text())
        return [
            LogEntry(
                id=item["id"],
                task_id=item["task_id"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                entry_type=item["entry_type"],
                payload=item["payload"],
            )
            for item in raw
        ]


