from __future__ import annotations

import json
import hashlib
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
        self._lock = Lock()
        self.root = root

    @property
    def root(self) -> Path:
        return self._root

    @root.setter
    def root(self, value: Path) -> None:
        """
        Update the store root and all derived file paths.

        Tests sometimes rebind the store root after constructing `TaskOrchestrator`;
        keeping derived paths in sync avoids accidental writes to `~/.spec_agent`.
        """
        self._root = value
        self._root.mkdir(parents=True, exist_ok=True)
        self.tasks_file = self._root / "tasks.json"
        self.logs_file = self._root / "logs.json"
        self.index_file = self._root / "repository_index.json"
        # Per-repo cache of indexes so chat can reuse multiple repositories.
        self.indexes_dir = self._root / "repository_indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        # Convenience export for the (potentially large) Serena semantic tree so it's easy
        # to inspect in ~/.spec_agent without digging through repository_index.json.
        self.serena_tree_file = self._root / "serena_semantic_tree.json"

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

    # --- Repository index persistence ------------------------------------
    @staticmethod
    def _repo_key(repo_path: Path, branch: str | None = None) -> str:
        """
        Deterministic key for a repository path (+ optional branch).
        """
        normalized = str(repo_path.resolve())
        if branch:
            normalized = f"{normalized}::{branch}"
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def _index_file_for_repo(self, repo_path: Path, branch: str | None = None) -> Path:
        return self.indexes_dir / f"{self._repo_key(repo_path, branch)}.json"

    def save_repository_index(self, index_data: Dict) -> None:
        """Save repository index data for later use."""
        with self._lock:
            self.index_file.write_text(json.dumps(index_data, indent=2, default=str))

        # Also save a per-repo copy (best-effort).
        try:
            repo_path_str = index_data.get("repo_path")
            if repo_path_str:
                repo_path = Path(str(repo_path_str)).resolve()
                branch = str(index_data.get("branch") or "") or None
                per_repo = self._index_file_for_repo(repo_path, branch)
                with self._lock:
                    per_repo.write_text(json.dumps(index_data, indent=2, default=str))

                # Back-compat: also write the repo-only key (older caches).
                legacy = self._index_file_for_repo(repo_path, None)
                if legacy != per_repo:
                    with self._lock:
                        legacy.write_text(json.dumps(index_data, indent=2, default=str))
        except Exception:
            # Best-effort only; don't fail indexing.
            pass

    def load_repository_index(self) -> Dict:
        """Load the most recent repository index data."""
        if not self.index_file.exists():
            raise ValueError("No repository index found. Please run 'index' command first.")
        with self._lock:
            return json.loads(self.index_file.read_text())

    def load_repository_index_for_repo(self, repo_path: Path, branch: str | None = None) -> Dict:
        """
        Load a cached repository index for a specific repo path (+ optional branch).
        """
        # Prefer branch-specific cache when branch is provided.
        candidates: list[Path] = []
        if branch:
            candidates.append(self._index_file_for_repo(repo_path, branch))
        # Fallback to repo-only cache.
        candidates.append(self._index_file_for_repo(repo_path, None))

        for file_path in candidates:
            if file_path.exists():
                with self._lock:
                    return json.loads(file_path.read_text())

        raise ValueError(f"No repository index found for: {repo_path}{f' ({branch})' if branch else ''}")
