from __future__ import annotations

from pathlib import Path

from spec_agent.domain.models import Task, TaskStatus
from spec_agent.workflow.orchestrator import TaskOrchestrator


def test_bounded_index_task_does_not_persist_empty_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")

    orch = TaskOrchestrator()
    orch.store.root = tmp_path / "state"
    orch.store.root.mkdir(exist_ok=True)

    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="desc",
        status=TaskStatus.PLANNING,
        metadata={"repository_summary": {"file_count": 10}},
    )
    orch.store.upsert_task(task)

    # Target does not exist -> summarize_targets returns empty aggregate/file_count=0
    summary = orch.bounded_index_task(task.id, [";"])
    assert (summary.get("aggregate") or {}).get("file_count", 0) == 0

    task_after = orch._get_task(task.id)
    assert "bounded_context" not in task_after.metadata or "manual" not in (task_after.metadata.get("bounded_context") or {})

