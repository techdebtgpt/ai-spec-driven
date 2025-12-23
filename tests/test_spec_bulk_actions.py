from __future__ import annotations

from pathlib import Path

from spec_agent.config.settings import AgentSettings
from spec_agent.domain.models import Task, TaskStatus
from spec_agent.workflow.orchestrator import TaskOrchestrator


def test_approve_all_and_skip_all_specs(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    orch = TaskOrchestrator(settings=AgentSettings(state_dir=state_dir))

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")

    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="desc",
        status=TaskStatus.PLANNING,
        metadata={
            "boundary_specs": [
                {"id": "s1", "boundary_name": "A", "status": "PENDING"},
                {"id": "s2", "boundary_name": "B", "status": "PENDING"},
            ]
        },
    )
    orch.store.upsert_task(task)

    res = orch.approve_all_specs(task.id)
    assert res["approved_count"] == 2
    after = orch._get_task(task.id)
    assert all(s.get("status") == "APPROVED" for s in after.metadata.get("boundary_specs", []))

    # Reset to pending and test skip_all
    after.metadata["boundary_specs"][0]["status"] = "PENDING"
    after.metadata["boundary_specs"][1]["status"] = "PENDING"
    orch.store.upsert_task(after)
    res2 = orch.skip_all_specs(task.id)
    assert res2["skipped_count"] == 2
    after2 = orch._get_task(task.id)
    assert all(s.get("status") == "SKIPPED" for s in after2.metadata.get("boundary_specs", []))

