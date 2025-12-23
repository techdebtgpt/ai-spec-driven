from __future__ import annotations

from pathlib import Path

import pytest

from spec_agent.domain.models import Task, TaskStatus
from spec_agent.workflow.orchestrator import TaskOrchestrator


@pytest.fixture()
def orchestrator(tmp_path: Path) -> TaskOrchestrator:
    orch = TaskOrchestrator()
    # Redirect state to temp directory so tests don't touch user state.
    orch.store.root = tmp_path / "state"
    orch.store.root.mkdir(exist_ok=True)
    return orch


def test_export_approved_plan_markdown_creates_docs_and_includes_scope(orchestrator: TaskOrchestrator, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")

    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="Change something important",
        status=TaskStatus.IMPLEMENTING,
        metadata={
            "plan_approved": True,
            "plan_approved_at": "2025-12-17T00:00:00Z",
            "plan_preview": {
                "id": "plan-1",
                "steps": [
                    {"description": "Update API handler", "target_files": ["src/api.py"], "notes": "Keep behavior same"},
                    {"description": "Add tests", "target_files": ["tests/test_api.py"], "notes": ""},
                ],
                "risks": ["Breaking backwards compatibility"],
                "refactors": ["Extract shared validation"],
            },
            "repository_summary": {"file_count": 100, "has_tests": True},
            "bounded_context": {"manual": {"aggregate": {"file_count": 7}, "targets": {"src/": {"file_count": 7}}}},
            "boundary_specs": [],
        },
    )
    orchestrator.store.upsert_task(task)

    out_path = orchestrator.export_approved_plan_markdown(task.id)
    assert out_path.exists()
    assert out_path.parent == repo / "docs" / "plans"
    assert out_path.name.startswith("change-something-important")
    assert out_path.suffix == ".md"

    contents = out_path.read_text(encoding="utf-8")
    assert "## Plan steps" in contents
    assert "1. Update API handler" in contents
    assert "Total files (scoped)" in contents

