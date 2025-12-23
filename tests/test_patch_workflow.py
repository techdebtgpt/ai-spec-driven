from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from spec_agent.domain.models import Patch, PatchKind, PatchStatus, Plan, PlanStep, Task, TaskStatus
from spec_agent.services.patches.engine import PatchEngine
from spec_agent.workflow.orchestrator import TaskOrchestrator


@pytest.fixture()
def orchestrator(tmp_path: Path, monkeypatch: Any) -> TaskOrchestrator:
    store_dir = tmp_path / "state"
    store_dir.mkdir()

    orch = TaskOrchestrator()
    orch.store.root = store_dir
    return orch


def make_task(tmp_path: Path) -> Task:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    return Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="Epic3",
        status=TaskStatus.PLANNING,
    )


def make_patch(kind: PatchKind = PatchKind.IMPLEMENTATION, status: PatchStatus = PatchStatus.PENDING) -> Patch:
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        steps=[PlanStep(description="do work")],
    )
    engine = PatchEngine()
    patch = engine._placeholder_patch(plan, 1, "do work", kind=kind)
    patch.status = status
    return patch


@patch.object(subprocess, "run")
def test_has_manual_edits_detects_manual_changes(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value.stdout = "M new_file.py"
    orch = TaskOrchestrator()
    orch.store.root = tmp_path / "state"
    orch.store.root.mkdir(exist_ok=True)
    task = make_task(tmp_path)
    task.metadata["patch_queue_state"] = [make_patch().to_dict()]
    task.metadata["worktree_status"] = "A test.py"
    orch.store.upsert_task(task)

    assert orch.has_manual_edits(task.id)
    # Listing patches should still return the cached queue
    patches = orch.list_patches(task.id)
    assert patches


@patch.object(subprocess, "run")
def test_approve_patch_applies_diff(mock_run: MagicMock, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="desc",
        status=TaskStatus.PLANNING,
        metadata={"patch_queue_state": [make_patch().to_dict()]},
    )
    orch = TaskOrchestrator()
    orch.store.root = tmp_path / "state"
    orch.store.root.mkdir(exist_ok=True)
    orch.store.upsert_task(task)

    mock_run.side_effect = [
        MagicMock(stdout="main"),  # ensure branch
        MagicMock(stdout=""),  # git apply --3way
        MagicMock(stdout=""),  # git apply --ignore-space-change
        MagicMock(stdout=""),  # git apply fallback
        MagicMock(stdout=""),  # git status snapshot
        MagicMock(stdout="deadbeef"),  # commit snapshot
    ]

    patch = orch.approve_patch(task.id, orch.list_patches(task.id)[0].id)
    assert patch.status == PatchStatus.APPLIED


@patch.object(subprocess, "run")
def test_reject_patch_marks_status_and_regenerates(mock_run: MagicMock, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="desc",
        status=TaskStatus.PLANNING,
        metadata={"patch_queue_state": [make_patch().to_dict()]},
    )
    orch = TaskOrchestrator()
    orch.store.root = tmp_path / "state"
    orch.store.root.mkdir(exist_ok=True)
    orch.store.upsert_task(task)

    mock_run.side_effect = [
        MagicMock(stdout=""),
        MagicMock(stdout="deadbeef"),
    ]

    orch.reject_patch(task.id, orch.list_patches(task.id)[0].id)
    task_after = orch._get_task(task.id)
    assert "patch_queue_state" not in task_after.metadata


def test_restart_clarifications_preserves_manual_bounded_context(tmp_path: Path) -> None:
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
            "repository_summary": {"file_count": 100, "has_tests": True},
            "bounded_context": {
                "manual": {"aggregate": {"file_count": 7}},
                "plan_targets": {"aggregate": {"file_count": 42}},
            },
        },
    )
    orch = TaskOrchestrator()
    orch.store.root = tmp_path / "state"
    orch.store.root.mkdir(exist_ok=True)
    orch.store.upsert_task(task)

    with patch.object(orch.clarifier, "generate_questions", return_value=[]):
        orch.restart_clarifications(task.id, reason="scope was too broad")

    task_after = orch._get_task(task.id)
    bounded = task_after.metadata.get("bounded_context") or {}
    assert "manual" in bounded
    assert bounded["manual"]["aggregate"]["file_count"] == 7
    assert "plan_targets" not in bounded
