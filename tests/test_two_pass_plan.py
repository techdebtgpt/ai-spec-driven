from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from spec_agent.config.settings import AgentSettings
from spec_agent.domain.models import Plan, PlanStep, Task, TaskStatus
from spec_agent.workflow.orchestrator import TaskOrchestrator


@pytest.fixture()
def orchestrator(tmp_path: Path) -> TaskOrchestrator:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    settings = AgentSettings(state_dir=state_dir)
    orch = TaskOrchestrator(settings=settings)
    return orch


def test_generate_plan_builds_frozen_plan_target_scope_and_is_preliminary(orchestrator: TaskOrchestrator, tmp_path: Path) -> None:
    # Repo skeleton with a resolvable target directory.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("print('a')\n")

    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="desc",
        status=TaskStatus.CLARIFYING,
        metadata={"repository_summary": {"file_count": 1, "has_tests": True}},
    )
    orchestrator.store.upsert_task(task)

    # Planning returns a plan that includes a resolvable "src" target.
    first_plan = Plan(id="plan-1", task_id=task.id, steps=[PlanStep(description="step", target_files=["src"])])

    build_plan_mock: Any = MagicMock(side_effect=[first_plan])
    orchestrator.plan_builder.build_plan = build_plan_mock

    # Boundary and refactor can be cheap stubs.
    orchestrator.refactor_advisor.suggest = MagicMock(return_value=[])
    orchestrator.boundary_manager = None  # not used here

    # Force boundary manager required_specs to return empty list on both passes.
    from spec_agent.services.specs.boundary_manager import BoundaryManager

    original_required_specs = BoundaryManager.required_specs
    BoundaryManager.required_specs = MagicMock(return_value=[])
    try:
        orchestrator.generate_plan(task.id)
    finally:
        BoundaryManager.required_specs = original_required_specs

    task_after = orchestrator._get_task(task.id)
    assert (task_after.metadata.get("plan_stage") or "").upper() == "PRELIMINARY"
    assert task_after.metadata.get("plan_preview", {}).get("id") == "plan-1"
    # Frozen scope from plan targets should exist and include a non-empty allowlist.
    plan_targets_scope = ((task_after.metadata.get("bounded_context") or {}).get("plan_targets") or {}).get("scope") or {}
    assert plan_targets_scope.get("frozen") is True
    assert (plan_targets_scope.get("allowed_files") or [])  # non-empty


def test_finalize_plan_creates_final_plan_and_preserves_preliminary(orchestrator: TaskOrchestrator, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("print('a')\n")

    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="desc",
        status=TaskStatus.CLARIFYING,
        metadata={"repository_summary": {"file_count": 1, "has_tests": True}},
    )
    orchestrator.store.upsert_task(task)

    # First plan produces plan_targets scope; second plan is the final rebuild.
    first_plan = Plan(id="plan-1", task_id=task.id, steps=[PlanStep(description="step", target_files=["src"])])
    final_plan = Plan(id="plan-2", task_id=task.id, steps=[PlanStep(description="final", target_files=["src"])])
    orchestrator.plan_builder.build_plan = MagicMock(side_effect=[first_plan, final_plan])
    orchestrator.refactor_advisor.suggest = MagicMock(return_value=[])

    from spec_agent.services.specs.boundary_manager import BoundaryManager

    original_required_specs = BoundaryManager.required_specs
    BoundaryManager.required_specs = MagicMock(return_value=[])
    try:
        orchestrator.generate_plan(task.id)
        orchestrator.build_final_plan_with_frozen_scope(task.id)
    finally:
        BoundaryManager.required_specs = original_required_specs

    task_after = orchestrator._get_task(task.id)
    assert (task_after.metadata.get("plan_stage") or "").upper() == "FINAL"
    assert task_after.metadata.get("plan_preview", {}).get("id") == "plan-2"
    assert task_after.metadata.get("plan_preview_preliminary") is not None

