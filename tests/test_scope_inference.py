from __future__ import annotations

from pathlib import Path

import pytest

from spec_agent.config.settings import AgentSettings
from spec_agent.domain.models import ClarificationStatus, Task, TaskStatus
from spec_agent.workflow.orchestrator import TaskOrchestrator


@pytest.fixture()
def orchestrator(tmp_path: Path) -> TaskOrchestrator:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    settings = AgentSettings(state_dir=state_dir)
    return TaskOrchestrator(settings=settings)


def _setup_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")

    target_dir = repo / "dbt_data_transformer" / "models" / "marts" / "reporting"
    target_dir.mkdir(parents=True)
    target_file = target_dir / "reporting_fynapse_transaction_fact.sql"
    target_file.write_text("select 1\n", encoding="utf-8")
    return repo, target_file


def test_scope_extraction_detects_short_identifiers(orchestrator: TaskOrchestrator, tmp_path: Path) -> None:
    repo, target_file = _setup_repo(tmp_path)

    clarifications = [
        {
            "id": "c1",
            "question": "Which modules are in scope?",
            "answer": "Focus on the Fynapse transaction_fact audit models.",
            "status": "ANSWERED",
        }
    ]

    task = Task(
        id="task-1",
        repo_path=repo,
        branch="main",
        description="Add audit for Fynapse transaction fact",
        status=TaskStatus.CLARIFYING,
        metadata={
            "repository_summary": {},  # minimal summary; fallback search should still resolve files
            "clarifications": clarifications,
        },
    )
    orchestrator.store.upsert_task(task)

    scope_candidates = orchestrator._extract_scope_from_clarifications(task.id)

    rel_target = str(target_file.relative_to(repo)).replace("\\", "/")
    assert rel_target in scope_candidates


def test_scope_extraction_uses_description_when_answers_unknown(orchestrator: TaskOrchestrator, tmp_path: Path) -> None:
    repo, target_file = _setup_repo(tmp_path)

    clarifications = [
        {
            "id": "c1",
            "question": "Which modules are in scope?",
            "answer": "I don't know",
            "status": "ANSWERED",
        }
    ]

    task = Task(
        id="task-2",
        repo_path=repo,
        branch="main",
        description="Add audit for Fynapse transaction fact",
        status=TaskStatus.CLARIFYING,
        metadata={
            "repository_summary": {},
            "clarifications": clarifications,
        },
    )
    orchestrator.store.upsert_task(task)

    scope_candidates = orchestrator._extract_scope_from_clarifications(task.id)

    rel_target = str(target_file.relative_to(repo)).replace("\\", "/")
    assert rel_target in scope_candidates


def test_update_clarification_records_scope_suggestions(orchestrator: TaskOrchestrator, tmp_path: Path) -> None:
    repo, target_file = _setup_repo(tmp_path)

    clarifications = [
        {
            "id": "c1",
            "question": "Which modules are in scope?",
            "answer": "",
            "status": "PENDING",
        }
    ]

    task = Task(
        id="task-3",
        repo_path=repo,
        branch="main",
        description="Add audit for Fynapse transaction fact",
        status=TaskStatus.CLARIFYING,
        metadata={
            "repository_summary": {},
            "clarifications": clarifications,
        },
    )
    orchestrator.store.upsert_task(task)

    updated = orchestrator.update_clarification(
        task.id,
        "c1",
        answer="I don't know",
        status=ClarificationStatus.ANSWERED,
    )

    suggestions = updated.get("auto_scope_suggestions") or []
    rel_target = str(target_file.relative_to(repo)).replace("\\", "/")
    assert rel_target in suggestions


def test_scope_extraction_handles_directory_keywords(orchestrator: TaskOrchestrator, tmp_path: Path) -> None:
    repo, target_file = _setup_repo(tmp_path)

    clarifications = [
        {
            "id": "c1",
            "question": "Which modules?",
            "answer": "dbtdatatransformer modules utilities audit area",
            "status": "ANSWERED",
        }
    ]

    task = Task(
        id="task-4",
        repo_path=repo,
        branch="main",
        description="Verify fynapse transaction audit implementation",
        status=TaskStatus.CLARIFYING,
        metadata={
            "repository_summary": {},
            "clarifications": clarifications,
        },
    )
    orchestrator.store.upsert_task(task)

    scope_candidates = orchestrator._extract_scope_from_clarifications(task.id)

    rel_target = str(target_file.relative_to(repo)).replace("\\", "/")
    assert rel_target in scope_candidates
