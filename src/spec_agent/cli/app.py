from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..domain.models import TaskStatus
from ..workflow.orchestrator import TaskOrchestrator

app = typer.Typer(add_completion=False, help="Spec-driven development agent CLI.")
console = Console()


def _get_orchestrator() -> TaskOrchestrator:
    return TaskOrchestrator()


@app.command()
def start(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, resolve_path=True),
    branch: str = typer.Option("main", "--branch", "-b"),
    description: str = typer.Option(..., "--description", "-d", prompt=True),
) -> None:
    """
    Create a new task and kick off contextual analysis + clarification.
    """

    orchestrator = _get_orchestrator()
    task = orchestrator.create_task(repo_path=repo, branch=branch, description=description)

    summary = task.metadata.get("repository_summary", {})
    clarifications = task.metadata.get("clarifications", [])

    console.print(f"[bold green]Created task[/] {task.id}")
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Files: {summary.get('file_count', 0)}",
                    f"Directories: {summary.get('directory_count', 0)}",
                    f"Top languages: {', '.join(summary.get('top_languages', [])) or 'n/a'}",
                    f"Top modules: {', '.join(summary.get('top_modules', [])) or 'n/a'}",
                ]
            ),
            title="Repository Summary",
        )
    )

    if clarifications:
        table = Table(title="Clarification Questions")
        table.add_column("ID")
        table.add_column("Question")
        for item in clarifications:
            table.add_row(item["id"], item["question"])
        console.print(table)
    else:
        console.print("[cyan]No clarification questions detected.[/]")


@app.command("tasks")
def list_tasks(
    status: Optional[TaskStatus] = typer.Option(None, "--status", "-s", case_sensitive=False),
) -> None:
    """
    Show task history recorded in the local state directory.
    """

    orchestrator = _get_orchestrator()
    tasks = orchestrator.list_tasks(status=status)
    if not tasks:
        console.print("[yellow]No tasks recorded yet.[/]")
        return

    table = Table(title="Spec Agent Tasks")
    table.add_column("ID", style="bold", width=36, no_wrap=True)
    table.add_column("Repo")
    table.add_column("Branch")
    table.add_column("Status")
    table.add_column("Updated")

    for task in tasks:
        table.add_row(
            str(UUID(task.id)),
            str(task.repo_path),
            task.branch,
            task.status.value,
            task.updated_at.isoformat(timespec="seconds"),
        )

    console.print(table)


@app.command()
def plan(task_id: str = typer.Argument(..., help="UUID of the task to plan.")) -> None:
    """
    Generate a plan, boundary specs, patch queue, and test suggestions for a task.
    """

    orchestrator = _get_orchestrator()
    payload = orchestrator.generate_plan(task_id)

    console.print(Panel.fit("\n".join(payload["plan"]["steps"]), title="Plan Steps"))
    console.print(Panel.fit("\n".join(payload["plan"]["risks"]), title="Risks"))
    console.print(Panel.fit("\n".join(payload["plan"]["refactors"]), title="Refactor Suggestions"))

    if payload["pending_specs"]:
        console.print(Panel.fit("\n".join(payload["pending_specs"]), title="Pending Boundary Specs"))
    if payload["patch_queue"]:
        console.print(Panel.fit("\n".join(payload["patch_queue"]), title="Patch Queue"))
    if payload["test_suggestions"]:
        console.print(Panel.fit("\n".join(payload["test_suggestions"]), title="Test Suggestions"))


def main() -> None:
    app()


