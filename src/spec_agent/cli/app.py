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


@app.command()
def specs(task_id: str = typer.Argument(..., help="UUID of the task.")) -> None:
    """
    View detailed boundary specifications for a task.
    """
    orchestrator = _get_orchestrator()
    specs = orchestrator.get_boundary_specs(task_id)

    if not specs:
        console.print("[yellow]No boundary specs found for this task.[/]")
        return

    for i, spec in enumerate(specs, 1):
        status_color = {
            "PENDING": "yellow",
            "APPROVED": "green",
            "SKIPPED": "blue"
        }.get(spec.get("status", "PENDING"), "white")

        console.print(f"\n[bold]{'='*70}[/]")
        console.print(f"[bold]Boundary Spec #{i}: {spec.get('boundary_name')} [[{status_color}]{spec.get('status', 'PENDING')}[/]]")
        console.print(f"[bold]{'='*70}[/]\n")
        console.print(f"[bold]ID:[/] {spec.get('id')}\n")
        console.print(f"[bold cyan]Description:[/]")
        console.print(f"{spec.get('human_description', 'No description provided.')}\n")
        console.print(f"[bold cyan]Mermaid Diagram:[/]")
        console.print(f"[dim]{spec.get('diagram_text', 'No diagram provided.')}[/]\n")
        console.print(f"[bold cyan]Machine Spec:[/]")

        machine_spec = spec.get('machine_spec', {})
        console.print(f"[bold]Actors:[/] {', '.join(machine_spec.get('actors', []))}")
        console.print(f"[bold]Interfaces:[/]")
        for interface in machine_spec.get('interfaces', []):
            console.print(f"  - {interface}")
        console.print(f"[bold]Invariants:[/]")
        for invariant in machine_spec.get('invariants', []):
            console.print(f"  - {invariant}")

    # Show summary at the end
    console.print(f"\n[bold]{'='*70}[/]")
    pending = [s for s in specs if s.get("status") == "PENDING"]
    if pending:
        console.print(f"\n[yellow]{len(pending)} spec(s) require approval.[/]")
        console.print(f"[dim]Approve: ./spec-agent approve-spec {task_id} <spec-id>[/]")
        console.print(f"[dim]Skip: ./spec-agent skip-spec {task_id} <spec-id>[/]")
    else:
        console.print(f"\n[green]All specs resolved.[/]")


@app.command()
def approve_spec(
    task_id: str = typer.Argument(..., help="UUID of the task."),
    spec_id: str = typer.Argument(..., help="ID of the boundary spec to approve."),
) -> None:
    """
    Approve a boundary specification.
    """
    orchestrator = _get_orchestrator()

    try:
        result = orchestrator.approve_spec(task_id, spec_id)
        console.print(f"[green]Approved boundary spec: {result['spec_id']}[/]")

        # Check if all specs are resolved
        specs = orchestrator.get_boundary_specs(task_id)
        pending = [s for s in specs if s.get("status") == "PENDING"]
        if not pending:
            console.print(f"\n[bold green]All boundary specs resolved![/] Ready to generate patches.")
        else:
            console.print(f"\n[yellow]{len(pending)} spec(s) still pending approval.[/]")
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/]")
        raise typer.Exit(code=1)


@app.command()
def skip_spec(
    task_id: str = typer.Argument(..., help="UUID of the task."),
    spec_id: str = typer.Argument(..., help="ID of the boundary spec to skip."),
) -> None:
    """
    Skip (override) a boundary specification.
    """
    orchestrator = _get_orchestrator()

    try:
        result = orchestrator.skip_spec(task_id, spec_id)
        console.print(f"[blue]Skipped boundary spec: {result['spec_id']}[/]")

        # Check if all specs are resolved
        specs = orchestrator.get_boundary_specs(task_id)
        pending = [s for s in specs if s.get("status") == "PENDING"]
        if not pending:
            console.print(f"\n[bold green]All boundary specs resolved![/] Ready to generate patches.")
        else:
            console.print(f"\n[yellow]{len(pending)} spec(s) still pending approval.[/]")
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/]")
        raise typer.Exit(code=1)


def main() -> None:
    app()


