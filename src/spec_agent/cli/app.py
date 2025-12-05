from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..domain.models import RefactorSuggestionStatus, TaskStatus
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


@app.command("patches")
def review_patches(
    task_id: str = typer.Argument(..., help="UUID of the task whose patches should be reviewed."),
    list_only: bool = typer.Option(False, "--list", help="Only list patches without prompting for approval."),
) -> None:
    """
    Inspect and approve/reject incremental patches for a task.
    """

    orchestrator = _get_orchestrator()
    patches = orchestrator.list_patches(task_id)
    if not patches:
        console.print("[yellow]No patches generated yet. Run 'spec-agent plan' first.[/]")
        return

    if list_only:
        table = Table(title="Patch Queue")
        table.add_column("ID", style="bold", width=10)
        table.add_column("Step Reference")
        table.add_column("Kind")
        table.add_column("Status")
        for patch in patches:
            table.add_row(patch.id[:8], patch.step_reference, patch.kind.value.title(), patch.status.value)
        console.print(table)
        return

    # Show summary of all pending patches and files that will be changed
    pending_patches = [p for p in patches if p.status.value == "PENDING"]
    if pending_patches:
        changed_files = set()
        for p in pending_patches:
            # Extract file paths from diff (lines starting with --- a/ or +++ b/)
            for line in p.diff.split('\n'):
                if line.startswith('--- a/') or line.startswith('+++ b/'):
                    # Remove the --- a/ or +++ b/ prefix and get the file path
                    parts = line.split()
                    if len(parts) > 1:
                        file_path = parts[1]
                    else:
                        file_path = line[6:]  # Remove "--- a/" or "+++ b/"
                    
                    # Clean up the path (remove a/ or b/ prefix if present)
                    if file_path.startswith('a/') or file_path.startswith('b/'):
                        file_path = file_path[2:]
                    
                    if file_path and file_path != '/dev/null' and not file_path.startswith('a/') and not file_path.startswith('b/'):
                        changed_files.add(file_path)
        
        if changed_files:
            console.print(Panel.fit(
                "\n".join(sorted(changed_files)),
                title=f"Files that will be changed ({len(pending_patches)} pending patch{'es' if len(pending_patches) != 1 else ''})"
            ))
            console.print("[cyan]Note: Patches are only applied to your repository when you approve them.[/]\n")

    while True:
        patch = orchestrator.get_next_pending_patch(task_id)
        if not patch:
            console.print("[green]No pending patches remain.[/]")
            break

        # Extract files changed in this specific patch
        patch_files = set()
        for line in patch.diff.split('\n'):
            if line.startswith('--- a/') or line.startswith('+++ b/'):
                # Remove the --- a/ or +++ b/ prefix and get the file path
                parts = line.split()
                if len(parts) > 1:
                    file_path = parts[1]
                else:
                    file_path = line[6:]  # Remove "--- a/" or "+++ b/"
                
                # Clean up the path (remove a/ or b/ prefix if present)
                if file_path.startswith('a/') or file_path.startswith('b/'):
                    file_path = file_path[2:]
                
                if file_path and file_path != '/dev/null' and not file_path.startswith('a/') and not file_path.startswith('b/'):
                    patch_files.add(file_path)
        
        if patch_files:
            console.print(f"[bold cyan]Files in this patch:[/] {', '.join(sorted(patch_files))}\n")

        # Analyze the diff to show what will actually change
        lines_added = sum(1 for line in patch.diff.split('\n') if line.startswith('+') and not line.startswith('+++'))
        lines_removed = sum(1 for line in patch.diff.split('\n') if line.startswith('-') and not line.startswith('---'))
        files_created = any('/dev/null' in line for line in patch.diff.split('\n') if line.startswith('---'))
        
        change_summary = []
        if lines_added > 0:
            change_summary.append(f"[green]+{lines_added} line{'s' if lines_added != 1 else ''} added[/]")
        if lines_removed > 0:
            change_summary.append(f"[red]-{lines_removed} line{'s' if lines_removed != 1 else ''} removed[/]")
        if files_created:
            change_summary.append("[cyan]new file(s)[/]")
        
        if change_summary:
            console.print(f"[bold]Change summary:[/] {' '.join(change_summary)}\n")

        console.print(Panel.fit(patch.diff, title=f"Patch {patch.id[:8]} ({patch.step_reference})"))
        console.print(Panel.fit(patch.rationale, title="Rationale"))
        console.print("[yellow]⚠️  This patch will be applied to your repository only if you approve it.[/]")
        console.print("[dim]Currently, nothing has changed in your repo. Approve to apply the changes above.[/]\n")
        choice = typer.prompt("Approve (a), Reject (r), or Skip (s)", default="s").strip().lower()

        if choice.startswith("a"):
            try:
                orchestrator.approve_patch(task_id, patch.id)
                console.print("[green]Patch applied to working tree.[/]")
            except Exception as exc:  # pragma: no cover - CLI guardrail
                console.print(f"[red]Failed to apply patch:[/] {exc}")
                break
        elif choice.startswith("r"):
            try:
                orchestrator.reject_patch(task_id, patch.id)
                console.print("[yellow]Patch rejected. Plan regenerated; review new queue when ready.[/]")
            except Exception as exc:  # pragma: no cover - CLI guardrail
                console.print(f"[red]Failed to reject patch:[/] {exc}")
            break  # regenerate kicks off a new queue; exit loop
        else:
            console.print("[cyan]Leaving remaining patches pending.[/]")
            break


@app.command("refactors")
def manage_refactors(
    task_id: str = typer.Argument(..., help="UUID of the task whose refactor suggestions should be reviewed."),
    list_only: bool = typer.Option(False, "--list", help="Only list suggestions."),
) -> None:
    """
    Inspect and accept/reject refactor suggestions. Accepted refactors enqueue new patches.
    """

    orchestrator = _get_orchestrator()
    suggestions = orchestrator.list_refactors(task_id)
    if not suggestions:
        console.print("[yellow]No refactor suggestions recorded for this task.[/]")
        return

    if list_only:
        table = Table(title="Refactor Suggestions")
        table.add_column("ID", style="bold", width=10)
        table.add_column("Description")
        table.add_column("Status")
        for suggestion in suggestions:
            table.add_row(suggestion.id[:8], suggestion.description, suggestion.status.value)
        console.print(table)
        return

    for suggestion in suggestions:
        if suggestion.status != RefactorSuggestionStatus.PENDING:
            continue
        console.print(Panel.fit(suggestion.description, title=f"Refactor {suggestion.id[:8]}"))
        console.print(Panel.fit(suggestion.rationale, title="Rationale"))
        if suggestion.scope:
            console.print(f"[cyan]Scope:[/] {', '.join(suggestion.scope)}")
        choice = typer.prompt("Approve (a), Reject (r), or Skip (s)", default="s").strip().lower()
        if choice.startswith("a"):
            orchestrator.approve_refactor(task_id, suggestion.id)
            console.print("[green]Refactor approved. Review newly enqueued patches.[/]")
            break
        elif choice.startswith("r"):
            orchestrator.reject_refactor(task_id, suggestion.id)
            console.print("[yellow]Refactor rejected.[/]")
        else:
            console.print("[cyan]Skipping remaining suggestions.[/]")
            break


@app.command("status")
def show_status(task_id: str = typer.Argument(..., help="UUID of the task to inspect.")) -> None:
    """
    Display branch/commit alignment and uncommitted changes for a task's working tree.
    """

    orchestrator = _get_orchestrator()
    summary = orchestrator.get_task_status(task_id)

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Repository: {summary['repo']}",
                    f"Branch: {summary['branch']} (expected {summary['expected_branch']})",
                    f"Last commit: {summary['last_commit']}",
                    "Patch counts: "
                    + ", ".join(f"{status}:{count}" for status, count in summary["patch_counts"].items()),
                ]
            ),
            title="Task Status",
        )
    )
    console.print(Panel.fit(summary["git_status"], title="git status --short"))


def main() -> None:
    app()


