from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
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
def index(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, resolve_path=True),
    branch: str = typer.Option("main", "--branch", "-b"),
) -> None:
    """
    Index a repository and save the context for later use.
    """

    orchestrator = _get_orchestrator()
    index_data = orchestrator.index_repository(repo_path=repo, branch=branch)

    summary = index_data.get("repository_summary", {})
    git_info = index_data.get("git_info", {})

    console.print(f"[bold green]Repository indexed successfully[/]\n")
    
    # Merged Repository Info and Semantic Analysis Panel
    info_lines = []
    
    # Basic repository information
    info_lines.append(f"[bold cyan]Repository:[/] {index_data.get('repo_name', repo.name)}")
    info_lines.append(f"[bold cyan]Path:[/] {repo}")
    info_lines.append(f"[bold cyan]Branch:[/] {branch}")
    
    # Git information
    if git_info.get("current_commit"):
        info_lines.append(f"[bold cyan]Commit:[/] {git_info['current_commit'][:12]}")
    if git_info.get("commit_message"):
        commit_msg = git_info['commit_message']
        if len(commit_msg) > 60:
            commit_msg = commit_msg[:60] + "..."
        info_lines.append(f"[bold cyan]Message:[/] {commit_msg}")
    if git_info.get("commit_author"):
        info_lines.append(f"[bold cyan]Author:[/] {git_info['commit_author']}")
    if git_info.get("remote_url"):
        remote_url = git_info['remote_url']
        if len(remote_url) > 60:
            remote_url = remote_url[:57] + "..."
        info_lines.append(f"[bold cyan]Remote:[/] {remote_url}")
    
    # Semantic index information
    semantic_index = index_data.get('semantic_index')
    if semantic_index:
        repo_info = semantic_index.get('repository', {})
        
        # Add separator
        info_lines.append("")
        
        # Primary languages
        primary_languages = repo_info.get('primaryLanguages', [])
        if primary_languages:
            lang_str = ', '.join(primary_languages[:5])  # Limit to 5 languages
            info_lines.append(f"[bold cyan]Languages:[/] {lang_str}")
        
        # Frameworks
        frameworks = repo_info.get('frameworks', [])
        if frameworks:
            fw_str = ', '.join(frameworks[:5])  # Limit to 5 frameworks
            info_lines.append(f"[bold cyan]Frameworks:[/] {fw_str}")
        
        # Architecture style
        if repo_info.get('architectureStyle'):
            info_lines.append(f"[bold cyan]Architecture:[/] {repo_info['architectureStyle']}")
        
        # Add another separator before counts
        info_lines.append("")
        
        # Module and domain counts
        structure = semantic_index.get('structure', {})
        modules = structure.get('modules', [])
        domains = semantic_index.get('domains', [])
        
        if modules:
            info_lines.append(f"[bold cyan]Modules:[/] {len(modules)} detected")
        if domains:
            info_lines.append(f"[bold cyan]Domains:[/] {len(domains)} identified")
        
        # Public interfaces
        public_interfaces = semantic_index.get('publicInterfaces', {})
        http_apis = public_interfaces.get('httpApis', [])
        cli_commands = public_interfaces.get('cliCommands', [])
        events = public_interfaces.get('events', [])
        
        interface_parts = []
        if http_apis:
            interface_parts.append(f"{len(http_apis)} HTTP API{'s' if len(http_apis) != 1 else ''}")
        if cli_commands:
            interface_parts.append(f"{len(cli_commands)} CLI command{'s' if len(cli_commands) != 1 else ''}")
        if events:
            interface_parts.append(f"{len(events)} event{'s' if len(events) != 1 else ''}")
        
        if interface_parts:
            info_lines.append(f"[bold cyan]Public Interfaces:[/] {', '.join(interface_parts)}")
        
        # Key components and integrations
        key_components = semantic_index.get('keyComponents', [])
        external_integrations = semantic_index.get('externalIntegrations', [])
        
        if key_components:
            info_lines.append(f"[bold cyan]Key Components:[/] {len(key_components)}")
        if external_integrations:
            info_lines.append(f"[bold cyan]External Integrations:[/] {len(external_integrations)}")
    
    console.print(Panel.fit("\n".join(info_lines), title="Repository Overview", border_style="bright_blue"))
    
    # Serena Status
    if summary.get('serena_enabled'):
        console.print("[dim]Enhanced with Serena language detection[/]")
    else:
        console.print("[dim]Basic language detection (Serena not enabled)[/]")
    
    console.print(f"\n[cyan]Index saved. You can now run:[/] [bold]./spec-agent start --description \"Your task\"[/]")



@app.command()
def start(
    description: str = typer.Option(..., "--description", "-d", prompt=True),
) -> None:
    """
    Create a new task using the previously indexed repository.
    """

    orchestrator = _get_orchestrator()
    task = orchestrator.create_task_from_index(description=description)

    clarifications = task.metadata.get("clarifications", [])

    console.print(f"[bold green]Created task[/] {task.id}")

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
def plan(
    task_id: str = typer.Argument(..., help="UUID of the task to plan."),
    fast: bool = typer.Option(False, "--fast", help="Skip rationale enhancement for faster execution"),
) -> None:
    """
    Generate a plan, boundary specs, patch queue, and test suggestions for a task.
    
    Use --fast to skip rationale enhancement (Epic 4.1) for faster execution.
    """

    orchestrator = _get_orchestrator()
    
    # Show progress
    console.print("[cyan]Generating plan...[/]")
    if fast:
        console.print("[yellow]Fast mode: Skipping rationale enhancement[/]")
    
    payload = orchestrator.generate_plan(task_id, skip_rationale_enhancement=fast)

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

    # Track progress through patches
    patch_index = 0
    total_pending = len(pending_patches)
    
    while True:
        patch = orchestrator.get_next_pending_patch(task_id)
        if not patch:
            console.print("[green]No pending patches remain.[/]")
            break

        patch_index += 1
        
        # Show progress indicator
        console.print(f"\n[bold cyan]{'='*70}[/]")
        console.print(f"[bold cyan]Patch {patch_index} of {total_pending}[/]")
        console.print(f"[bold cyan]{'='*70}[/]\n")

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
        
        # Improved prompt that accepts 'a', 'A', 'approve', etc.
        choice = typer.prompt(
            f"[bold]Patch {patch_index}/{total_pending}:[/] Approve (a), Reject (r), or Skip (s)",
            default="s"
        ).strip().lower()

        # More lenient matching - accept 'a', 'approve', 'y', 'yes'
        if choice in ['a', 'approve', 'y', 'yes', 'ok']:
            try:
                orchestrator.approve_patch(task_id, patch.id)
                console.print(f"[green]✓ Patch {patch_index}/{total_pending} applied to working tree.[/]")
                if patch_index < total_pending:
                    console.print(f"[dim]Continuing to next patch...[/]\n")
            except Exception as exc:  # pragma: no cover - CLI guardrail
                console.print(f"[red]Failed to apply patch:[/] {exc}")
                # Ask if they want to continue or stop
                continue_choice = typer.prompt(
                    "Continue with remaining patches? (y/n)",
                    default="n"
                ).strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
        elif choice in ['r', 'reject', 'n', 'no']:
            try:
                orchestrator.reject_patch(task_id, patch.id)
                console.print("[yellow]Patch rejected. Plan regenerated; review new queue when ready.[/]")
            except Exception as exc:  # pragma: no cover - CLI guardrail
                console.print(f"[red]Failed to reject patch:[/] {exc}")
            break  # regenerate kicks off a new queue; exit loop
        else:
            console.print("[cyan]Leaving remaining patches pending.[/]")
            break


@app.command("ask-patch")
def ask_patch_question(
    task_id: str = typer.Argument(..., help="UUID of the task."),
    patch_id: str = typer.Argument(..., help="UUID of the patch to ask about."),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Question to ask. If not provided, will prompt."),
) -> None:
    """
    Ask a follow-up question about a patch's rationale.
    
    Epic 4.1: Rationale-Based Code Review - Support for follow-up questions.
    """
    orchestrator = _get_orchestrator()
    
    # Get the patch
    patches = orchestrator.list_patches(task_id)
    patch = next((p for p in patches if p.id == patch_id), None)
    
    if not patch:
        console.print(f"[red]Patch {patch_id} not found for task {task_id}.[/]")
        raise typer.Exit(code=1)
    
    # Get the plan step
    task = orchestrator._get_task(task_id)
    plan_preview = task.metadata.get("plan_preview", {})
    plan_steps = plan_preview.get("steps", [])
    
    # Find the step that matches this patch
    plan_step = None
    for step_desc in plan_steps:
        if step_desc == patch.step_reference:
            # Create a minimal PlanStep for the rationale enhancer
            from ..domain.models import PlanStep, Plan
            plan_step = PlanStep(description=step_desc)
            break
    
    if not plan_step:
        plan_step = PlanStep(description=patch.step_reference)
    
    # Create a minimal Plan for context
    from ..domain.models import Plan
    plan = Plan(
        id=task.id,
        task_id=task.id,
        steps=[plan_step],
        risks=plan_preview.get("risks", []),
    )
    
    # Prompt for question if not provided
    if not question:
        console.print(Panel.fit(patch.rationale, title=f"Patch {patch_id[:8]} Rationale"))
        console.print("\n[cyan]Ask a follow-up question about this patch's rationale.[/]")
        question = typer.prompt("Question")
    
    if not question.strip():
        console.print("[yellow]No question provided.[/]")
        return
    
    # Get answer from rationale enhancer
    try:
        from ..services.review.rationale_enhancer import RationaleEnhancer
        enhancer = RationaleEnhancer(llm_client=orchestrator.llm_client)
        answer = enhancer.answer_followup_question(
            patch=patch,
            question=question,
            plan_step=plan_step,
            plan=plan,
        )
        
        console.print(Panel.fit(answer, title="Answer"))
        
        # Store the question and answer in task history
        orchestrator.logger.record(
            task_id,
            "PATCH_QUESTION_ASKED",
            {
                "patch_id": patch_id,
                "question": question,
                "answer": answer,
            },
        )
        
    except Exception as exc:
        console.print(f"[red]Error generating answer:[/] {exc}")
        console.print("[yellow]LLM client may not be configured. Set SPEC_AGENT_OPENAI_API_KEY.[/]")
        raise typer.Exit(code=1)


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


@app.command("clean-logs")
def clean_logs(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    tasks: bool = typer.Option(False, "--tasks", help="Also delete all tasks (not just logs)"),
) -> None:
    """
    Clear all reasoning logs. Optionally clear tasks as well.
    
    Logs are stored in ~/.spec_agent/logs.json
    Tasks are stored in ~/.spec_agent/tasks.json
    """
    from ..config.settings import get_settings
    from ..persistence.store import JsonStore
    
    settings = get_settings()
    store = JsonStore(settings.state_dir)
    
    logs_file = store.logs_file
    tasks_file = store.tasks_file
    
    # Count existing logs/tasks
    existing_logs = store.load_logs()
    existing_tasks = store.load_tasks()
    
    if not existing_logs and not (tasks and existing_tasks):
        console.print("[yellow]No logs to clean.[/]")
        return
    
    # Show what will be deleted
    if existing_logs:
        console.print(f"[yellow]Found {len(existing_logs)} log entries[/]")
    if tasks and existing_tasks:
        console.print(f"[yellow]Found {len(existing_tasks)} tasks[/]")
    
    # Confirm deletion
    if not confirm:
        if tasks:
            response = typer.prompt(
                f"Delete all logs ({len(existing_logs)} entries) and tasks ({len(existing_tasks)} tasks)? [y/N]",
                default="n"
            )
        else:
            response = typer.prompt(
                f"Delete all logs ({len(existing_logs)} entries)? [y/N]",
                default="n"
            )
        
        if response.lower() not in {"y", "yes"}:
            console.print("[cyan]Cancelled.[/]")
            return
    
    # Delete logs
    try:
        if logs_file.exists():
            logs_file.unlink()
            console.print("[green]✓ Cleared all reasoning logs[/]")
        else:
            console.print("[yellow]No logs file found.[/]")
    except Exception as exc:
        console.print(f"[red]Error deleting logs: {exc}[/]")
        raise typer.Exit(code=1)
    
    # Delete tasks if requested
    if tasks:
        try:
            if tasks_file.exists():
                tasks_file.unlink()
                console.print("[green]✓ Cleared all tasks[/]")
            else:
                console.print("[yellow]No tasks file found.[/]")
        except Exception as exc:
            console.print(f"[red]Error deleting tasks: {exc}[/]")
            raise typer.Exit(code=1)
    
    console.print("[bold green]Cleanup complete![/]")


def main() -> None:
    app()


