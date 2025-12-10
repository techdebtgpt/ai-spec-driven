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


def _render_directory_tree(node: Dict, prefix: str = "", is_last: bool = True, max_items: int = 10) -> List[str]:
    """
    Render a directory tree structure as a list of formatted strings.
    
    Args:
        node: Directory node with children
        prefix: Current line prefix for indentation
        is_last: Whether this is the last item in the current level
        max_items: Maximum items to show per directory level
    
    Returns:
        List of formatted tree lines
    """
    lines = []
    
    if not node:
        return lines
    
    # Choose the right tree characters
    current_prefix = "└── " if is_last else "├── "
    next_prefix = "    " if is_last else "│   "
    
    # Format the node name with metadata
    name = node.get("name", "")
    node_type = node.get("type", "directory")
    
    if node_type == "directory":
        file_count = node.get("file_count", 0)
        dir_count = node.get("dir_count", 0)
        total_size_kb = node.get("total_size", 0) / 1024
        
        if node.get("depth", 0) == 0:
            # Root level
            meta = f"[bold cyan]{name}[/] [dim]({file_count} files, {dir_count} dirs, {total_size_kb:.1f} KB)[/]"
        else:
            meta = f"[cyan]{name}/[/] [dim]({file_count} files)[/]"
        
        lines.append(f"{prefix}{current_prefix}{meta}")
        
        # Render children
        children = node.get("children", [])
        # Limit children to avoid overwhelming output
        shown_children = children[:max_items]
        remaining = len(children) - len(shown_children)
        
        for i, child in enumerate(shown_children):
            is_last_child = (i == len(shown_children) - 1) and (remaining == 0)
            child_lines = _render_directory_tree(
                child, 
                prefix + next_prefix, 
                is_last_child,
                max_items
            )
            lines.extend(child_lines)
        
        if remaining > 0:
            lines.append(f"{prefix}{next_prefix}[dim]... and {remaining} more items[/]")
    else:
        # File
        size = node.get("size", 0)
        size_kb = size / 1024
        ext = node.get("extension", "")
        
        if size_kb > 1024:
            size_str = f"{size_kb/1024:.1f} MB"
        elif size_kb > 1:
            size_str = f"{size_kb:.1f} KB"
        else:
            size_str = f"{size} B"
        
        meta = f"{name} [dim]({size_str})[/]"
        lines.append(f"{prefix}{current_prefix}{meta}")
    
    return lines


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
    
    # Repository Overview Panel
    repo_info_lines = [
        f"[bold]Repository:[/] {index_data.get('repo_name', repo.name)}",
        f"[bold]Path:[/] {repo}",
        f"[bold]Branch:[/] {branch}",
    ]
    
    # Add git info if available
    if git_info.get("current_commit"):
        repo_info_lines.append(f"[bold]Commit:[/] {git_info['current_commit'][:12]}")
    if git_info.get("commit_message"):
        repo_info_lines.append(f"[bold]Message:[/] {git_info['commit_message'][:60]}...")
    if git_info.get("commit_author"):
        repo_info_lines.append(f"[bold]Author:[/] {git_info['commit_author']}")
    if git_info.get("remote_url"):
        remote_url = git_info['remote_url']
        if len(remote_url) > 60:
            remote_url = remote_url[:57] + "..."
        repo_info_lines.append(f"[bold]Remote:[/] {remote_url}")
    
    console.print(Panel.fit("\n".join(repo_info_lines), title="Repository Info", border_style="blue"))
    
    # Statistics Panel
    stats_lines = [
        f"[bold]Files:[/] {summary.get('file_count', 0):,}",
        f"[bold]Directories:[/] {summary.get('directory_count', 0):,}",
    ]
    
    if summary.get('total_size_mb'):
        stats_lines.append(f"[bold]Total Size:[/] {summary['total_size_mb']:,.2f} MB")
    
    # Show hotspots count if any
    hotspots = summary.get('hotspots', [])
    if hotspots:
        stats_lines.append(f"[bold]Large Files:[/] {len(hotspots)} (>{orchestrator.settings.hotspot_loc_threshold} LOC)")
    
    console.print(Panel.fit("\n".join(stats_lines), title="Statistics", border_style="cyan"))
    
    # Languages & Frameworks Panel
    lang_lines = []
    
    # Project type
    if summary.get('project_type'):
        lang_lines.append(f"[bold]Project Type:[/] {summary['project_type']}")
    
    # Top languages
    top_langs = summary.get('top_languages', [])
    if top_langs:
        lang_lines.append(f"[bold]Languages:[/] {', '.join(top_langs)}")
    
    # Frameworks
    frameworks = summary.get('frameworks', [])
    if frameworks:
        lang_lines.append(f"[bold]Frameworks:[/] {', '.join(frameworks[:5])}")
    
    # Language details
    lang_details = summary.get('language_details', [])
    if lang_details and len(lang_details) > 0:
        lang_lines.append("")
        lang_lines.append("[bold]Language Breakdown:[/]")
        for detail in lang_details[:5]:
            lang_name = detail.get('language', 'unknown')
            file_count = detail.get('file_count', 0)
            detected_by = detail.get('detected_by', '')
            if detected_by:
                lang_lines.append(f"  • {lang_name}: {file_count} files (via {detected_by})")
            else:
                lang_lines.append(f"  • {lang_name}: {file_count} files")
    
    if lang_lines:
        console.print(Panel.fit("\n".join(lang_lines), title="Languages & Frameworks", border_style="green"))
    
    # Structure Panel (Modules, Namespaces, Directories)
    structure_lines = []
    
    # Top modules
    top_modules = summary.get('top_modules', [])
    if top_modules:
        structure_lines.append("[bold]Top Modules:[/]")
        for mod in top_modules[:5]:
            structure_lines.append(f"  • {mod}")
    
    # Namespaces (for .NET projects)
    namespaces = summary.get('namespaces', [])
    if namespaces:
        if structure_lines:
            structure_lines.append("")
        structure_lines.append("[bold]Namespaces:[/]")
        for ns in namespaces[:5]:
            structure_lines.append(f"  • {ns}")
    
    # Top directories
    top_dirs = summary.get('top_directories', [])
    if top_dirs:
        if structure_lines:
            structure_lines.append("")
        structure_lines.append("[bold]Top Directories:[/]")
        for d in top_dirs[:5]:
            structure_lines.append(f"  • {d}")
    
    if structure_lines:
        console.print(Panel.fit("\n".join(structure_lines), title="🗂️  Project Structure", border_style="magenta"))
    
    # Directory Tree Panel
    directory_structure = summary.get('directory_structure')
    if directory_structure:
        tree_lines = _render_directory_tree(directory_structure, max_items=8)
        if tree_lines:
            # Limit total lines to avoid overwhelming output
            max_tree_lines = 30
            if len(tree_lines) > max_tree_lines:
                tree_lines = tree_lines[:max_tree_lines]
                tree_lines.append("[dim]... (tree truncated for display)[/]")
            
            console.print(Panel.fit(
                "\n".join(tree_lines), 
                title="Directory Tree", 
                border_style="yellow"
            ))
    
    # File Extensions Panel
    top_extensions = summary.get('top_file_extensions', [])
    if top_extensions:
        ext_table = Table(title="Top File Extensions", show_header=True, header_style="bold yellow")
        ext_table.add_column("Extension", style="cyan")
        ext_table.add_column("Count", justify="right", style="green")
        
        for ext in top_extensions[:10]:
            parts = ext.rsplit(' (', 1)
            if len(parts) == 2:
                extension = parts[0]
                count = parts[1].rstrip(')')
                ext_table.add_row(extension, count)
        
        console.print(ext_table)
    
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


