from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..domain.models import ClarificationStatus, RefactorSuggestionStatus, TaskStatus
from ..workflow.orchestrator import TaskOrchestrator
from .dashboard import run_task_dashboard

app = typer.Typer(add_completion=False, help="Spec-driven development agent CLI.")
console = Console()


def _get_orchestrator() -> TaskOrchestrator:
    return TaskOrchestrator()



def _infer_step_scenario(description: str, notes: Optional[str] = None, *, has_tests: bool = True) -> List[str]:
    """
    Create a small, human-readable Given/When/Then scenario for a plan step.

    This is intentionally heuristic: it converts high-level plan text into a
    quick "what success looks like" summary for senior engineers skimming.
    """
    desc = (description or "").strip()
    if not desc:
        return []

    desc_l = desc.lower()
    notes_clean = (notes or "").strip()

    given = "Given the codebase is indexed and the current behavior is understood"

    if desc_l.startswith("implement"):
        when = f"When we {desc_l}"
        then = "Then the new component exists and is ready to be integrated"
    elif desc_l.startswith("update") or desc_l.startswith("refactor"):
        when = f"When we {desc_l}"
        then = "Then the existing flow uses the updated configuration path"
    elif "test" in desc_l and has_tests:
        when = f"When we {desc_l}"
        then = "Then automated tests validate both happy-path and failure cases"
    elif desc_l.startswith("document"):
        when = f"When we {desc_l}"
        then = "Then the team can maintain the setup confidently without tribal knowledge"
    else:
        when = f"When we {desc_l}"
        then = "Then the change is implemented with clear acceptance criteria"

    lines = [
        f"Given: {given}",
        f"When:  {when}",
        f"Then:  {then}",
    ]
    if notes_clean:
        lines.append(f"And:   {notes_clean}")
    return lines


def _format_plan_steps_human(steps: List[Any], *, has_tests: bool = True) -> Dict[str, List[str]]:
    """
    Turn plan step payloads (dicts or strings) into:
    - a readable step list
    - a readable scenario list (Given/When/Then)
    """
    step_lines: List[str] = []
    scenario_lines: List[str] = []

    for idx, step in enumerate(steps or [], start=1):
        if isinstance(step, dict):
            description = (step.get("description") or "").strip()
            target_files = step.get("target_files") or []
            notes = step.get("notes")

            step_lines.append(f"{idx}. {description or '(missing step description)'}")
            if target_files:
                step_lines.append(f"   Targets: {', '.join(str(t) for t in target_files)}")
            if notes:
                step_lines.append(f"   Notes: {notes}")

            scenario = _infer_step_scenario(description, notes, has_tests=has_tests)
            if scenario:
                if scenario_lines:
                    scenario_lines.append("")  # spacer between scenarios
                scenario_lines.append(f"Step {idx}: {description or '(missing step description)'}")
                for line in scenario:
                    scenario_lines.append(f"- {line}")
        else:
            text = str(step)
            step_lines.append(f"{idx}. {text}")
            scenario = _infer_step_scenario(text, None, has_tests=has_tests)
            if scenario:
                if scenario_lines:
                    scenario_lines.append("")
                scenario_lines.append(f"Step {idx}: {text}")
                for line in scenario:
                    scenario_lines.append(f"- {line}")

    return {"steps": step_lines, "scenarios": scenario_lines}


@app.command()
def index(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, resolve_path=True),
    branch: str = typer.Option("main", "--branch", "-b"),
    json_output: bool = typer.Option(False, "--json", help="Emit repository summary as JSON after indexing."),
    serena_semantic_tree: bool = typer.Option(
        False,
        "--serena-semantic-tree",
        help="Export a full Serena semantic symbol tree (slow; requires Serena enabled).",
    ),
) -> None:
    """
    Index a repository and save the context for later use.
    """

    orchestrator = _get_orchestrator()
    index_data = orchestrator.index_repository(
        repo_path=repo,
        branch=branch,
        include_serena_semantic_tree=serena_semantic_tree,
    )

    summary = index_data.get("repository_summary", {})
    git_info = index_data.get("git_info", {})
    resolved_repo_path = Path(index_data.get("repo_path") or repo).resolve()

    console.print("[bold green]Repository indexed successfully[/]\n")

    if serena_semantic_tree:
        tree_payload = summary.get("serena_semantic_tree") or {}
        stats = tree_payload.get("stats") if isinstance(tree_payload, dict) else None
        if isinstance(stats, dict) and stats:
            console.print(
                f"[dim]Serena semantic tree: indexed {stats.get('indexed_files', 0)} files "
                f"(failed {stats.get('failed_files', 0)}) in {stats.get('elapsed_seconds', '?')}s[/]"
            )
        elif tree_payload:
            console.print("[dim]Serena semantic tree: generated (details saved to repository index)[/]")
        else:
            console.print("[yellow]Serena semantic tree requested but not generated (Serena disabled or unavailable).[/]")
    
    # Merged Repository Info and Semantic Analysis Panel
    info_lines = []
    
    # Basic repository information
    info_lines.append(f"[bold cyan]Repository:[/] {index_data.get('repo_name', repo.name)}")
    info_lines.append(f"[bold cyan]Path:[/] {resolved_repo_path}")
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

    # Serena semantic modules panel
    semantic_modules = summary.get('serena_semantic_modules') or []
    if semantic_modules:
        sem_table = Table(title="Serena Semantic Modules", show_header=True, header_style="bold magenta")
        sem_table.add_column("Module / File", style="cyan")
        sem_table.add_column("Key Symbols", style="green")
        sem_table.add_column("Referenced By")
        
        for module in semantic_modules[:5]:
            module_name = module.get('module', 'unknown')
            file_path = module.get('file')
            module_label = module_name
            if file_path:
                module_label = f"{module_name}\n[dim]{file_path}[/]"
            
            symbols = module.get('top_symbols') or []
            symbol_names = [sym.get('symbol', '') for sym in symbols if sym.get('symbol')]
            key_symbols = ", ".join(symbol_names[:4]) if symbol_names else "—"
            
            referenced_by = module.get('referenced_by_modules') or []
            referenced_display = ", ".join(referenced_by[:4]) if referenced_by else "—"
            
            sem_table.add_row(module_label, key_symbols, referenced_display)
        
        console.print(sem_table)
    
    # Serena Status
    if summary.get('serena_enabled'):
        console.print("[dim]Enhanced with Serena language detection[/]")
    else:
        console.print("[dim]Basic language detection (Serena not enabled)[/]")
    
    console.print("\n[cyan]Index saved. You can now run:[/] [bold]./spec-agent start --description \"Your task\"[/]")

    if json_output:
        console.print_json(data=json.dumps(index_data, default=str))


@app.command("context-summary")
def context_summary(
    top: int = typer.Option(5, "--top", help="Number of entries to show for modules and hotspots."),
    json_output: bool = typer.Option(False, "--json", help="Emit summary as JSON instead of tables."),
) -> None:
    """
    Show the most recent indexed repository summary (modules, hotspots, dependency graph).
    """
    orchestrator = _get_orchestrator()

    try:
        index_data = orchestrator.get_cached_repository_index()
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)

    summary = index_data.get("repository_summary", {})
    if not summary:
        console.print("[yellow]No repository summary found. Run './spec-agent index' first.[/]")
        raise typer.Exit(code=1)

    if json_output:
        payload = {
            "top_modules": summary.get("top_modules", [])[:top],
            "dependency_graph": summary.get("dependency_graph", {}),
            "hotspots": summary.get("hotspots", [])[:top],
            "languages": summary.get("top_languages", []),
        }
        console.print_json(data=json.dumps(payload, default=str))
        return

    modules = summary.get("top_modules", [])[:top]
    if modules:
        table = Table(title="Key Modules / Packages")
        table.add_column("Module")
        for module in modules:
            table.add_row(module)
        console.print(table)

    dep_graph = summary.get("dependency_graph") or {}
    fan_in = dep_graph.get("top_fan_in", [])[:top]
    if fan_in:
        table = Table(title="Most Referenced Modules (Fan-in)")
        table.add_column("Module")
        table.add_column("References", justify="right")
        for item in fan_in:
            table.add_row(item.get("module", ""), str(item.get("references", 0)))
        console.print(table)

    hotspots = summary.get("hotspots", [])[:top]
    if hotspots:
        table = Table(title="Legacy Hotspots")
        table.add_column("Path")
        table.add_column("Lines", justify="right")
        for hotspot in hotspots:
            table.add_row(hotspot.get("path", ""), str(hotspot.get("lines", "")))
        console.print(table)

    if not (modules or fan_in or hotspots):
        console.print("[yellow]Summary exists but no enriched data available yet.[/]")


@app.command("bounded-index")
def bounded_index(
    task_id: str = typer.Argument(..., help="UUID of the task to attach the bounded index to."),
    targets: List[Path] = typer.Argument(..., help="Paths (files or directories) to scan relative to the repo."),
    serena_semantic_tree: bool = typer.Option(
        False,
        "--serena-semantic-tree",
        help="Also export a scoped Serena semantic symbol tree for these targets (slow).",
    ),
) -> None:
    """
    Run a scoped index for specific files/directories after clarifications.
    """
    orchestrator = _get_orchestrator()

    if not targets:
        console.print("[red]Provide at least one target path.[/]")
        raise typer.Exit(code=1)

    target_args = [str(path) for path in targets]
    summary = orchestrator.bounded_index_task(
        task_id,
        target_args,
        include_serena_semantic_tree=serena_semantic_tree,
    )
    aggregate = summary.get("aggregate", {})

    console.print("[bold green]Bounded index completed[/]")
    if aggregate:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Files: {aggregate.get('file_count', 0)}",
                        f"Size: {aggregate.get('total_size_bytes', 0)} bytes",
                        f"Languages: {', '.join(aggregate.get('top_languages', [])) or 'unknown'}",
                    ]
                ),
                title="Aggregate",
            )
        )

    targets_summary = summary.get("targets", {})
    if targets_summary:
        table = Table(title="Targets Indexed")
        table.add_column("Path")
        table.add_column("Files", justify="right")
        table.add_column("Top Languages")
        for path, stats in targets_summary.items():
            table.add_row(
                path,
                str(stats.get("file_count", 0)),
                ", ".join(stats.get("top_languages", [])) or "unknown",
            )
        console.print(table)

    impact = summary.get("impact") or {}
    if impact:
        details = []
        top_dirs = impact.get("top_directories") or []
        namespaces = impact.get("namespaces") or []
        if top_dirs:
            details.append(f"Top directories: {', '.join(top_dirs[:10])}")
        if namespaces:
            details.append(f"Namespaces: {', '.join(namespaces[:10])}")
        if details:
            console.print(Panel.fit("\n".join(details), title="Impacted Modules (Bounded Scope)"))

    if serena_semantic_tree:
        tree_payload = summary.get("serena_semantic_tree") or {}
        stats = tree_payload.get("stats") if isinstance(tree_payload, dict) else None
        if isinstance(stats, dict) and stats:
            console.print(
                f"[dim]Scoped Serena semantic tree: indexed {stats.get('indexed_files', 0)} files "
                f"(failed {stats.get('failed_files', 0)}) in {stats.get('elapsed_seconds', '?')}s[/]"
            )
    elif not targets_summary and not (aggregate.get("file_count") or 0):
        console.print("[yellow]No matching files were indexed for the provided targets.[/]")


@app.command("context")
def manage_context(
    task_id: str = typer.Argument(..., help="UUID of the task whose context should be managed."),
    include: List[str] = typer.Option([], "--include", "-i", help="Paths to force-include in context."),
    exclude: List[str] = typer.Option([], "--exclude", "-x", help="Paths to exclude from future expansions."),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Why these paths are being adjusted."),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Show a scoped summary for included paths."),
) -> None:
    """
    Inspect and modify the iterative context expansion history for a task.
    """
    orchestrator = _get_orchestrator()
    performed_update = False

    if include or exclude:
        note = reason or "Manual context update"
        result = orchestrator.update_context(
            task_id,
            include=include,
            exclude=exclude,
            trigger="engineer-adjustment",
            note=note,
            summarize=summarize,
        )
        performed_update = True
        console.print("[green]Logged context update.[/]")
        if result.get("summary"):
            aggregate = result["summary"].get("aggregate", {})
            if aggregate:
                details = [
                    f"Files: {aggregate.get('file_count', 0)}",
                    f"Size: {aggregate.get('total_size_bytes', 0)} bytes",
                    f"Languages: {', '.join(aggregate.get('top_languages', [])) or 'unknown'}",
                ]
                console.print(Panel.fit("\n".join(details), title="Scoped Summary"))

    history = orchestrator.get_context_history(task_id)
    if not history:
        if not performed_update:
            console.print("[yellow]No context history recorded yet.[/]")
        return

    table = Table(title="Context Expansion Steps", show_lines=True)
    table.add_column("When")
    table.add_column("Trigger")
    table.add_column("Included")
    table.add_column("Excluded")
    for step in history[-15:]:
        table.add_row(
            step.get("timestamp", ""),
            f"{step.get('trigger', '')}\n{step.get('note', '') or ''}",
            "\n".join(step.get("included", []) or ["—"]),
            "\n".join(step.get("excluded", []) or ["—"]),
        )
    console.print(table)


@app.command()
def start(
    description: str = typer.Option(..., "--description", "-d", prompt=True),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Short task title shown in dashboards."),
    summary: Optional[str] = typer.Option(
        None,
        "--summary",
        help="Short summary shown in task lists (defaults to first line of description).",
    ),
    client: Optional[str] = typer.Option(
        None,
        "--client",
        help="Editor/chat client driving this task (e.g. cursor, copilot, claude, terminal).",
    ),
) -> None:
    """
    Create a new task using the previously indexed repository.
    """

    orchestrator = _get_orchestrator()
    task = orchestrator.create_task_from_index(description=description, title=title, summary=summary, client=client)

    clarifications = task.metadata.get("clarifications", [])

    console.print(f"[bold green]Created task[/] {task.id}")

    if clarifications:
        table = Table(title="Clarification Questions")
        table.add_column("ID")
        table.add_column("Question")
        for item in clarifications:
            table.add_row(item["id"], item["question"])
        console.print(table)
        console.print(f"[dim]Answer them via:[/] ./spec-agent clarifications {task.id}")
    else:
        console.print("[cyan]No clarification questions detected.[/]")


@app.command("clarifications")
def handle_clarifications(
    task_id: str = typer.Argument(..., help="UUID of the task whose clarifications should be reviewed."),
    list_only: bool = typer.Option(False, "--list", help="Only display questions without prompting for answers."),
) -> None:
    """
    Review and answer clarification questions for a task.
    """
    orchestrator = _get_orchestrator()
    clarifications = orchestrator.list_clarifications(task_id)

    if not clarifications:
        console.print("[cyan]No clarification questions recorded for this task.[/]")
        return

    if list_only:
        table = Table(title="Clarification Questions")
        table.add_column("ID", style="bold")
        table.add_column("Status")
        table.add_column("Question")
        for item in clarifications:
            table.add_row(item.get("id", ""), item.get("status", "PENDING"), item.get("question", ""))
        console.print(table)
        return

    handled = False
    for item in clarifications:
        status = item.get("status") or ClarificationStatus.PENDING.value
        console.print()
        console.print(Panel.fit(item.get("question", "No question text provided."), title=f"Question {item.get('id','')} [{status}]"))
        if status != ClarificationStatus.PENDING.value:
            existing = item.get("answer") or "(no answer recorded)"
            console.print(f"[dim]Already resolved: {existing}[/]")
            continue

        answer = typer.prompt("Your answer (leave blank to skip)", default="").strip()
        if answer:
            orchestrator.update_clarification(
                task_id,
                item.get("id"),
                answer=answer,
                status=ClarificationStatus.ANSWERED,
            )
            console.print("[green]Saved answer.[/]")
            handled = True
        else:
            if typer.confirm("Override/skip this question?", default=True):
                orchestrator.update_clarification(
                    task_id,
                    item.get("id"),
                    answer="",
                    status=ClarificationStatus.OVERRIDDEN,
                )
                console.print("[yellow]Question overridden.[/]")
                handled = True

    if handled:
        console.print("\n[bold green]All clarifications processed![/]")
    else:
        console.print("\n[yellow]No changes made.[/]")


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
    table.add_column("Client", width=10)
    table.add_column("Title", style="bold", width=34)
    table.add_column("ID", style="dim", width=10, no_wrap=True)
    table.add_column("Repo")
    table.add_column("Branch")
    table.add_column("Status")
    table.add_column("Updated")

    for task in tasks:
        table.add_row(
            (task.client or "—")[:10],
            (task.title or task.description.splitlines()[0] if task.description else f"task-{task.id[:8]}")[:34],
            task.id[:8],
            str(task.repo_path),
            task.branch,
            task.status.value,
            task.updated_at.isoformat(timespec="seconds"),
        )

    console.print(table)


@app.command("dashboard")
def dashboard(
    task_id: Optional[str] = typer.Option(None, "--task", help="Focus the dashboard on a specific task id."),
    status: Optional[TaskStatus] = typer.Option(None, "--status", "-s", case_sensitive=False),
    show_all: bool = typer.Option(False, "--all", help="Include completed/cancelled tasks."),
    refresh: float = typer.Option(1.0, "--refresh", help="Refresh interval in seconds (min 0.2)."),
) -> None:
    """
    Live dashboard showing active tasks and their workflow stage.

    Exit with Ctrl+C.
    """
    orchestrator = _get_orchestrator()
    run_task_dashboard(orchestrator, task_id=task_id, status=status, show_all=show_all, refresh_seconds=refresh)


@app.command("web")
def web_dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address for the web dashboard."),
    port: int = typer.Option(8844, "--port", help="Port for the web dashboard."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the dashboard in your browser."),
) -> None:
    """
    Web dashboard (mockup-style) powered by tasks/logs under the state dir.

    Reads from `SPEC_AGENT_STATE_DIR` (defaults to ~/.spec_agent).
    """
    from ..web.server import run_dashboard_server

    url = f"http://{host}:{int(port)}"
    console.print(f"[bold cyan]Web dashboard:[/] {url}")
    console.print("[dim]Press Ctrl+C to stop.[/]")

    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    try:
        run_dashboard_server(host=host, port=int(port))
    except KeyboardInterrupt:
        return


@app.command("task-edit")
def edit_task(
    task_id: str = typer.Argument(..., help="UUID of the task to edit."),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="New task description. If omitted, you will be prompted.",
    ),
    keep_metadata: bool = typer.Option(
        False,
        "--keep-metadata",
        help="Only update description; do NOT reset clarifications/plan/specs/patches context.",
    ),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Why the description is changing."),
) -> None:
    """
    Update a task description safely.

    By default this treats the new description as a new intent: it regenerates
    clarifications and clears stale metadata (plan/specs/patches/bounded context)
    to avoid mismatched context.
    """
    orchestrator = _get_orchestrator()

    if description is None:
        description = typer.prompt("New description", default="").strip()

    try:
        task = orchestrator.update_task_description(
            task_id,
            description or "",
            reason=reason,
            reset_metadata=not keep_metadata,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)

    if keep_metadata:
        console.print(f"[green]Updated description for task {task.id}.[/]")
        return

    console.print(f"[green]Updated task {task.id} and reset metadata for the new description.[/]")
    clarifications = task.metadata.get("clarifications", []) or []
    if clarifications:
        console.print(f"[dim]Generated {len(clarifications)} clarifying question(s). Review with:[/] ./spec-agent clarifications {task.id}")


@app.command()
def plan(
    task_id: str = typer.Argument(..., help="UUID of the task to plan."),
    fast: bool = typer.Option(False, "--fast", help="Skip rationale enhancement for faster execution"),
) -> None:
    """
    Generate a plan, boundary specs, and refactor suggestions for a task.
    
    Use --fast to skip rationale enhancement (Epic 4.1) for faster execution.
    """

    orchestrator = _get_orchestrator()

    if orchestrator.has_pending_clarifications(task_id):
        console.print("[red]Clarification questions are still pending.[/]")
        console.print(f"[dim]Resolve them via './spec-agent clarifications {task_id}' before planning.[/]")
        raise typer.Exit(code=1)
    
    # Show progress
    console.print("[cyan]Generating plan...[/]")
    if fast:
        console.print("[yellow]Fast mode: Skipping rationale enhancement[/]")
    
    payload = orchestrator.generate_plan(task_id, skip_rationale_enhancement=fast)

    plan_data = payload["plan"]
    task = orchestrator._get_task(task_id)
    has_tests = bool((task.metadata.get("repository_summary") or {}).get("has_tests", False))
    formatted = _format_plan_steps_human(plan_data.get("steps", []), has_tests=has_tests)
    step_lines = formatted.get("steps") or []
    scenario_lines = formatted.get("scenarios") or []
    if step_lines:
        console.print(Panel.fit("\n".join(step_lines), title="Plan Steps"))
    if scenario_lines:
        console.print(Panel.fit("\n".join(scenario_lines), title="Scenarios (Given / When / Then)"))
    if not has_tests:
        console.print("[dim]No automated tests detected in this repository. Consider adding them later.[/]")
    if plan_data.get("risks"):
        console.print(Panel.fit("\n".join(plan_data["risks"]), title="Risks"))
    if plan_data.get("refactors"):
        console.print(Panel.fit("\n".join(plan_data["refactors"]), title="Refactor Suggestions"))

    if payload["pending_specs"]:
        console.print(Panel.fit("\n".join(payload["pending_specs"]), title="Pending Boundary Specs"))

    bounded = payload.get("bounded_context", {})
    if bounded:
        aggregate = bounded.get("aggregate", {})
        details = [
            f"Files: {aggregate.get('file_count', 0)}",
            f"Size: {aggregate.get('total_size_bytes', 0)} bytes",
            f"Languages: {', '.join(aggregate.get('top_languages', [])) or 'unknown'}",
        ]
        console.print(Panel.fit("\n".join(details), title="Scoped Context (Plan Targets)"))

        # If the bounded context is empty, explain why (targets are logical module names).
        resolution = bounded.get("plan_targets_resolution") if isinstance(bounded, dict) else None
        if aggregate.get("file_count", 0) == 0 and isinstance(resolution, dict):
            raw_targets = resolution.get("raw_targets") or []
            resolved_targets = resolution.get("resolved_targets") or []
            unresolved_targets = resolution.get("unresolved_targets") or []

            explain_lines = [
                "Scoped context is empty because plan targets are module labels (not filesystem paths).",
                "Use a bounded index with real directories/files to scope context precisely.",
            ]
            if raw_targets:
                explain_lines.append("")
                explain_lines.append(f"Plan targets: {', '.join(str(t) for t in raw_targets[:12])}{' ...' if len(raw_targets) > 12 else ''}")
            if resolved_targets:
                explain_lines.append(f"Resolved to paths: {', '.join(str(t) for t in resolved_targets[:12])}{' ...' if len(resolved_targets) > 12 else ''}")
            if unresolved_targets:
                explain_lines.append(f"Unresolved: {', '.join(str(t) for t in unresolved_targets[:12])}{' ...' if len(unresolved_targets) > 12 else ''}")

            console.print(Panel.fit("\n".join(explain_lines), title="Why Files=0", border_style="yellow"))

        impact = bounded.get("impact") or {}
        if impact:
            impact_lines = []
            top_dirs = impact.get("top_directories") or []
            namespaces = impact.get("namespaces") or []
            if top_dirs:
                impact_lines.append(f"Top directories: {', '.join(top_dirs[:10])}")
            if namespaces:
                impact_lines.append(f"Namespaces: {', '.join(namespaces[:10])}")
            if impact_lines:
                console.print(Panel.fit("\n".join(impact_lines), title="Impacted Modules (Scoped Context)"))

        tree_payload = bounded.get("serena_semantic_tree") or {}
        stats = tree_payload.get("stats") if isinstance(tree_payload, dict) else None
        if isinstance(stats, dict) and stats:
            console.print(
                f"[dim]Scoped Serena semantic tree: indexed {stats.get('indexed_files', 0)} files "
                f"(failed {stats.get('failed_files', 0)}) in {stats.get('elapsed_seconds', '?')}s[/]"
            )

    console.print()
    console.print("[dim]Next steps:[/]")
    console.print(f"  1. Resolve boundary specs with ./spec-agent specs {task_id}")
    console.print(f"  2. Approve the plan via ./spec-agent approve-plan {task_id}")
    console.print(f"  3. Generate patches via ./spec-agent generate-patches {task_id}")

@app.command("approve-plan")
def approve_plan_cmd(
    task_id: str = typer.Argument(..., help="UUID of the task whose plan should be approved."),
) -> None:
    """
    Approve the current implementation plan (all specs must be resolved first).
    """
    orchestrator = _get_orchestrator()
    try:
        orchestrator.approve_plan(task_id)
        console.print("[bold green]Plan approved. You can now generate patches.[/]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)


@app.command("generate-patches")
def generate_patches_cmd(
    task_id: str = typer.Argument(..., help="UUID of the task to generate patches for."),
    fast: bool = typer.Option(False, "--fast", help="Skip rationale enhancement for faster execution"),
) -> None:
    """
    Generate patches (and optionally test suggestions) for an approved plan.
    """
    orchestrator = _get_orchestrator()
    try:
        console.print("[cyan]Drafting patches...[/]")
        result = orchestrator.generate_patches(task_id, skip_rationale_enhancement=fast)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)

    tests_skipped_reason = (result or {}).get("tests_skipped_reason")
    if tests_skipped_reason:
        body = f"Patches: {result['patch_count']}\nTest suggestions: skipped ({tests_skipped_reason})"
    else:
        body = f"Patches: {result['patch_count']}\nTest suggestions: {result['test_count']}"
    console.print(Panel.fit(body, title="Generation Complete"))
    console.print(f"[dim]Review patches with './spec-agent patches {task_id}'.[/]")

@app.command("patches")
def review_patches(
    task_id: str = typer.Argument(..., help="UUID of the task whose patches should be reviewed."),
    list_only: bool = typer.Option(False, "--list", help="Only list patches without prompting for approval."),
) -> None:
    """
    Inspect and approve/reject incremental patches for a task.
    """

    orchestrator = _get_orchestrator()
    if orchestrator.has_manual_edits(task_id):
        console.print("[yellow]Detected manual edits in the working tree.[/]")
        console.print("[dim]Patches may no longer apply cleanly.[/]")
        choice = typer.prompt("Regenerate plan (r), continue anyway (c), or abort (a)?", default="r").strip().lower()
        if choice.startswith("r"):
            console.print("[cyan]Regenerating plan to account for manual edits...[/]")
            orchestrator.generate_plan(task_id)
            console.print(f"[yellow]Plan regenerated. Review specs, run './spec-agent approve-plan {task_id}', then './spec-agent generate-patches {task_id}' before reviewing patches.[/]")
            return
        if choice.startswith("c"):
            orchestrator.acknowledge_manual_edits(task_id)
            console.print("[cyan]Continuing with existing patch queue.[/]")
        else:
            console.print("[cyan]Aborted patch review.[/]")
            return

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
                    console.print("[dim]Continuing to next patch...[/]\n")
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

@app.command("logs")
def show_logs(
    task_id: Optional[str] = typer.Option(None, "--task", help="Filter by task ID."),
    entry_type: List[str] = typer.Option([], "--type", "-t", help="Filter by log entry type."),
    limit: int = typer.Option(50, "--limit", help="Maximum number of entries to display."),
    json_output: bool = typer.Option(False, "--json", help="Emit logs as JSON."),
) -> None:
    """
    Display reasoning log entries, filtered by task or entry type.
    """
    orchestrator = _get_orchestrator()
    entries = orchestrator.store.load_logs()

    if task_id:
        entries = [entry for entry in entries if entry.task_id == task_id]
    if entry_type:
        allowed = {item.upper() for item in entry_type}
        entries = [entry for entry in entries if entry.entry_type.upper() in allowed]

    if not entries:
        console.print("[yellow]No matching log entries found.[/]")
        return

    entries.sort(key=lambda item: item.timestamp, reverse=True)
    entries = entries[:limit]

    if json_output:
        payload = [
            {
                "id": entry.id,
                "task_id": entry.task_id,
                "timestamp": entry.timestamp.isoformat(),
                "entry_type": entry.entry_type,
                "payload": entry.payload,
            }
            for entry in entries
        ]
        console.print_json(data=json.dumps(payload, default=str))
        return

    table = Table(title="Reasoning Log")
    table.add_column("Time")
    table.add_column("Task")
    table.add_column("Type")
    table.add_column("Details")

    for entry in entries:
        payload_preview = entry.payload if isinstance(entry.payload, str) else json.dumps(entry.payload, default=str)
        table.add_row(
            entry.timestamp.isoformat(timespec="seconds"),
            entry.task_id,
            entry.entry_type,
            payload_preview[:120] + ("…" if len(payload_preview) > 120 else ""),
        )

    console.print(table)


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


@app.command("clean-tasks")
def clean_tasks(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """
    Clear all tasks (but keep reasoning logs).

    Tasks are stored in ~/.spec_agent/tasks.json
    """
    from ..config.settings import get_settings
    from ..persistence.store import JsonStore

    settings = get_settings()
    store = JsonStore(settings.state_dir)

    tasks_file = store.tasks_file
    existing_tasks = store.load_tasks()

    if not existing_tasks:
        console.print("[yellow]No tasks to clean.[/]")
        return

    console.print(f"[yellow]Found {len(existing_tasks)} tasks[/]")

    if not confirm:
        response = typer.prompt(
            f"Delete all tasks ({len(existing_tasks)} tasks)? [y/N]",
            default="n",
        )
        if response.lower() not in {"y", "yes"}:
            console.print("[cyan]Cancelled.[/]")
            return

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


@app.command()
def chat() -> None:
    """
    Start an interactive chat session for spec-driven development.

    This provides a conversational interface where you can:
    - Create and manage tasks interactively
    - Answer clarifying questions one by one
    - Review plans and boundary specs
    - Approve or reject patches

    The chat mode guides you through the entire workflow with menus and prompts.
    """
    from .chat import ChatSession

    orchestrator = _get_orchestrator()
    session = ChatSession(orchestrator)
    session.run()


def main() -> None:
    app()
