from __future__ import annotations

from enum import Enum
import shlex
from pathlib import Path
from typing import Optional
from threading import Lock, Thread
from uuid import uuid4

from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from ..domain.models import Task, TaskStatus
from ..workflow.orchestrator import TaskOrchestrator
from .dashboard import run_task_dashboard


console = Console()


def _infer_step_scenario(description: str, notes: Optional[str] = None, *, has_tests: bool = True) -> list[str]:
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


def _format_plan_steps_human(steps: list[object], *, has_tests: bool = True) -> tuple[list[str], list[str]]:
    """
    Turn plan step payloads (dicts or strings) into:
    - a readable step list
    - a readable scenario list (Given/When/Then)
    """
    step_lines: list[str] = []
    scenario_lines: list[str] = []

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

    return step_lines, scenario_lines


class ConversationState(str, Enum):
    """Tracks the current state of the chat conversation."""
    MAIN_MENU = "MAIN_MENU"
    INDEXING = "INDEXING"
    TASK_SETUP = "TASK_SETUP"
    CLARIFYING = "CLARIFYING"
    PLANNING = "PLANNING"
    REVIEWING_PLAN = "REVIEWING_PLAN"
    REVIEWING_SPECS = "REVIEWING_SPECS"
    REVIEWING_PATCHES = "REVIEWING_PATCHES"
    EXITING = "EXITING"


class ChatSession:
    """
    Interactive chat session for spec-driven development.
    Provides a conversational interface to the orchestrator.
    """

    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.state = ConversationState.MAIN_MENU
        self.current_task: Optional[Task] = None
        self.indexed_repo: Optional[Path] = None
        self.indexed_branch: str = "main"
        self.index_only: bool = False  # Track if we're just indexing vs starting a task
        self._bg_lock = Lock()
        self._bg_jobs: dict[str, dict] = {}
        self._maybe_load_cached_index()

    def _maybe_load_cached_index(self) -> None:
        """
        Best-effort: reuse the last indexed repo for chat sessions.

        This avoids forcing engineers to re-index on every run and makes
        "Start a new task" flow seamless after setup.
        """
        try:
            index_data = self.orchestrator.get_cached_repository_index()
        except Exception:
            return

        try:
            repo_path = Path(index_data.get("repo_path", "")).resolve()
        except Exception:
            return

        if not repo_path.exists() or not repo_path.is_dir():
            return

        self.indexed_repo = repo_path
        self.indexed_branch = str(index_data.get("branch") or "main")

    def _ask_menu_choice(self, prompt: str, choices: list[str], default: str) -> str:
        """
        Robust choice prompt for auto-started terminals.

        Terminals auto-focused on workspace open can inject stray characters (whitespace,
        escape sequences, etc). We parse very defensively:
        - blank/whitespace => default
        - if any allowed digit appears in the input => choose it
        - otherwise => default (no noisy error loop)
        """
        raw = Prompt.ask(prompt, default=default)
        value = (raw or "").strip()
        if not value:
            return default

        # If the input contains a valid option anywhere (e.g. "1 ", "1\n", or stray chars),
        # pick the first matching choice.
        for ch in value:
            if ch in choices:
                return ch

        return default

    def _start_background_job(self, name: str, fn) -> str:
        job_id = str(uuid4())
        with self._bg_lock:
            self._bg_jobs[job_id] = {
                "name": name,
                "status": "RUNNING",
                "error": None,
                "result": None,
                "notified": False,
            }

        def runner() -> None:
            try:
                result = fn()
                with self._bg_lock:
                    self._bg_jobs[job_id]["status"] = "DONE"
                    self._bg_jobs[job_id]["result"] = result
            except Exception as exc:  # pragma: no cover
                with self._bg_lock:
                    self._bg_jobs[job_id]["status"] = "FAILED"
                    self._bg_jobs[job_id]["error"] = str(exc)

        Thread(target=runner, daemon=True).start()
        return job_id

    def _notify_background_jobs(self) -> None:
        to_notify: list[dict] = []
        with self._bg_lock:
            for job in self._bg_jobs.values():
                if job.get("notified"):
                    continue
                status = job.get("status")
                if status in {"DONE", "FAILED"}:
                    job["notified"] = True
                    to_notify.append(dict(job))

        for job in to_notify:
            name = job.get("name") or "job"
            if job.get("status") == "DONE":
                console.print(f"[green]✓ Background job finished:[/] {name}")
            else:
                err = job.get("error") or "unknown error"
                console.print(f"[red]✗ Background job failed:[/] {name} ({err})")

    def _show_cli_command(self, command: str) -> None:
        """Mirror the CLI command that corresponds to the current action."""
        console.print(f"[dim]→ Running:[/] [bold]{command}[/]")

    def run(self) -> None:
        """Main conversation loop."""
        self._show_welcome()

        while self.state != ConversationState.EXITING:
            try:
                # Surface any background job completions between steps.
                self._notify_background_jobs()
                if self.state == ConversationState.MAIN_MENU:
                    self._handle_main_menu()
                elif self.state == ConversationState.INDEXING:
                    self._handle_indexing()
                elif self.state == ConversationState.TASK_SETUP:
                    self._handle_task_setup()
                elif self.state == ConversationState.CLARIFYING:
                    self._handle_clarifying()
                elif self.state == ConversationState.PLANNING:
                    self._handle_planning()
                elif self.state == ConversationState.REVIEWING_PLAN:
                    self._handle_plan_review()
                elif self.state == ConversationState.REVIEWING_SPECS:
                    self._handle_spec_review()
                elif self.state == ConversationState.REVIEWING_PATCHES:
                    self._handle_patch_review()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Returning to main menu.[/]")
                self.state = ConversationState.MAIN_MENU
            except Exception as exc:
                # Defensive: exceptions may contain Rich markup-like tokens (e.g. '[/]').
                # Print without markup to avoid MarkupError masking the real issue.
                console.print(f"Error: {exc}", style="red", markup=False)
                console.print("[yellow]Returning to main menu.[/]")
                self.state = ConversationState.MAIN_MENU

        self._show_goodbye()

    def _show_welcome(self) -> None:
        """Display welcome message."""
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Welcome to Spec Agent Interactive Mode[/]\n\n"
            "I'll guide you through spec-driven development with clarifying questions,\n"
            "boundary specifications, and incremental patches.\n\n"
            "[dim]Press Ctrl+C at any time to return to the main menu.[/]",
            border_style="cyan"
        ))
        console.print()

    def _show_goodbye(self) -> None:
        """Display goodbye message."""
        console.print()
        console.print("[bold cyan]Goodbye! Happy coding![/]")
        console.print()

    def _handle_main_menu(self) -> None:
        """Handle main menu interaction."""
        console.print()
        console.print("[bold]What would you like to do?[/]")
        console.print()
        console.print("  [cyan]1.[/] Start a new task")
        console.print("  [cyan]2.[/] Continue existing task")
        console.print("  [cyan]3.[/] View task history")
        console.print("  [cyan]4.[/] Index a repository")
        console.print("  [cyan]5.[/] Live tasks dashboard")
        console.print("  [cyan]6.[/] Exit")
        console.print()

        choice = self._ask_menu_choice("Choice", ["1", "2", "3", "4", "5", "6"], "1")

        if choice == "1":
            self.index_only = False
            # Always let the engineer choose a repo path, then reuse cached index if present.
            console.print()
            default_repo = str(self.indexed_repo) if self.indexed_repo else str(Path.cwd())
            console.print(f"Repository path (press Tab for completion) [default: {default_repo}]:")
            try:
                repo_path_str = prompt_toolkit_prompt(
                    "> ",
                    completer=PathCompleter(only_directories=True, expanduser=True),
                    default=default_repo,
                ).strip()
                if not repo_path_str:
                    repo_path_str = default_repo
            except (KeyboardInterrupt, EOFError):
                console.print("[yellow]Cancelled[/]")
                self.state = ConversationState.MAIN_MENU
                return

            requested = Path(repo_path_str).resolve()
            if not requested.exists() or not requested.is_dir():
                console.print(f"[red]Error: {requested} is not a valid directory[/]")
                self.state = ConversationState.MAIN_MENU
                return

            repo_root = self.orchestrator.resolve_repo_root(requested)

            # Ask for branch before deciding whether we can reuse a cached index.
            branch_default = self.indexed_branch or "main"
            branch = Prompt.ask("Branch", default=branch_default)

            # Try to reuse an existing cached index for this repo.
            try:
                cached = self.orchestrator.get_repository_index_for_repo_and_branch(repo_root, branch)
                self.indexed_repo = Path(cached.get("repo_path") or repo_root).resolve()
                self.indexed_branch = str(cached.get("branch") or branch or "main")
                console.print(f"[cyan]Using cached index for:[/] {self.indexed_repo} ({self.indexed_branch})")
                self.state = ConversationState.TASK_SETUP
            except Exception:
                # No cached index: go index now.
                console.print(f"[yellow]No cached index for {repo_root} ({branch}). Indexing now...[/]")
                self.indexed_repo = repo_root
                self.indexed_branch = branch
                self.state = ConversationState.INDEXING
        elif choice == "2":
            self._continue_task()
        elif choice == "3":
            self._show_task_history()
        elif choice == "4":
            self.index_only = True
            self.state = ConversationState.INDEXING
        elif choice == "5":
            console.print()
            console.print("[dim]Live dashboard — press Ctrl+C to return to the menu.[/]")
            focus = self.current_task.id if self.current_task else None
            run_task_dashboard(self.orchestrator, task_id=focus, show_all=False, refresh_seconds=1.0)
            self.state = ConversationState.MAIN_MENU
        elif choice == "6":
            self.state = ConversationState.EXITING

    def _handle_indexing(self) -> None:
        """Handle repository indexing."""
        console.print()
        console.print("[bold cyan]Repository Setup[/]")
        console.print()

        # Ask for repository path with tab completion (reuse pre-selected repo when available)
        default_path = str(self.indexed_repo) if self.indexed_repo else str(Path.cwd())
        console.print(f"Repository path (press Tab for completion) [default: {default_path}]:")

        try:
            repo_path_str = prompt_toolkit_prompt(
                "> ",
                completer=PathCompleter(only_directories=True, expanduser=True),
                default=default_path
            ).strip()

            # Use default if empty
            if not repo_path_str:
                repo_path_str = default_path

        except (KeyboardInterrupt, EOFError):
            console.print("[yellow]Cancelled[/]")
            self.state = ConversationState.MAIN_MENU
            return

        repo_path = Path(repo_path_str).resolve()

        if not repo_path.exists() or not repo_path.is_dir():
            console.print(f"[red]Error: {repo_path} is not a valid directory[/]")
            self.state = ConversationState.MAIN_MENU
            return

        # Ask for branch (reuse last branch if known)
        branch_default = self.indexed_branch or "main"
        branch = Prompt.ask("Branch", default=branch_default)

        # Show equivalent CLI command and index the repository
        command = f"./spec-agent index {shlex.quote(str(repo_path))}"
        if branch != "main":
            command += f" --branch {shlex.quote(branch)}"
        self._show_cli_command(command)

        console.print()
        console.print("[yellow]Indexing repository...[/]")

        try:
            # In interactive chat, we always run the "full index" that includes the
            # Serena semantic tree (when Serena is enabled). This avoids a second,
            # confusing "background semantic tree export" step.
            index_data = self.orchestrator.index_repository(
                repo_path=repo_path,
                branch=branch,
                include_serena_semantic_tree=True,
            )
            summary = index_data.get("repository_summary", {})
            git_info = index_data.get("git_info", {})
            resolved_repo_path = Path(index_data.get("repo_path") or repo_path).resolve()

            console.print()
            console.print("[bold green]Repository indexed successfully[/]\n")
            
            # Merged Repository Info and Semantic Analysis Panel
            info_lines = []
            
            # Basic repository information
            info_lines.append(f"[bold cyan]Repository:[/] {index_data.get('repo_name', repo_path.name)}")
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
            
            # Serena Status
            if summary.get('serena_enabled'):
                console.print("[dim]Enhanced with Serena language detection[/]")
            else:
                console.print("[dim]Basic language detection (Serena not enabled)[/]")

            # If the semantic tree was produced, print a small stat line.
            tree_payload = summary.get("serena_semantic_tree") or {}
            stats = tree_payload.get("stats") if isinstance(tree_payload, dict) else None
            if isinstance(stats, dict) and stats:
                console.print(
                    f"[dim]Serena semantic tree: indexed {stats.get('indexed_files', 0)} files "
                    f"(failed {stats.get('failed_files', 0)}) in {stats.get('elapsed_seconds', '?')}s[/]"
                )

            self.indexed_repo = resolved_repo_path
            self.indexed_branch = branch

            # If just indexing, return to menu. Otherwise, proceed to task setup
            if self.index_only:
                console.print()
                console.print("[cyan]Repository indexed! You can now start a task.[/]")
                self.state = ConversationState.MAIN_MENU
            else:
                self.state = ConversationState.TASK_SETUP

        except Exception as exc:
            console.print(f"Failed to index repository: {exc}", style="red", markup=False)
            self.state = ConversationState.MAIN_MENU

    def _handle_task_setup(self) -> None:
        """Handle task description input."""
        console.print()
        console.print("[bold cyan]Task Setup[/]")
        console.print()
        console.print("Describe the change you want to make:")
        console.print("[dim](Be as detailed or brief as you like - I'll ask clarifying questions)[/]")
        console.print()

        description = Prompt.ask("Description")

        if not description or description.strip() == "":
            console.print("[yellow]Description cannot be empty.[/]")
            return

        if not self.indexed_repo:
            console.print("[red]Please index a repository before creating a task.[/]")
            self.state = ConversationState.MAIN_MENU
            return

        command = f"./spec-agent start --description {shlex.quote(description)}"
        self._show_cli_command(command)

        # Create the task using the indexed repository metadata
        console.print()
        console.print("[yellow]Creating task...[/]")

        try:
            self.current_task = self.orchestrator.create_task_from_index(description=description)
            self.indexed_repo = self.current_task.repo_path
            self.indexed_branch = self.current_task.branch

            console.print(f"[green]✓[/] Task created: [cyan]{self.current_task.id}[/]")
            self.state = ConversationState.CLARIFYING

        except Exception as exc:
            console.print(f"Failed to create task: {exc}", style="red", markup=False)
            self.state = ConversationState.MAIN_MENU

    def _handle_clarifying(self) -> None:
        """Handle clarifying questions."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

        # Reload task to pick up any auto-resets (e.g. description changed in tasks.json)
        self.current_task = self.orchestrator._get_task(self.current_task.id)

        clarifications = self.current_task.metadata.get("clarifications", [])

        if not clarifications:
            console.print()
            console.print("[cyan]No clarifying questions needed. Proceeding to planning...[/]")
            self.state = ConversationState.PLANNING
            return

        console.print()
        console.print(Panel.fit(
            f"[bold]I have {len(clarifications)} clarifying question(s)[/]\n\n"
            "Your answers will help me generate a better implementation plan.",
            title="Clarifying Questions",
            border_style="cyan"
        ))

        # Ask questions one by one
        for i, item in enumerate(clarifications, 1):
            console.print()
            console.print("─" * 60)
            console.print(f"[bold]Question {i} of {len(clarifications)}[/]")
            console.print("─" * 60)
            console.print()
            console.print(item["question"])
            console.print()

            answer = Prompt.ask("Your answer", default="")

            if answer and answer.strip():
                # Store the answer
                item["answer"] = answer.strip()
                item["status"] = "ANSWERED"
                console.print("[green]✓[/] Got it!")
            else:
                console.print("[yellow]Skipped[/]")
                item["status"] = "OVERRIDDEN"

        # Update task with answers
        self.current_task.metadata["clarifications"] = clarifications
        self.orchestrator.store.upsert_task(self.current_task)

        console.print()
        console.print("[green]✓[/] All questions processed!")

        # Note: we infer/freeze scope AFTER preliminary plan approval (not here),
        # because the engineer may reject the preliminary plan or adjust direction.

        # Proceed to planning
        self.state = ConversationState.PLANNING

    def _handle_planning(self) -> None:
        """Handle plan generation."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

        # Reload task to ensure metadata matches current description/clarifications
        self.current_task = self.orchestrator._get_task(self.current_task.id)

        console.print()
        console.print("[yellow]Generating implementation plan...[/]")

        try:
            self.orchestrator.generate_plan(self.current_task.id)

            # Reload the task to get updated metadata
            self.current_task = self.orchestrator._get_task(self.current_task.id)

            console.print("[green]✓[/] Plan generated!")
            self.state = ConversationState.REVIEWING_PLAN

        except Exception as exc:
            console.print(f"Failed to generate plan: {exc}", style="red", markup=False)
            self.state = ConversationState.MAIN_MENU

    def _handle_plan_review(self) -> None:
        """Handle plan review and approval."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

        # Reload in case plan/spec metadata was regenerated elsewhere
        self.current_task = self.orchestrator._get_task(self.current_task.id)

        plan_preview = self.current_task.metadata.get("plan_preview", {})
        steps = plan_preview.get("steps", [])
        risks = plan_preview.get("risks", [])
        refactors = plan_preview.get("refactors", [])
        stage = (self.current_task.metadata.get("plan_stage") or "").strip().upper()

        # Display steps (disable markup to avoid conflicts with LLM-generated content)
        if steps:
            console.print()
            has_tests = bool((self.current_task.metadata.get("repository_summary") or {}).get("has_tests", False))
            step_lines, scenario_lines = _format_plan_steps_human(steps, has_tests=has_tests)
            step_text = "\n".join(step_lines) if step_lines else ""
            console.print(Panel.fit(
                step_text,
                title="Final Plan (Frozen Scope)" if stage == "FINAL" else "Preliminary Plan",
                border_style="blue"
            ), markup=False)

            if scenario_lines:
                console.print()
                console.print(
                    Panel.fit(
                        "\n".join(scenario_lines),
                        title="Scenarios (Given / When / Then)",
                        border_style="blue",
                    ),
                    markup=False,
                )
            if not has_tests:
                console.print("[dim]No automated tests detected in this repository. Consider adding them later.[/]")
        else:
            console.print()
            console.print("[yellow]No plan steps generated.[/]")

        if risks:
            console.print()
            risk_text = "\n".join(f"• {risk}" for risk in risks)
            console.print(Panel.fit(
                risk_text,
                title="Risks",
                border_style="yellow"
            ), markup=False)

        if refactors:
            console.print()
            refactor_text = "\n".join(f"• {refactor}" for refactor in refactors)
            console.print(Panel.fit(
                refactor_text,
                title="Refactoring Suggestions",
                border_style="cyan"
            ), markup=False)

        # Show boundary specs for information (not approval)
        specs = self.orchestrator.get_boundary_specs(self.current_task.id)
        if specs:
            console.print()
            # Escape spec names to avoid Rich markup conflicts
            spec_names = ', '.join(s.get('boundary_name', 'Unknown') for s in specs)
            console.print(
                f"Note: {len(specs)} boundary specification(s) identified",
                style="dim",
            )
            console.print(f"  → {spec_names}", style="dim")

        # Show scoped/bounded impact if available (post-clarifications indexing)
        bounded_context = self.current_task.metadata.get("bounded_context", {}) or {}
        manual_bounded = bounded_context.get("manual")
        plan_targets_bounded = bounded_context.get("plan_targets")
        bounded = manual_bounded or plan_targets_bounded
        if bounded:
            console.print()
            aggregate = bounded.get("aggregate", {}) if isinstance(bounded, dict) else {}
            impact = bounded.get("impact", {}) if isinstance(bounded, dict) else {}
            resolution = bounded.get("plan_targets_resolution") if isinstance(bounded, dict) else None

            scope_title = "Scoped Context (Bounded Index)" if manual_bounded else "Scoped Context (Plan Targets)"
            scope_lines = [
                f"Files: {aggregate.get('file_count', 0)}",
                f"Size: {aggregate.get('total_size_bytes', 0)} bytes",
                f"Languages: {', '.join(aggregate.get('top_languages', [])) or 'unknown'}",
            ]
            console.print(
                Panel.fit(
                    "\n".join(scope_lines),
                    title=scope_title,
                    border_style="magenta",
                ),
                markup=False,
            )

            # If we have an empty scope, explain why (plan targets are logical module names).
            if aggregate.get("file_count", 0) == 0 and isinstance(resolution, dict):
                raw_targets = resolution.get("raw_targets") or []
                resolved_targets = resolution.get("resolved_targets") or []
                unresolved_targets = resolution.get("unresolved_targets") or []

                explain_lines = [
                    "This scoped index is empty because plan targets are module labels (e.g. 'Api', 'Data')",
                    "and don't necessarily correspond to real directories/files under the indexed repo path.",
                ]
                if raw_targets:
                    explain_lines.append("")
                    explain_lines.append(f"Plan targets: {', '.join(str(t) for t in raw_targets[:12])}{' ...' if len(raw_targets) > 12 else ''}")
                if resolved_targets:
                    explain_lines.append(f"Resolved to paths: {', '.join(str(t) for t in resolved_targets[:12])}{' ...' if len(resolved_targets) > 12 else ''}")
                if unresolved_targets:
                    explain_lines.append(f"Unresolved: {', '.join(str(t) for t in unresolved_targets[:12])}{' ...' if len(unresolved_targets) > 12 else ''}")
                    explain_lines.append("")
                    explain_lines.append("Tip: run a bounded index on real paths (directories/files) you care about.")
                    explain_lines.append("Tip (chat): run scoped indexing on real paths (directories/files) you care about.")
                    explain_lines.append(f"CLI example: ./spec-agent bounded-index {self.current_task.id} <path1> <path2>")

                console.print(
                    Panel.fit(
                        "\n".join(explain_lines),
                        title="Why Files=0",
                        border_style="yellow",
                    ),
                    markup=False,
                )

            impact_lines = []
            top_dirs = impact.get("top_directories") or []
            namespaces = impact.get("namespaces") or []
            files_sample = impact.get("files_sample") or []
            if top_dirs:
                impact_lines.append(f"Top directories: {', '.join(top_dirs[:10])}")
            if namespaces:
                impact_lines.append(f"Namespaces: {', '.join(namespaces[:10])}")
            if files_sample:
                impact_lines.append("")
                impact_lines.append("Impacted files (sample):")
                for p in files_sample[:12]:
                    impact_lines.append(f"- {p}")

            if impact_lines:
                console.print(
                    Panel.fit(
                        "\n".join(impact_lines),
                        title="Impacted Modules / Files (Scoped)",
                        border_style="magenta",
                    ),
                    markup=False,
                )

        console.print()
        console.print("[bold]Plan Approval[/]")
        console.print()
        if stage == "FINAL":
            console.print("  [cyan]1.[/] Approve final plan (export Markdown to docs/)")
        else:
            console.print("  [cyan]1.[/] Approve preliminary plan (generate final plan with frozen scope)")
        console.print("  [cyan]2.[/] Reject and regenerate plan")
        console.print("  [cyan]3.[/] View boundary specs (details)")
        console.print("  [cyan]4.[/] Return to main menu")
        console.print()

        choice = self._ask_menu_choice("Choice", ["1", "2", "3", "4"], "1")

        if choice == "1":
            try:
                if stage != "FINAL":
                    # Approve preliminary plan: infer scope now, freeze it, then build final plan.
                    console.print("[yellow]Approving preliminary plan: inferring frozen scope...[/]")
                    inferred = self.orchestrator.infer_scope_targets(self.current_task.id)
                    targets = inferred.get("targets") or []
                    reason = inferred.get("reason") or "inferred"
                    sample_files = inferred.get("sample_files") or []

                    if not targets:
                        console.print("[yellow]Could not infer a scope automatically. You can reject the plan and refine the request, or run bounded-index manually.[/]")
                        return

                    lines = [f"Reason: {reason}", "", "Targets:"] + [f"- {t}" for t in targets[:12]]
                    if sample_files:
                        lines += ["", "Matched files (sample):"] + [f"- {p}" for p in sample_files[:12]]
                    console.print(Panel.fit("\n".join(lines), title="Inferred Frozen Scope", border_style="magenta"), markup=False)

                    if not Confirm.ask("Freeze scope to these targets and generate the final plan?", default=True):
                        console.print("[dim]Okay — scope not frozen. You can reject the preliminary plan or run bounded-index manually.[/]")
                        return

                    console.print("[yellow]Freezing scope (bounded index)...[/]")
                    summary = self.orchestrator.bounded_index_task(self.current_task.id, list(targets))
                    aggregate = (summary or {}).get("aggregate", {}) if isinstance(summary, dict) else {}
                    file_count = aggregate.get("file_count", 0) if isinstance(aggregate, dict) else 0
                    self.current_task = self.orchestrator._get_task(self.current_task.id)
                    if not (isinstance(file_count, int) and file_count > 0):
                        console.print("[yellow]Inferred scope matched 0 files; cannot generate final plan. Try rejecting/refining or run bounded-index manually.[/]")
                        return

                    console.print("[yellow]Generating final plan with frozen scope...[/]")
                    self.orchestrator.build_final_plan_with_frozen_scope(self.current_task.id)
                    self.current_task = self.orchestrator._get_task(self.current_task.id)
                    # Don't add a trailing '[/]' here; the green tag is already closed after ✓.
                    console.print("[green]✓[/] Final plan generated. Please review and approve.")
                    self.state = ConversationState.REVIEWING_PLAN
                    return

                # FINAL stage: approve + export + optional patches
                specs = self.orchestrator.get_boundary_specs(self.current_task.id)
                pending = [s for s in (specs or []) if s.get("status") == "PENDING"]
                if pending:
                    console.print()
                    console.print(
                        f"[yellow]{len(pending)} boundary spec(s) are still pending.[/]\n"
                        "[dim]Resolve or skip them before approving the final plan.[/]"
                    )
                    self.state = ConversationState.REVIEWING_SPECS
                    return

                self.orchestrator.approve_plan(self.current_task.id)
                console.print("[green]✓[/] Final plan approved!")

                console.print()
                exported = self.orchestrator.export_approved_plan_markdown(self.current_task.id)
                console.print(f"[green]✓[/] Saved approved plan to: [cyan]{exported}[/]")

                generate_now = Confirm.ask("Generate patches now? (you can do this later)", default=False)
                if generate_now:
                    has_tests = bool((self.current_task.metadata.get("repository_summary") or {}).get("has_tests", False))
                    if has_tests:
                        console.print("[cyan]Generating patches and test suggestions...[/]")
                    else:
                        console.print("[cyan]Generating patches...[/]")
                    patch_result = self.orchestrator.generate_patches(self.current_task.id)
                    patch_count = patch_result.get("patch_count", 0)
                    test_count = patch_result.get("test_count", 0)
                    tests_skipped_reason = patch_result.get("tests_skipped_reason")
                    if tests_skipped_reason:
                        console.print(f"[green]✓[/] Generated {patch_count} patch(es); test suggestions skipped ({tests_skipped_reason})")
                    else:
                        console.print(f"[green]✓[/] Generated {patch_count} patch(es) and {test_count} test suggestion(s)")

                    # Reload task to get updated metadata with patches
                    self.current_task = self.orchestrator._get_task(self.current_task.id)
                    self.state = ConversationState.REVIEWING_PATCHES
                else:
                    console.print("[green]✓[/] Ready to generate patches later.")
                    console.print(f"[dim]Run: ./spec-agent generate-patches {self.current_task.id}[/]")
                    self.state = ConversationState.MAIN_MENU
            except Exception as exc:
                console.print(f"Error approving plan: {exc}", style="red", markup=False)
        elif choice == "2":
            # Reject: route back through clarifications so we can tighten requirements,
            # then regenerate the plan with better inputs.
            console.print()
            console.print("[bold yellow]Plan rejected[/]")
            feedback = Prompt.ask(
                "What is missing/incorrect? (This will be used to generate clarifying questions)",
                default="",
            )
            try:
                self.orchestrator.restart_clarifications(self.current_task.id, reason=feedback)
                # Reload to get the newly generated clarifications
                self.current_task = self.orchestrator._get_task(self.current_task.id)
                console.print("[cyan]Returning to clarifying questions...[/]")
                self.state = ConversationState.CLARIFYING
            except Exception as exc:
                console.print(f"Failed to restart clarifications: {exc}", style="red", markup=False)
                console.print("[yellow]Falling back to plan regeneration...[/]")
                self.state = ConversationState.PLANNING
        elif choice == "3":
            # Show spec details but don't require approval
            self._show_spec_details()
        elif choice == "4":
            self.state = ConversationState.MAIN_MENU

    def _show_spec_details(self) -> None:
        """Show detailed boundary specifications for information only."""
        if not self.current_task:
            return

        specs = self.orchestrator.get_boundary_specs(self.current_task.id)

        if not specs:
            console.print()
            console.print("[yellow]No boundary specifications found.[/]")
            return

        for i, spec in enumerate(specs, 1):
            console.print()
            console.print("=" * 70)
            console.print(f"[bold cyan]Boundary Spec {i}/{len(specs)}: {spec.get('boundary_name')}[/]")
            console.print("=" * 70)
            console.print()

            console.print("[bold]Description:[/]")
            console.print(spec.get('human_description', 'No description'))
            console.print()

            console.print("[bold]Mermaid Diagram:[/]")
            console.print(f"[dim]{spec.get('diagram_text', 'No diagram')}[/]")
            console.print()

            machine_spec = spec.get('machine_spec', {})
            console.print("[bold]Machine Spec:[/]")
            console.print(f"  Actors: {', '.join(machine_spec.get('actors', []))}")
            console.print("  Interfaces:")
            for interface in machine_spec.get('interfaces', []):
                console.print(f"    • {interface}")
            console.print("  Invariants:")
            for invariant in machine_spec.get('invariants', []):
                console.print(f"    • {invariant}")

        console.print()
        console.print("[dim]Press Enter to return to plan review...[/]")
        input()

    def _handle_spec_review(self) -> None:
        """Handle boundary spec review and approval."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

        specs = self.orchestrator.get_boundary_specs(self.current_task.id)

        if not specs:
            console.print()
            console.print("[cyan]No boundary specifications needed. Returning to plan approval...[/]")
            self.state = ConversationState.REVIEWING_PLAN
            return

        pending_specs = [s for s in specs if s.get("status") == "PENDING"]

        if not pending_specs:
            console.print()
            console.print("[green]✓[/] All boundary specs resolved!")
            self.state = ConversationState.REVIEWING_PLAN
            return

        for spec in pending_specs:
            console.print()
            console.print("=" * 70)
            console.print(f"[bold]Boundary Spec: {spec.get('boundary_name')}[/]")
            console.print("=" * 70)
            console.print()

            console.print("[bold cyan]Description:[/]")
            console.print(spec.get('human_description', 'No description'))
            console.print()

            console.print("[bold cyan]Mermaid Diagram:[/]")
            console.print(f"[dim]{spec.get('diagram_text', 'No diagram')}[/]")
            console.print()

            machine_spec = spec.get('machine_spec', {})
            console.print("[bold cyan]Machine Spec:[/]")
            console.print(f"[bold]Actors:[/] {', '.join(machine_spec.get('actors', []))}")
            console.print("[bold]Interfaces:[/]")
            for interface in machine_spec.get('interfaces', []):
                console.print(f"  • {interface}")
            console.print("[bold]Invariants:[/]")
            for invariant in machine_spec.get('invariants', []):
                console.print(f"  • {invariant}")
            console.print()

            console.print("[bold]Actions:[/]")
            console.print("  [cyan]1.[/] Approve this spec")
            console.print("  [cyan]2.[/] Skip this spec")
            console.print("  [cyan]3.[/] Approve all remaining specs")
            console.print("  [cyan]4.[/] Skip all remaining specs")
            console.print()

            choice = self._ask_menu_choice("Choice", ["1", "2", "3", "4"], "1")

            if choice == "1":
                self.orchestrator.approve_spec(self.current_task.id, spec["id"])
                console.print("[green]✓[/] Spec approved")
            elif choice == "2":
                self.orchestrator.skip_spec(self.current_task.id, spec["id"])
                console.print("[yellow]Spec skipped[/]")
            elif choice == "3":
                self.orchestrator.approve_all_specs(self.current_task.id)
                console.print("[green]✓[/] All specs approved")
                break
            elif choice == "4":
                self.orchestrator.skip_all_specs(self.current_task.id)
                console.print("[yellow]All specs skipped[/]")
                break

        console.print()
        console.print("[green]✓[/] All boundary specs resolved!")
        console.print("[cyan]Returning to plan approval...[/]")
        self.state = ConversationState.REVIEWING_PLAN

    def _handle_patch_review(self) -> None:
        """Handle patch review and approval."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

        console.print()
        console.print("[cyan]Patch review would go here...[/]")
        console.print("[dim]This will be similar to the patches command but interactive[/]")
        console.print()

        # For now, just return to main menu
        console.print("[green]✓[/] Task workflow complete![/]")
        console.print(f"[cyan]Task ID: {self.current_task.id}[/]")
        console.print()

        if Confirm.ask("Return to main menu?", default=True):
            self.state = ConversationState.MAIN_MENU

    def _continue_task(self) -> None:
        """Continue an existing task."""
        tasks = self.orchestrator.list_tasks()

        if not tasks:
            console.print()
            console.print("[yellow]No existing tasks found.[/]")
            return

        console.print()
        console.print("[bold]Select a task to continue:[/]")
        console.print()

        for i, task in enumerate(tasks[:10], 1):
            status_color = {
                "CLARIFYING": "yellow",
                "PLANNING": "cyan",
                "SPEC_PENDING": "blue",
                "IMPLEMENTING": "green"
            }.get(task.status.value, "white")

            title = (task.title or "").strip() or (task.description.splitlines()[0] if task.description else f"task-{task.id[:8]}")
            console.print(
                f"  [cyan]{i}.[/] [{status_color}]{task.status.value}[/] - {(task.client or '—')[:10]} - {title[:50]}"
            )

        console.print()
        choice = Prompt.ask("Choice", default="1")

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(tasks):
                self.current_task = tasks[idx]
                self.indexed_repo = self.current_task.repo_path
                self.indexed_branch = self.current_task.branch

                # Optional: allow updating the description when resuming.
                if Confirm.ask("Update task description before continuing?", default=False):
                    new_desc = Prompt.ask("New description", default=self.current_task.description).strip()
                    if new_desc and new_desc != self.current_task.description:
                        self.current_task = self.orchestrator.update_task_description(
                            self.current_task.id,
                            new_desc,
                            reason="Updated during interactive resume",
                            reset_metadata=True,
                        )

                # Reload to pick up any auto-resets (e.g. tasks.json description edits)
                self.current_task = self.orchestrator._get_task(self.current_task.id)

                # Determine state based on task status
                if self.current_task.status == TaskStatus.CLARIFYING:
                    self.state = ConversationState.CLARIFYING
                elif self.current_task.status == TaskStatus.PLANNING:
                    self.state = ConversationState.PLANNING
                elif self.current_task.status == TaskStatus.SPEC_PENDING:
                    self.state = ConversationState.REVIEWING_SPECS
                else:
                    self.state = ConversationState.REVIEWING_PATCHES
            else:
                console.print("[red]Invalid choice[/]")
        except ValueError:
            console.print("[red]Invalid choice[/]")

    def _show_task_history(self) -> None:
        """Show task history."""
        tasks = self.orchestrator.list_tasks()

        if not tasks:
            console.print()
            console.print("[yellow]No tasks found.[/]")
            return

        console.print()
        table = Table(title="Task History")
        table.add_column("Status", style="bold")
        table.add_column("Client")
        table.add_column("Title")
        table.add_column("Description")
        table.add_column("Updated")

        for task in tasks[:10]:
            status_color = {
                "CLARIFYING": "yellow",
                "PLANNING": "cyan",
                "SPEC_PENDING": "blue",
                "IMPLEMENTING": "green"
            }.get(task.status.value, "white")

            table.add_row(
                f"[{status_color}]{task.status.value}[/]",
                (task.client or "—")[:10],
                ((task.title or "").strip() or (task.description.splitlines()[0] if task.description else f"task-{task.id[:8]}"))[:40],
                task.description[:60],
                task.updated_at.strftime("%Y-%m-%d %H:%M")
            )

        console.print(table)
        console.print()
