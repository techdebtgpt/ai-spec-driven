from __future__ import annotations

from enum import Enum
import shlex
import subprocess
from pathlib import Path
from typing import Optional
from threading import Lock, Thread
from uuid import uuid4
import socket
from datetime import datetime, timezone

from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from ..domain.models import Task, TaskStatus, ClarificationStatus
from ..workflow.orchestrator import TaskOrchestrator
from .dashboard import run_task_dashboard
from ..web.server import run_dashboard_server


console = Console()


def _safe(text: Optional[str]) -> str:
    """Escape user-provided text for Rich markup output."""
    return escape(text or "")


def _format_relative(ts: str | None) -> str:
    if not ts:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return ts
    now = datetime.now(timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


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
        # When we already asked for repo/branch in the main menu and need to index due to
        # missing cache, skip re-prompting inside the INDEXING handler.
        self._indexing_preselected: bool = False
        self._bg_lock = Lock()
        self._bg_jobs: dict[str, dict] = {}
        self._web_dashboard_url: str | None = None
        self._prefill_description: str = ""
        self._prefill_clarification_answers: list[dict] = []
        self._current_index_data: Optional[dict] = None
        self._maybe_load_cached_index()

    def _recent_repositories(self) -> list[dict]:
        options: list[dict] = []
        try:
            cached = self.orchestrator.list_cached_indexes()
        except Exception:
            return options
        seen: set[tuple[str, str]] = set()
        for data in cached:
            repo_path = str(data.get("repo_path") or "").strip()
            branch = str(data.get("branch") or "main")
            if not repo_path:
                continue
            key = (repo_path, branch)
            if key in seen:
                continue
            seen.add(key)
            options.append(
                {
                    "repo_path": Path(repo_path),
                    "branch": branch,
                    "indexed_at": data.get("indexed_at"),
                    "index_data": data,
                }
            )
            if len(options) >= 7:
                break
        return options

    def _find_recent_task(self, repo_root: Path, branch: str) -> Optional[Task]:
        try:
            tasks = self.orchestrator.store.load_tasks()
        except Exception:
            return None
        repo_resolved = repo_root.resolve()
        latest: Optional[Task] = None
        for task in tasks:
            try:
                task_repo_path = task.repo_path.resolve()
            except Exception:
                task_repo_path = Path(str(task.repo_path))
            matches_repo = task_repo_path == repo_resolved
            if not matches_repo:
                try:
                    matches_repo = repo_resolved.is_relative_to(task_repo_path) or task_repo_path.is_relative_to(repo_resolved)
                except ValueError:
                    matches_repo = False
            if not matches_repo:
                continue
            if (task.branch or "") != (branch or ""):
                continue
            if not latest or task.updated_at > latest.updated_at:
                latest = task
        return latest

    def _apply_saved_clarifications(self) -> int:
        """
        Reapply previously answered clarifications when the engineer reuses a description.
        """
        if not self.current_task or not self._prefill_clarification_answers:
            return 0

        answers_map = {}
        for item in self._prefill_clarification_answers:
            question = str(item.get("question") or "").strip().lower()
            answer = str(item.get("answer") or "").strip()
            if question and answer:
                answers_map[question] = answer

        if not answers_map:
            return 0

        clarifications = self.current_task.metadata.get("clarifications", []) or []
        reused = 0
        for item in clarifications:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().upper()
            if status and status != "PENDING":
                continue
            question = str(item.get("question") or "").strip()
            key = question.lower()
            answer = answers_map.get(key)
            if not answer:
                continue
            try:
                self.orchestrator.update_clarification(
                    self.current_task.id,
                    item["id"],
                    answer,
                    ClarificationStatus.ANSWERED,
                )
                reused += 1
            except Exception:
                continue
        return reused

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

    def _ask_menu_choice(self, prompt: str, choices: list[str], default: Optional[str]) -> str:
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
            return default or ""

        # If the input contains a valid option anywhere (e.g. "1 ", "1\n", or stray chars),
        # pick the first matching choice.
        for ch in value:
            if ch in choices:
                return ch

        return default or ""

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
        console.print("  [cyan]5.[/] Open dashboard (web)")
        console.print("  [cyan]6.[/] Exit")
        console.print()

        try:
            has_tasks = bool(self.orchestrator.list_tasks())
        except Exception:
            has_tasks = False

        # Require explicit selection to avoid auto-starting new tasks.
        choice = self._ask_menu_choice("Choice", ["1", "2", "3", "4", "5", "6"], None)

        if not choice:
            return

        if choice == "1":
            self.index_only = False
            console.print()

            repo_root: Path | None = None
            branch_default = self.indexed_branch or "main"
            branch: str = branch_default
            current_branch: str | None = None

            recent = self._recent_repositories()
            selected_cached: dict | None = None
            if recent:
                console.print("Cached repositories:")
                for idx, item in enumerate(recent, start=1):
                    when = _format_relative(item.get("indexed_at"))
                    console.print(f"  [cyan]{idx}.[/] {item['repo_path']} ({item['branch']}) · indexed {when}")
                console.print(f"  [cyan]{len(recent)+1}.[/] Enter another path")
                selection = Prompt.ask("Select repository", default="1").strip()
                if selection.isdigit():
                    idx = int(selection)
                    if 1 <= idx <= len(recent):
                        selected_cached = recent[idx - 1]
                    elif idx != len(recent) + 1:
                        # default to first if invalid
                        selected_cached = recent[0]

            if selected_cached:
                repo_root = selected_cached["repo_path"].resolve()
                cached_branch = (selected_cached.get("branch") or "").strip()
                try:
                    current_branch = self.orchestrator._get_current_branch(repo_root)
                except Exception:
                    current_branch = None

                # Prefer the live branch when available; fall back to cached.
                branch_default = current_branch or cached_branch or branch_default

                # Only reuse cached index when branch matches; otherwise force reindex.
                if current_branch and cached_branch and current_branch == cached_branch:
                    self._current_index_data = dict(selected_cached.get("index_data") or {})
                else:
                    self._current_index_data = None
                    if cached_branch and current_branch and cached_branch != current_branch:
                        console.print(
                            f"[yellow]Cached index is for branch '{cached_branch}', current branch is '{current_branch}'. Will re-run indexing for the current branch.[/]")
            else:
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
                self._current_index_data = None

            branch = Prompt.ask("Branch", default=branch_default).strip()
            cached_branch = str(selected_cached.get("branch") or "").strip() if selected_cached else ""

            if current_branch and branch != current_branch:
                console.print(
                    f"[yellow]Current branch is '{current_branch}', requested '{branch}'.[/]"
                )
                if Confirm.ask(f"Switch working tree to '{branch}'?", default=False):
                    try:
                        subprocess.run([
                            "git",
                            "checkout",
                            branch,
                        ], check=True, cwd=repo_root)
                        current_branch = branch
                        console.print(f"[green]✓[/] Checked out branch '{branch}'.")
                    except subprocess.CalledProcessError as exc:
                        console.print(f"[red]Failed to checkout '{branch}': {exc}[/]")
                        branch = current_branch
                else:
                    console.print(f"[dim]Keeping current branch '{current_branch}'.[/]")
                    branch = current_branch

            branch_changed = bool(selected_cached and cached_branch and cached_branch != branch)

            if not self._current_index_data and not branch_changed:
                try:
                    self._current_index_data = self.orchestrator.get_repository_index_for_repo_and_branch(repo_root, branch)
                except Exception:
                    self._current_index_data = None

            self._prefill_description = ""
            self._prefill_clarification_answers = []
            self._current_index_data = None
            previous_task = self._find_recent_task(repo_root, branch)
            if previous_task and previous_task.description:
                preview = previous_task.description.strip().splitlines()[0][:80]
                console.print(f"[dim]Last task for this repo/branch:[/] {preview}")
                reuse_choice = Prompt.ask("Reuse previous description? (y/n/edit)", default="y").strip().lower()
                if reuse_choice.startswith("y") or reuse_choice == "":
                    self._prefill_description = previous_task.description.strip()
                    answers = []
                    clar_list = previous_task.metadata.get("clarifications", [])
                    if isinstance(clar_list, list):
                        for item in clar_list:
                            if not isinstance(item, dict):
                                continue
                            if str(item.get("status")).upper() == "ANSWERED" and item.get("answer"):
                                answers.append(
                                    {
                                        "question": str(item.get("question") or ""),
                                        "answer": str(item.get("answer") or ""),
                                    }
                                )
                    self._prefill_clarification_answers = answers
                    console.print("[dim]Will reuse previous description and answers. You can edit them in the next step.[/]")
                elif reuse_choice.startswith("e"):
                    edited = Prompt.ask("Description", default=previous_task.description).strip()
                    if edited:
                        self._prefill_description = edited
                # if 'n', leave defaults empty

            # Try to reuse an existing cached index for this repo unless the branch changed.
            try:
                if branch_changed:
                    raise ValueError("Branch changed from cached selection; forcing re-index")

                cached = self.orchestrator.get_repository_index_for_repo_and_branch(repo_root, branch)
                self.indexed_repo = Path(cached.get("repo_path") or repo_root).resolve()
                self.indexed_branch = str(cached.get("branch") or branch or "main")
                self._current_index_data = cached
                console.print(f"[cyan]Using cached index for:[/] {self.indexed_repo} ({self.indexed_branch})")
                self._indexing_preselected = False
                self.state = ConversationState.TASK_SETUP
            except Exception:
                # No cached index or branch switched: go index now.
                reason = "branch change" if branch_changed else f"{branch} cache"
                console.print(f"[yellow]No usable cached index for {repo_root} ({branch}) [{reason}]. Indexing now...[/]")
                self.indexed_repo = repo_root
                self.indexed_branch = branch
                self._indexing_preselected = True
                self.state = ConversationState.INDEXING
        elif choice == "2":
            self._continue_task()
        elif choice == "3":
            self._show_task_history()
        elif choice == "4":
            self.index_only = True
            self._indexing_preselected = False
            self.state = ConversationState.INDEXING
        elif choice == "5":
            self._open_web_dashboard()
            self.state = ConversationState.MAIN_MENU
        elif choice == "6":
            self.state = ConversationState.EXITING

    def _open_web_dashboard(self) -> None:
        """
        Start the web dashboard in the background and open it in the browser.
        """
        import webbrowser

        if self._web_dashboard_url:
            console.print(f"[dim]Dashboard already running:[/] {self._web_dashboard_url}")
            try:
                webbrowser.open(self._web_dashboard_url)
            except Exception:
                pass
            return

        # Find a free port (avoid conflicts during demos).
        host = "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = int(s.getsockname()[1])

        url = f"http://{host}:{port}"
        self._web_dashboard_url = url

        console.print()
        console.print(f"[cyan]Opening dashboard:[/] {url}")

        # Start server in background thread.
        self._start_background_job(
            name=f"web-dashboard ({url})",
            fn=lambda: run_dashboard_server(host=host, port=port),
        )

        try:
            webbrowser.open(url)
        except Exception:
            pass

        console.print("[dim]Tip: keep this terminal open while the dashboard is running.[/]")

    def _handle_indexing(self) -> None:
        """Handle repository indexing."""
        console.print()
        console.print("[bold cyan]Repository Setup[/]")
        console.print()

        # If we already asked for repo/branch in the main menu (start task flow), don't ask again.
        if self._indexing_preselected and self.indexed_repo:
            repo_path = self.indexed_repo
            branch = self.indexed_branch or "main"
            console.print(f"[dim]Using selected repo:[/] {repo_path}")
            console.print(f"[dim]Using selected branch:[/] {branch}")
        else:
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
            branch = Prompt.ask("Branch", default=branch_default).strip()

        # Align branch with the current git branch to avoid stale caches.
        try:
            resolved_repo_root = self.orchestrator.resolve_repo_root(repo_path)
            git_branch = self.orchestrator._get_current_branch(resolved_repo_root)
        except Exception:
            git_branch = None

        if git_branch and git_branch.lower() != "unknown" and branch != git_branch:
            console.print(f"[yellow]Branch mismatch: using current git branch '{git_branch}' instead of '{branch}'.[/]")
            branch = git_branch

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
            # Clear the preselected indexing flag after a successful run.
            self._indexing_preselected = False

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
                try:
                    self._current_index_data = self.orchestrator.get_repository_index_for_repo_and_branch(self.indexed_repo, self.indexed_branch)
                except Exception:
                    self._current_index_data = None
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

        default_desc = (self._prefill_description or "").strip()
        if default_desc:
            description = Prompt.ask("Description", default=default_desc).strip()
        else:
            description = Prompt.ask("Description").strip()
        self._prefill_description = ""

        if not description or description.strip() == "":
            console.print("[yellow]Description cannot be empty.[/]")
            return

        if not self.indexed_repo:
            console.print("[red]Please index a repository before creating a task.[/]")
            self.state = ConversationState.MAIN_MENU
            return

        # If an existing task matches the same repo/branch/description, resume it instead of creating a duplicate.
        try:
            repo_resolved = self.indexed_repo.resolve()
        except Exception:
            repo_resolved = self.indexed_repo

        matching: Optional[Task] = None
        try:
            for task in self.orchestrator.list_tasks():
                try:
                    task_repo = task.repo_path.resolve()
                except Exception:
                    task_repo = Path(str(task.repo_path))

                if task_repo != repo_resolved:
                    try:
                        if not (repo_resolved.is_relative_to(task_repo) or task_repo.is_relative_to(repo_resolved)):
                            continue
                    except ValueError:
                        continue

                if (task.branch or "") != (self.indexed_branch or ""):
                    continue
                if (task.description or "").strip() != description:
                    continue
                if not matching or task.updated_at > matching.updated_at:
                    matching = task
        except Exception:
            matching = None

        if matching:
            console.print()
            console.print("[cyan]Found an existing task with the same repo, branch, and description. Resuming it instead of creating a new one.[/]")
            self.current_task = matching
            self.indexed_repo = matching.repo_path
            self.indexed_branch = matching.branch

            if matching.status == TaskStatus.CLARIFYING:
                self.state = ConversationState.CLARIFYING
            elif matching.status == TaskStatus.PLANNING:
                self.state = ConversationState.PLANNING
            elif matching.status == TaskStatus.SPEC_PENDING:
                self.state = ConversationState.REVIEWING_SPECS
            else:
                self.state = ConversationState.REVIEWING_PATCHES
            return

        command = f"./spec-agent start --description {shlex.quote(description)}"
        self._show_cli_command(command)

        # Create the task using the indexed repository metadata
        console.print()
        console.print("[yellow]Creating task...[/]")

        try:
            self.current_task = self.orchestrator.create_task_from_index(
                description=description,
                index_data=self._current_index_data,
            )
            self.indexed_repo = self.current_task.repo_path
            self.indexed_branch = self.current_task.branch

            title = (self.current_task.title or "").strip() or (self.current_task.description.splitlines()[0] if self.current_task.description else "Untitled task")
            console.print(f"[green]✓[/] Task created: [cyan]{title}[/]")
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

        clarifications = self.current_task.metadata.get("clarifications", []) or []

        if self._prefill_clarification_answers:
            reused = self._apply_saved_clarifications()
            if reused:
                console.print(f"[dim]Reused {reused} previous clarification answer(s).[/]")
                self.current_task = self.orchestrator._get_task(self.current_task.id)
                clarifications = self.current_task.metadata.get("clarifications", []) or []
            self._prefill_clarification_answers = []

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

            while True:
                answer = Prompt.ask(
                    "Your answer (type 'skip' to move on)",
                    default=None,
                    show_default=False,
                )

                normalized = (answer or "").strip()

                if normalized.lower() == "skip":
                    console.print("[yellow]Skipped[/]")
                    item["status"] = "OVERRIDDEN"
                    break
                if normalized:
                    item["answer"] = normalized
                    item["status"] = "ANSWERED"
                    console.print("[green]✓[/] Got it!")
                    break

                console.print("[yellow]Please provide an answer or type 'skip' to continue.[/]")

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
            def progress(msg: str) -> None:
                # Render plan progress in the same "note" style used elsewhere.
                console.print(f"[dim]{msg}[/]")

            self.orchestrator.generate_plan(self.current_task.id, progress=progress)

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
                title="Implementation Plan",
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

        # Show scoped files as a dedicated panel (like other sections), but compact.
        bounded_context = self.current_task.metadata.get("bounded_context", {}) or {}
        manual_bounded = bounded_context.get("manual")
        plan_targets_bounded = bounded_context.get("plan_targets")
        bounded = manual_bounded or plan_targets_bounded
        if bounded:
            aggregate = bounded.get("aggregate", {}) if isinstance(bounded, dict) else {}
            impact = bounded.get("impact", {}) if isinstance(bounded, dict) else {}
            scope_meta = bounded.get("scope", {}) if isinstance(bounded, dict) else {}
            file_count = aggregate.get("file_count", 0)
            files_sample = impact.get("files_sample") or []
            allowed_files = scope_meta.get("allowed_files") or []
            target_list = scope_meta.get("targets") or []

            if not files_sample and allowed_files:
                files_sample = allowed_files

            if isinstance(file_count, int) and file_count > 0:
                console.print()
                scope_source = "scoped index" if manual_bounded else "plan targets"

                scope_lines = [f"Scope: {file_count} file(s) ({scope_source})"]
                if target_list:
                    scope_lines.append("")
                    scope_lines.append("Scope targets:")
                    scope_lines.extend([f"- {_safe(str(p))}" for p in target_list[:8]])
                if files_sample:
                    scope_lines.append("")
                    scope_lines.append("Impacted files (sample):")
                    scope_lines.extend([f"- {_safe(str(p))}" for p in files_sample[:12]])

                console.print(
                    Panel.fit(
                        "\n".join(scope_lines),
                        title="Impacted files (scope)",
                        border_style="green",
                    ),
                    markup=False,
                )

        console.print()
        console.print("[bold]Plan Approval[/]")
        console.print()
        console.print("  [cyan]1.[/] Approve plan")
        console.print("  [cyan]2.[/] Reject and regenerate plan")
        console.print("  [cyan]3.[/] View boundary specs (details)")
        console.print("  [cyan]4.[/] Return to main menu")
        console.print()

        choice = self._ask_menu_choice("Choice", ["1", "2", "3", "4"], "1")

        if choice == "1":
            try:
                if stage != "FINAL":
                    # Plan is still preliminary - need to freeze scope first
                    # (This path is only taken if auto_freeze_scope=False was used)
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
                        f"[dim]Auto-approving {len(pending)} pending boundary spec(s) before final approval...[/]"
                    )
                    try:
                        self.orchestrator.approve_all_specs(self.current_task.id)
                        self.current_task = self.orchestrator._get_task(self.current_task.id)
                        console.print("[green]✓[/] Boundary specs approved")
                    except Exception as exc:
                        console.print(f"[yellow]Could not auto-approve specs: {exc}[/]")
                        self.state = ConversationState.REVIEWING_SPECS
                        return

                self.orchestrator.approve_plan(self.current_task.id)
                console.print("[green]✓[/] Plan approved!")

                console.print()
                exported = self.orchestrator.export_approved_plan_markdown(self.current_task.id)
                console.print(f"[green]✓[/] Plan exported to: [cyan]{exported}[/]")

                console.print()
                console.print("[bold]Plan exported. What would you like to do next?[/]")
                console.print()
                console.print("  [cyan]1.[/] Generate patches (next milestone)")
                console.print("  [cyan]2.[/] Open the plan file")
                console.print("  [cyan]3.[/] Generate patches + apply in Cursor (MCP) + sync back")
                console.print("  [cyan]4.[/] Return to main menu")
                console.print()

                next_choice = self._ask_menu_choice("Choice", ["1", "2", "3", "4"], "4")

                if next_choice == "2":
                    try:
                        import subprocess
                        import sys

                        # Best-effort: open the file in the OS default app.
                        if sys.platform == "darwin":
                            subprocess.run(["open", str(exported)], check=False)
                        elif sys.platform.startswith("linux"):
                            subprocess.run(["xdg-open", str(exported)], check=False)
                        else:
                            console.print(f"[dim]Plan file path:[/] {exported}")
                    except Exception:
                        console.print(f"[dim]Plan file path:[/] {exported}")
                    self.state = ConversationState.MAIN_MENU
                    return

                if next_choice == "1":
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
                elif next_choice == "3":
                    # Option B: generate patches, then apply in Cursor (MCP) and sync back.
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

                    patches = self.orchestrator.list_patches(self.current_task.id)
                    pending = [p for p in patches if p.status.value == "PENDING"]
                    example_patch = pending[0] if pending else (patches[0] if patches else None)

                    panel_lines = [
                        "[bold]Cursor (MCP) apply flow[/]",
                        "",
                        "1) Open Cursor on the repo and apply the patch changes in the editor.",
                        "2) After edits, run sync so Spec Agent records the real git diff:",
                        "",
                        f"   ./spec-agent sync-external {self.current_task.id}"
                        + (f" --patch-id {example_patch.id}" if len(pending) == 1 and example_patch else "")
                        + " --client cursor",
                        "",
                        "Tip: If you have Spec Agent MCP installed in Cursor, you can also call:",
                        f"   sync_external_patch(task_id=\"{self.current_task.id}\", patch_id=\"{example_patch.id if len(pending) == 1 and example_patch else ''}\", client=\"cursor\")",
                        "",
                    ]

                    if pending:
                        panel_lines.extend(
                            [
                                "[bold]Pending patches (apply all):[/]",
                                *[f"  - {p.id}: {p.step_reference}" for p in pending],
                                "",
                                "Apply each pending patch in order. For every patch:",
                                "  a) Call get_patch_details and apply the diff to the workspace files.",
                                "  b) Do NOT call approve_patch from Cursor.",
                                f"  c) Run sync_external_patch(task_id=\"{self.current_task.id}\", patch_id=\"<PATCH_ID>\", client=\"cursor\") after each patch.",
                                "",
                            ]
                        )

                    pending_ids = ", ".join(p.id for p in pending) if pending else "(none)"
                    cursor_prompt = (
                        f"Use spec-agent MCP tools. For task {self.current_task.id}, apply ALL pending patches (in order): {pending_ids}. "
                        "For each patch: get_patch_details, apply the diff to the workspace files, do NOT call approve_patch, "
                        f"and then call sync_external_patch(task_id=\"{self.current_task.id}\", patch_id=\"<PATCH_ID>\", client=\"cursor\")."
                    )

                    panel_lines.extend(
                        [
                            "[bold]Copy/paste into Cursor chat (recommended):[/]",
                            f"  \"{cursor_prompt}\"",
                        ]
                    )

                    console.print()
                    console.print(Panel.fit("\n".join(panel_lines), title="Option B: Apply in Cursor + Sync"))

                    # Best-effort: open Cursor on macOS (non-fatal if it fails).
                    try:
                        import subprocess
                        import sys
                        repo_path = str(self.current_task.repo_path)
                        if sys.platform == "darwin":
                            subprocess.run(["open", "-a", "Cursor", repo_path], check=False)
                    except Exception:
                        pass

                    # Auto-sync (best-effort): watch for git changes and sync when a single pending patch exists.
                    if len(pending) == 1 and example_patch:
                        try:
                            import time
                            import subprocess

                            repo_path = str(self.current_task.repo_path)
                            task_id = self.current_task.id
                            patch_id = example_patch.id

                            def _git_status_short() -> str:
                                try:
                                    r = subprocess.run(
                                        ["git", "status", "--short"],
                                        capture_output=True,
                                        text=True,
                                        check=True,
                                        cwd=repo_path,
                                    )
                                    return (r.stdout or "").strip()
                                except Exception:
                                    return ""

                            baseline = _git_status_short()

                            def _watch_and_sync():
                                # Wait up to 20 minutes for Cursor edits (demo-friendly).
                                deadline = time.time() + 20 * 60
                                while time.time() < deadline:
                                    current = _git_status_short()
                                    if current != baseline and current.strip():
                                        # Capture and persist the real diff on the patch/task.
                                        return self.orchestrator.sync_external_patch(
                                            task_id,
                                            patch_id=patch_id,
                                            client="cursor",
                                            include_staged=True,
                                        )
                                    time.sleep(2.0)
                                return {"timeout": True}

                            self._start_background_job(
                                name=f"auto-sync (cursor) {task_id[:8]}",
                                fn=_watch_and_sync,
                            )
                            console.print("[dim]Watching for Cursor edits… will auto-sync when changes are detected.[/]")
                        except Exception:
                            # Non-fatal; user can always run sync-external manually.
                            pass
                    else:
                        console.print(
                            "[dim]Multiple pending patches detected; run sync-external after each patch to mark them applied.[/]"
                        )

                    # Return to main menu; patches are now generated and can be synced later.
                    self.state = ConversationState.MAIN_MENU
                else:
                    console.print("[green]✓[/] Next milestone: patch generation.")
                    console.print("[dim]You can generate patches later from the main menu.[/]")
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

        patches = self.orchestrator.list_patches(self.current_task.id)

        console.print()
        if patches:
            console.print("[green]Patches generated and ready for review.[/]")
            console.print(f"[dim]Use './spec-agent patches {self.current_task.id}' to inspect and apply them in your editor.[/]")
            console.print()
            console.print("[bold]Copy/paste into Cursor or Claude MCP:[/]")
            console.print(
                f"  Use spec-agent MCP tools to apply ALL pending patches for task {self.current_task.id}:\n"
                f"  1) Call get_patch_details for each pending patch in order.\n"
                f"  2) Apply the diff to your workspace files.\n"
                f"  3) Do NOT call approve_patch from the editor.\n"
                f"  4) After each patch, call sync_external_patch(task_id=\"{self.current_task.id}\", patch_id=\"<PATCH_ID>\", client=\"cursor\")\n"
                f"  5) Repeat for all pending patches."
            )
            self.state = ConversationState.MAIN_MENU
            return

        console.print("[yellow]No patches have been generated for this task yet.[/]")
        if Confirm.ask("Generate patches now?", default=True):
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
            console.print(f"[dim]Use './spec-agent patches {self.current_task.id}' to inspect and apply them in your editor.[/]")
        else:
            console.print("[dim]Okay — you can generate patches later from the main menu.[/]")

        self.state = ConversationState.MAIN_MENU

    def _continue_task(self) -> None:
        """Continue an existing task."""
        tasks = sorted(
            self.orchestrator.list_tasks(),
            key=lambda t: t.updated_at,
            reverse=True,
        )

        if not tasks:
            console.print()
            console.print("[yellow]No existing tasks found.[/]")
            return

        console.print()
        console.print("[bold]Select a task to continue:[/]")
        console.print()
        console.print("[dim]Pick a task from this list; its repository and branch will be loaded automatically.[/]")
        console.print()

        for i, task in enumerate(tasks[:10], 1):
            status_color = {
                "CLARIFYING": "yellow",
                "PLANNING": "cyan",
                "SPEC_PENDING": "blue",
                "IMPLEMENTING": "green"
            }.get(task.status.value, "white")

            title = (task.title or "").strip() or (task.description.splitlines()[0] if task.description else f"task-{task.id[:8]}")
            repo_display = _safe(str(task.repo_path))
            branch_display = _safe(task.branch)
            title_display = _safe(title[:60])
            client_display = _safe((task.client or '—')[:12])
            updated = _safe(_format_relative(task.updated_at.isoformat()))

            console.print(f"  [cyan]{i}.[/] [{status_color}]{task.status.value}[/] • {branch_display}")
            console.print(f"      Repo: {repo_display}")
            console.print(f"      Title: {title_display}")
            console.print(f"      Client: {client_display} • Updated: {updated}")
            console.print()

        console.print()
        choice = Prompt.ask("Choice", default="1")

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(tasks):
                self.current_task = tasks[idx]
                self.indexed_repo = self.current_task.repo_path
                self.indexed_branch = self.current_task.branch
                console.print(
                    f"[dim]Resuming task {self.current_task.id[:8]} in {escape(str(self.indexed_repo))} ({escape(self.indexed_branch)}).[/]"
                )

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
        tasks = sorted(
            self.orchestrator.list_tasks(),
            key=lambda t: t.updated_at,
            reverse=True,
        )

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
                _safe((task.client or "—")[:10]),
                _safe(((task.title or "").strip() or (task.description.splitlines()[0] if task.description else f"task-{task.id[:8]}"))[:40]),
                _safe((task.description or "")[:60]),
                task.updated_at.strftime("%Y-%m-%d %H:%M")
            )

        console.print(table)
        console.print()
