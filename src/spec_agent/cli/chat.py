from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from ..domain.models import Task, TaskStatus
from ..workflow.orchestrator import TaskOrchestrator


console = Console()


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

    def run(self) -> None:
        """Main conversation loop."""
        self._show_welcome()

        while self.state != ConversationState.EXITING:
            try:
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
                console.print(f"[red]Error: {exc}[/]")
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
        console.print("  [cyan]5.[/] Exit")
        console.print()

        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "1":
            self.index_only = False
            # If we already have an indexed repo, skip indexing and go straight to task setup
            if self.indexed_repo and self.indexed_repo.exists():
                console.print()
                console.print(f"[cyan]Using indexed repository: {self.indexed_repo}[/]")
                self.state = ConversationState.TASK_SETUP
            else:
                self.state = ConversationState.INDEXING
        elif choice == "2":
            self._continue_task()
        elif choice == "3":
            self._show_task_history()
        elif choice == "4":
            self.index_only = True
            self.state = ConversationState.INDEXING
        elif choice == "5":
            self.state = ConversationState.EXITING

    def _handle_indexing(self) -> None:
        """Handle repository indexing."""
        console.print()
        console.print("[bold cyan]Repository Setup[/]")
        console.print()

        # Ask for repository path with tab completion
        default_path = str(Path.cwd())
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

        # Ask for branch
        branch = Prompt.ask("Branch", default="main")

        # Index the repository
        console.print()
        console.print("[yellow]Indexing repository...[/]")

        try:
            index_data = self.orchestrator.index_repository(repo_path=repo_path, branch=branch)
            summary = index_data.get("repository_summary", {})

            # Show summary
            console.print()
            console.print(Panel.fit(
                f"[green]✓[/] Repository indexed successfully!\n\n"
                f"Files: {summary.get('file_count', 0):,}\n"
                f"Directories: {summary.get('directory_count', 0):,}\n"
                f"Languages: {', '.join(summary.get('top_languages', [])[:3]) or 'unknown'}",
                title="Index Complete",
                border_style="green"
            ))

            self.indexed_repo = repo_path
            self.indexed_branch = branch

            # If just indexing, return to menu. Otherwise, proceed to task setup
            if self.index_only:
                console.print()
                console.print("[cyan]Repository indexed! You can now start a task.[/]")
                self.state = ConversationState.MAIN_MENU
            else:
                self.state = ConversationState.TASK_SETUP

        except Exception as exc:
            console.print(f"[red]Failed to index repository: {exc}[/]")
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

        # Create the task
        console.print()
        console.print("[yellow]Creating task...[/]")

        try:
            self.current_task = self.orchestrator.create_task(
                repo_path=self.indexed_repo,
                branch=self.indexed_branch,
                description=description
            )

            console.print(f"[green]✓[/] Task created: [cyan]{self.current_task.id}[/]")
            self.state = ConversationState.CLARIFYING

        except Exception as exc:
            console.print(f"[red]Failed to create task: {exc}[/]")
            self.state = ConversationState.MAIN_MENU

    def _handle_clarifying(self) -> None:
        """Handle clarifying questions."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

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

        # Proceed to planning
        self.state = ConversationState.PLANNING

    def _handle_planning(self) -> None:
        """Handle plan generation."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

        console.print()
        console.print("[yellow]Generating implementation plan...[/]")

        try:
            plan_data = self.orchestrator.generate_plan(self.current_task.id)

            # Reload the task to get updated metadata
            self.current_task = self.orchestrator._get_task(self.current_task.id)

            console.print("[green]✓[/] Plan generated!")
            self.state = ConversationState.REVIEWING_PLAN

        except Exception as exc:
            console.print(f"[red]Failed to generate plan: {exc}[/]")
            self.state = ConversationState.MAIN_MENU

    def _handle_plan_review(self) -> None:
        """Handle plan review and approval."""
        if not self.current_task:
            self.state = ConversationState.MAIN_MENU
            return

        plan_preview = self.current_task.metadata.get("plan_preview", {})
        steps = plan_preview.get("steps", [])
        risks = plan_preview.get("risks", [])
        refactors = plan_preview.get("refactors", [])

        # Display steps (disable markup to avoid conflicts with LLM-generated content)
        if steps:
            console.print()
            # Use markup=False to prevent Rich from parsing brackets in plan steps
            step_text = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
            console.print(Panel.fit(
                step_text,
                title="Implementation Plan",
                border_style="blue"
            ), markup=False)
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
            console.print(f"[dim]Note: {len(specs)} boundary specification(s) identified[/]")
            console.print(f"[dim]  → {spec_names}[/]", markup=False)

        console.print()
        console.print("[bold]Plan Approval[/]")
        console.print()
        console.print("  [cyan]1.[/] Approve plan and proceed to patches")
        console.print("  [cyan]2.[/] Reject and regenerate plan")
        console.print("  [cyan]3.[/] View boundary specs (details)")
        console.print("  [cyan]4.[/] Return to main menu")
        console.print()

        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            # Approve the plan
            try:
                self.orchestrator.approve_plan(self.current_task.id)
                console.print("[green]✓[/] Plan approved!")

                # Generate patches and test suggestions after plan approval
                console.print()
                console.print("[cyan]Generating patches and test suggestions...[/]")
                patch_result = self.orchestrator.generate_patches(self.current_task.id)
                patch_count = patch_result.get("patch_count", 0)
                test_count = patch_result.get("test_count", 0)
                console.print(f"[green]✓[/] Generated {patch_count} patch(es) and {test_count} test suggestion(s)")

                # Reload task to get updated metadata with patches
                self.current_task = self.orchestrator._get_task(self.current_task.id)

                self.state = ConversationState.REVIEWING_PATCHES
            except Exception as exc:
                console.print(f"[red]Error approving plan: {exc}[/]")
        elif choice == "2":
            # Reject and regenerate
            console.print("[yellow]Regenerating plan...[/]")
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
            console.print(f"  Interfaces:")
            for interface in machine_spec.get('interfaces', []):
                console.print(f"    • {interface}")
            console.print(f"  Invariants:")
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
            console.print("[cyan]No boundary specifications needed. Proceeding to patches...[/]")
            self.state = ConversationState.REVIEWING_PATCHES
            return

        pending_specs = [s for s in specs if s.get("status") == "PENDING"]

        if not pending_specs:
            console.print()
            console.print("[green]✓[/] All boundary specs resolved!")
            self.state = ConversationState.REVIEWING_PATCHES
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
            console.print(f"[bold]Interfaces:[/]")
            for interface in machine_spec.get('interfaces', []):
                console.print(f"  • {interface}")
            console.print(f"[bold]Invariants:[/]")
            for invariant in machine_spec.get('invariants', []):
                console.print(f"  • {invariant}")
            console.print()

            console.print("[bold]Actions:[/]")
            console.print("  [cyan]1.[/] Approve this spec")
            console.print("  [cyan]2.[/] Skip this spec")
            console.print("  [cyan]3.[/] Approve all remaining specs")
            console.print("  [cyan]4.[/] Skip all remaining specs")
            console.print()

            choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")

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
        self.state = ConversationState.REVIEWING_PATCHES

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

            console.print(f"  [cyan]{i}.[/] [{status_color}]{task.status.value}[/] - {task.description[:50]}")

        console.print()
        choice = Prompt.ask("Choice", default="1")

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(tasks):
                self.current_task = tasks[idx]
                self.indexed_repo = self.current_task.repo_path
                self.indexed_branch = self.current_task.branch

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
                task.description[:60],
                task.updated_at.strftime("%Y-%m-%d %H:%M")
            )

        console.print(table)
        console.print()
