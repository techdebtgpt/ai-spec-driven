from __future__ import annotations

import logging
import sys
from dataclasses import asdict
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from ..config.settings import AgentSettings, get_settings
from ..domain.models import (
    BoundarySpec,
    Patch,
    PatchKind,
    PatchStatus,
    Plan,
    PlanStep,
    RefactorSuggestion,
    RefactorSuggestionStatus,
    Task,
    TaskStatus,
    TestSuggestion,
)
from ..persistence.store import JsonStore
from ..services.context.indexer import ContextIndexer
from ..services.context.retriever import ContextRetriever
from ..services.integrations.serena_client import SerenaToolClient
from ..services.llm.openai_client import OpenAILLMClient
from ..services.planning.clarifier import Clarifier
from ..services.planning.plan_builder import PlanBuilder
from ..services.patches.engine import PatchEngine
from ..services.specs.boundary_manager import BoundaryManager
from ..services.planning.refactor_advisor import RefactorAdvisor
from ..services.tests.suggester import TestSuggester
from ..tracing.reasoning_log import ReasoningLog


LOG = logging.getLogger(__name__)


class TaskOrchestrator:
    """
    Coordinates the full lifecycle described in the MVP spec.
    """

    def __init__(self, settings: Optional[AgentSettings] = None) -> None:
        self.settings = settings or get_settings()
        self.store = JsonStore(self.settings.state_dir)
        self.logger = ReasoningLog(self.store)

        self.context_indexer = ContextIndexer(self.settings)
        self.context_retriever = ContextRetriever()
        self.clarifier = Clarifier()
        self.llm_client = self._maybe_create_llm_client()
        self.plan_builder = PlanBuilder(llm_client=self.llm_client)
        # Note: BoundaryManager created per-use in generate_plan with context
        self.boundary_manager = BoundaryManager()
        self.refactor_advisor = RefactorAdvisor()
        self.serena_client = self._maybe_create_serena_client()
        self.patch_engine = PatchEngine(serena_client=self.serena_client)
        self.test_suggester = TestSuggester()

    # ------------------------------------------------------------------ Tasks
    def create_task(self, repo_path: Path, branch: str, description: str) -> Task:
        task = Task(
            id=str(uuid4()),
            repo_path=repo_path.resolve(),
            branch=branch,
            description=description,
            status=TaskStatus.CLARIFYING,
        )

        summary = self.context_indexer.summarize_repository(task.repo_path)
        clarifications = self.clarifier.generate_questions(task.id, description)

        task.metadata["repository_summary"] = summary
        task.metadata["clarifications"] = [asdict(item) for item in clarifications]
        task.metadata["starting_commit"] = self._current_commit(task.repo_path)
        self._snapshot_worktree_status(task)

        self.store.upsert_task(task)
        self.logger.record(task.id, "TASK_CREATED", {"summary": summary, "clarifications": task.metadata["clarifications"]})

        return task

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        tasks = self.store.load_tasks()
        if status:
            tasks = [task for task in tasks if task.status == status]
        tasks.sort(key=lambda t: t.created_at)
        return tasks

    # ------------------------------------------------------------------ Planning
    def generate_plan(self, task_id: str) -> Dict[str, List[str]]:
        task = self._get_task(task_id)
        context_summary = task.metadata.get("repository_summary", {})
        plan = self.plan_builder.build_plan(task.id, task.description, context_summary)

        # Create BoundaryManager with LLM client and context for this plan
        boundary_manager = BoundaryManager(
            llm_client=self.llm_client,
            context_summary=context_summary
        )
        specs = boundary_manager.required_specs(plan)

        patches = self.patch_engine.draft_patches(
            plan, 
            repo_path=task.repo_path,
            boundary_specs=specs
        )
        tests = self.test_suggester.suggest(plan)
        refactors = self.refactor_advisor.suggest(plan)

        task.metadata["plan_preview"] = {
            "steps": [step.description for step in plan.steps],
            "risks": plan.risks,
            "refactors": plan.refactor_suggestions,
        }
        # Store full boundary spec objects for approval workflow
        task.metadata["boundary_specs"] = [
            {
                "id": spec.id,
                "boundary_name": spec.boundary_name,
                "human_description": spec.human_description,
                "diagram_text": spec.diagram_text,
                "machine_spec": spec.machine_spec,
                "status": spec.status.value,
            }
            for spec in specs
        ]
        # Also store just names for backward compatibility with CLI display
        task.metadata["pending_specs"] = [spec.boundary_name for spec in specs]
        task.metadata["patch_queue"] = [patch.step_reference for patch in patches]
        task.metadata["patch_queue_state"] = [patch.to_dict() for patch in patches]
        task.metadata["refactor_suggestions"] = [item.to_dict() for item in refactors]
        task.metadata["test_suggestions"] = [suggestion.description for suggestion in tests]
        task.status = TaskStatus.SPEC_PENDING if specs else TaskStatus.PLANNING
        self._snapshot_worktree_status(task)
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "PLAN_GENERATED",
            {
                "plan": task.metadata["plan_preview"],
                "pending_specs": task.metadata["pending_specs"],
                "patch_queue": task.metadata["patch_queue"],
                "refactor_suggestions": task.metadata["refactor_suggestions"],
                "test_suggestions": task.metadata["test_suggestions"],
            },
        )

        return {
            "plan": task.metadata["plan_preview"],
            "pending_specs": task.metadata["pending_specs"],
            "patch_queue": task.metadata["patch_queue"],
            "refactor_suggestions": task.metadata["refactor_suggestions"],
            "test_suggestions": task.metadata["test_suggestions"],
        }

    # ------------------------------------------------------------------ Boundary Specs
    def get_boundary_specs(self, task_id: str) -> List[Dict]:
        """
        Get all boundary specs for a task.
        """
        task = self._get_task(task_id)
        return task.metadata.get("boundary_specs", [])

    def approve_spec(self, task_id: str, spec_id: str) -> Dict:
        """
        Approve a boundary specification.
        """
        task = self._get_task(task_id)
        specs = task.metadata.get("boundary_specs", [])

        spec_found = False
        for spec in specs:
            if spec["id"] == spec_id:
                spec["status"] = "APPROVED"
                spec_found = True
                break

        if not spec_found:
            raise ValueError(f"Boundary spec not found: {spec_id}")

        task.metadata["boundary_specs"] = specs
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "SPEC_APPROVED",
            {"spec_id": spec_id, "boundary_name": spec.get("boundary_name")}
        )

        return {"spec_id": spec_id, "status": "APPROVED"}

    def skip_spec(self, task_id: str, spec_id: str) -> Dict:
        """
        Skip (override) a boundary specification.
        """
        task = self._get_task(task_id)
        specs = task.metadata.get("boundary_specs", [])

        spec_found = False
        for spec in specs:
            if spec["id"] == spec_id:
                spec["status"] = "SKIPPED"
                spec_found = True
                break

        if not spec_found:
            raise ValueError(f"Boundary spec not found: {spec_id}")

        task.metadata["boundary_specs"] = specs
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "SPEC_SKIPPED",
            {"spec_id": spec_id, "boundary_name": spec.get("boundary_name")}
        )

        return {"spec_id": spec_id, "status": "SKIPPED"}

    # ------------------------------------------------------------------ Helpers
    def _get_task(self, task_id: str) -> Task:
        for task in self.store.load_tasks():
            if task.id == task_id:
                return task
        raise ValueError(f"Task not found: {task_id}")

    def _maybe_create_llm_client(self) -> Optional[OpenAILLMClient]:
        """
        Initialize OpenAI LLM client if API key is configured.
        """
        api_key = self.settings.openai_api_key
        if not api_key:
            LOG.debug("OpenAI API key not configured, LLM features will be unavailable")
            return None

        try:
            return OpenAILLMClient(
                api_key=api_key,
                model=self.settings.openai_model,
                base_url=self.settings.openai_base_url,
                timeout_seconds=self.settings.openai_timeout_seconds,
            )
        except ValueError as exc:
            LOG.warning("Failed to initialize OpenAI client: %s", exc)
            return None

    def _maybe_create_serena_client(self) -> Optional[SerenaToolClient]:
        repo_root = Path(__file__).resolve().parents[3]
        default_wrapper = repo_root / "scripts" / "serena_patch_wrapper.py"

        enabled = self.settings.serena_enabled
        command = self.settings.serena_command

        if not command and default_wrapper.exists():
            command = (sys.executable, str(default_wrapper))
            enabled = True

        if not (enabled and command):
            return None

        try:
            return SerenaToolClient(command, self.settings.serena_timeout_seconds)
        except ValueError as exc:
            LOG.warning("Failed to initialize Serena integration: %s", exc)
            return None

    # ------------------------------------------------------------------ Patch queue helpers
    def list_patches(self, task_id: str) -> List[Patch]:
        task = self._get_task(task_id)
        if self._detect_manual_edits(task):
            self.generate_plan(task_id)
            task = self._get_task(task_id)
        return self._load_patch_queue(task)

    def get_next_pending_patch(self, task_id: str) -> Patch | None:
        task = self._get_task(task_id)
        if self._detect_manual_edits(task):
            self.generate_plan(task_id)
            task = self._get_task(task_id)
        patches = self._load_patch_queue(task)
        for patch in patches:
            if patch.status == PatchStatus.PENDING:
                return patch
        return None

    def approve_patch(self, task_id: str, patch_id: str) -> Patch:
        task = self._get_task(task_id)
        patches = self._load_patch_queue(task)
        patch = self._require_patch(patches, patch_id)
        if patch.status != PatchStatus.PENDING:
            raise ValueError("Patch has already been processed.")

        self._ensure_branch(task.repo_path, task.branch)
        self._apply_patch_diff(task.repo_path, patch.diff)
        patch.status = PatchStatus.APPLIED
        self._persist_patch_queue(task, patches)
        self._snapshot_worktree_status(task)
        self.logger.record(task.id, "PATCH_APPROVED", {"patch_id": patch.id, "step": patch.step_reference})
        return patch

    def reject_patch(self, task_id: str, patch_id: str) -> None:
        task = self._get_task(task_id)
        patches = self._load_patch_queue(task)
        patch = self._require_patch(patches, patch_id)
        patch.status = PatchStatus.REJECTED
        self._persist_patch_queue(task, patches)
        self.logger.record(task.id, "PATCH_REJECTED", {"patch_id": patch.id, "step": patch.step_reference})
        # Trigger plan regeneration so the queue refreshes.
        self.generate_plan(task_id)

    def _load_patch_queue(self, task: Task) -> List[Patch]:
        raw = task.metadata.get("patch_queue_state", [])
        return [Patch.from_dict(item) for item in raw]

    def _persist_patch_queue(self, task: Task, patches: List[Patch]) -> None:
        task.metadata["patch_queue_state"] = [patch.to_dict() for patch in patches]
        task.touch()
        self.store.upsert_task(task)

    @staticmethod
    def _require_patch(patches: List[Patch], patch_id: str) -> Patch:
        for patch in patches:
            if patch.id == patch_id:
                return patch
        raise ValueError(f"Patch not found: {patch_id}")

    def _ensure_branch(self, repo_path: Path, expected_branch: str) -> None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Unable to determine current branch: {exc.stderr}") from exc
        branch = result.stdout.strip()
        if branch != expected_branch:
            LOG.warning(
                f"Task was created on branch '{expected_branch}', but working tree is on '{branch}'. "
                f"Patch may not apply cleanly if file contents have diverged."
            )
            # Don't raise - allow applying patches on different branches with a warning

    @staticmethod
    def _apply_patch_diff(repo_path: Path, diff: str) -> None:
        # Ensure diff ends with newline (required by git apply)
        if diff and not diff.endswith('\n'):
            diff = diff + '\n'
        
        # Extract file paths from diff to provide better diagnostics
        affected_files = TaskOrchestrator._extract_files_from_diff(diff)
        
        # Capture git status before applying to detect partial applications
        git_status_before = TaskOrchestrator._get_git_status_snapshot(repo_path, affected_files)
        
        # Try applying with --3way first (allows 3-way merge for better conflict resolution)
        try:
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "--3way"],
                input=diff,
                text=True,
                check=True,
                capture_output=True,
                cwd=repo_path,
            )
            # Verify the patch was actually applied by checking if files changed
            if TaskOrchestrator._files_changed(repo_path, affected_files, git_status_before):
                return
            # If no changes detected, continue to next attempt
        except subprocess.CalledProcessError:
            # If --3way fails, try without it (some patches don't support 3-way)
            pass
        
        # Try with more lenient whitespace handling
        try:
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "--ignore-space-change"],
                input=diff,
                text=True,
                check=True,
                capture_output=True,
                cwd=repo_path,
            )
            # Verify the patch was actually applied
            if TaskOrchestrator._files_changed(repo_path, affected_files, git_status_before):
                return
        except subprocess.CalledProcessError:
            pass
        
        # Fallback to standard apply
        try:
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn"],
                input=diff,
                text=True,
                check=True,
                capture_output=True,
                cwd=repo_path,
            )
            # Verify the patch was actually applied
            if TaskOrchestrator._files_changed(repo_path, affected_files, git_status_before):
                return
        except subprocess.CalledProcessError as exc:
            # Log the diff for debugging if it fails
            LOG.error("Failed to apply patch. Diff (first 200 chars): %s", diff[:200])
            LOG.error("Git apply stderr: %s", exc.stderr)
            
            # Extract error message and failing line number
            error_msg = exc.stderr.strip()
            failing_line = TaskOrchestrator._extract_failing_line_number(error_msg)
            
            # Try to read actual file content for comparison
            file_diagnostics = []
            
            for file_path in affected_files:
                full_path = repo_path / file_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r') as f:
                            lines = f.readlines()
                            info = f"  {file_path}: {len(lines)} lines"
                            
                            # Show context around the failing line if we can identify it
                            if failing_line and 1 <= failing_line <= len(lines):
                                start = max(0, failing_line - 3)
                                end = min(len(lines), failing_line + 2)
                                context_lines = []
                                for i in range(start, end):
                                    marker = ">>>" if i == failing_line - 1 else "   "
                                    context_lines.append(f"    {marker} {i+1:3d}: {repr(lines[i].rstrip())}")
                                info += f"\n    Context around line {failing_line}:\n" + "\n".join(context_lines)
                            elif lines:
                                info += f", first line: {repr(lines[0].rstrip())}"
                            
                            file_diagnostics.append(info)
                    except Exception as e:
                        file_diagnostics.append(f"  {file_path}: Could not read ({e})")
                else:
                    file_diagnostics.append(f"  {file_path}: File does not exist")
            
            # Check if patch was partially applied by git apply (e.g., with --3way)
            files_changed = TaskOrchestrator._files_changed(repo_path, affected_files, git_status_before)
            if files_changed:
                # Patch was partially applied - don't try manual fallback to avoid duplicates
                LOG.warning("Patch was partially applied by git apply. Manual fallback skipped to prevent duplicates.")
                return
            
            # Try manual fallback for simple patches (additions only)
            if "patch does not apply" in error_msg:
                try:
                    TaskOrchestrator._try_manual_patch_apply(repo_path, diff, affected_files)
                    LOG.info("Successfully applied patch using manual fallback")
                    return
                except Exception as manual_exc:
                    LOG.debug("Manual patch application also failed: %s", manual_exc)
                    # Continue to show error message
            
            # Provide more helpful error message with manual application instructions
            diagnostic_info = "\n".join(file_diagnostics) if file_diagnostics else ""
            if "patch does not apply" in error_msg:
                # Extract what needs to be added manually
                manual_instructions = TaskOrchestrator._generate_manual_instructions(diff, affected_files, repo_path)
                
                error_message = (
                    f"Patch does not apply to current file state.\n\n"
                    f"Current file state:\n{diagnostic_info}\n\n"
                    f"**Why this happens:**\n"
                    f"  'git apply' uses 'context lines' (unchanged lines around the change) to find\n"
                    f"  where to apply the patch. If the surrounding lines don't match exactly,\n"
                    f"  it fails even if the target location is correct. This is called 'insufficient context'.\n\n"
                    f"This may happen if:\n"
                    f"  - Files have been modified since the patch was created\n"
                    f"  - You're on a different branch than when the patch was generated\n"
                    f"  - The patch was created against a different version of the file\n"
                    f"  - The surrounding context lines have changed\n\n"
                )
                
                if manual_instructions:
                    error_message += (
                        f"**Manual application suggested:**\n"
                        f"{manual_instructions}\n\n"
                    )
                
                error_message += f"Git error: {error_msg}"
                
                raise RuntimeError(error_message) from exc
            raise RuntimeError(f"Failed to apply patch: {error_msg}") from exc
    
    @staticmethod
    def _extract_files_from_diff(diff: str) -> List[str]:
        """Extract file paths from a unified diff."""
        files = set()
        for line in diff.split('\n'):
            if line.startswith('--- a/') or line.startswith('+++ b/'):
                # Remove the --- a/ or +++ b/ prefix
                parts = line.split()
                if len(parts) > 1:
                    file_path = parts[1]
                else:
                    file_path = line[6:]  # Remove "--- a/" or "+++ b/"
                
                # Clean up the path (remove a/ or b/ prefix if present)
                if file_path.startswith('a/') or file_path.startswith('b/'):
                    file_path = file_path[2:]
                
                if file_path and file_path != '/dev/null' and not file_path.startswith('a/') and not file_path.startswith('b/'):
                    files.add(file_path)
        return list(files)
    
    @staticmethod
    def _extract_failing_line_number(error_msg: str) -> Optional[int]:
        """Extract line number from git apply error message like 'patch failed: file.tf:7'."""
        import re
        # Match patterns like "file.tf:7" or "error: patch failed: file.tf:7"
        match = re.search(r':(\d+)(?:\s|$)', error_msg)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
    @staticmethod
    def _generate_manual_instructions(diff: str, affected_files: List[str], repo_path: Path) -> str:
        """
        Generate human-readable instructions for manually applying a patch.
        Returns empty string if manual application isn't straightforward.
        """
        import re
        
        instructions = []
        hunk_pattern = re.compile(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@')
        
        for file_path in affected_files:
            full_path = repo_path / file_path
            if not full_path.exists():
                continue
            
            # Parse diff for this file
            in_file = False
            hunks = []
            current_hunk = None
            additions = []
            
            for line in diff.split('\n'):
                if line.startswith(f'--- a/{file_path}') or line.startswith(f'+++ b/{file_path}'):
                    in_file = True
                    continue
                elif line.startswith('---') or line.startswith('+++'):
                    if in_file:
                        break
                    continue
                
                if not in_file:
                    continue
                
                hunk_match = hunk_pattern.match(line)
                if hunk_match:
                    if current_hunk is not None:
                        hunks.append((*current_hunk, additions))
                    start_line = int(hunk_match.group(1))
                    old_count = int(hunk_match.group(2))
                    new_count = int(hunk_match.group(4))
                    current_hunk = (start_line, old_count)
                    additions = []
                    continue
                
                if line.startswith('+') and not line.startswith('+++'):
                    # Extract the content to add (without the + prefix)
                    content = line[1:]
                    additions.append(content)
                elif line.startswith('-') and not line.startswith('---'):
                    # Has deletions - manual application is more complex
                    return ""
            
            # Save last hunk
            if current_hunk is not None:
                hunks.append((*current_hunk, additions))
            
            # Generate instructions for each hunk
            if hunks:
                instructions.append(f"File: {file_path}")
                for start_line, old_count, additions in hunks:
                    if additions:
                        # Filter out empty additions
                        non_empty = [a for a in additions if a.strip()]
                        if non_empty:
                            insert_after = start_line + old_count - 1
                            instructions.append(f"  After line {insert_after}, add:")
                            for addition in non_empty:
                                # Show the addition with proper indentation
                                addition_clean = addition.rstrip()
                                if addition_clean:
                                    instructions.append(f"    {addition_clean}")
                            instructions.append("")
        
        if instructions:
            return "\n".join(instructions)
        return ""
    
    @staticmethod
    def _try_manual_patch_apply(repo_path: Path, diff: str, affected_files: List[str]) -> None:
        """
        Attempt to manually apply a simple patch (additions only) when git apply fails.
        This is a fallback for cases where the patch format is correct but git apply
        is too strict about context matching.
        """
        import re
        
        # Parse the unified diff format
        # Format: @@ -start,count +start,count @@
        hunk_pattern = re.compile(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@')
        
        for file_path in affected_files:
            full_path = repo_path / file_path
            if not full_path.exists():
                raise ValueError(f"File {file_path} does not exist for manual patch application")
            
            # Read current file content
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            # Parse diff to find hunks for this file
            in_file_hunk = False
            hunks = []  # List of (start_line, old_count, new_count, additions)
            current_hunk = None
            current_additions = []
            
            for line in diff.split('\n'):
                if line.startswith(f'--- a/{file_path}') or line.startswith(f'+++ b/{file_path}'):
                    in_file_hunk = True
                    continue
                elif line.startswith('---') or line.startswith('+++'):
                    if in_file_hunk:
                        # Finished processing this file's hunks
                        break
                    continue
                
                if not in_file_hunk:
                    continue
                
                # Check for hunk header
                hunk_match = hunk_pattern.match(line)
                if hunk_match:
                    # Save previous hunk if any
                    if current_hunk is not None:
                        hunks.append((*current_hunk, current_additions))
                    
                    # Start new hunk
                    start_line = int(hunk_match.group(1)) - 1  # Convert to 0-indexed
                    old_count = int(hunk_match.group(2))
                    new_count = int(hunk_match.group(4))
                    current_hunk = (start_line, old_count, new_count)
                    current_additions = []
                    continue
                
                # Process hunk lines
                if line.startswith(' '):
                    # Context line - skip (we'll verify position when applying)
                    continue
                elif line.startswith('+'):
                    # Addition - preserve the line content
                    content = line[1:]
                    # Ensure it ends with newline if it's not empty
                    if content and not content.endswith('\n'):
                        content += '\n'
                    elif not content:
                        content = '\n'
                    current_additions.append(content)
                elif line.startswith('-'):
                    # Deletion - manual apply doesn't handle deletions well
                    raise ValueError("Manual patch application doesn't support deletions")
            
            # Save last hunk
            if current_hunk is not None:
                hunks.append((*current_hunk, current_additions))
            
            # Check if content already exists to avoid duplicates
            file_content = ''.join(lines)
            all_additions_exist = True
            for start_line, old_count, new_count, additions in hunks:
                # Check if any of the additions are missing
                for addition in additions:
                    if addition.strip():  # Only check non-empty lines
                        # Check if this addition (or a very similar one) already exists
                        addition_stripped = addition.strip()
                        if addition_stripped not in file_content:
                            all_additions_exist = False
                            break
                if not all_additions_exist:
                    break
            
            if all_additions_exist and hunks:
                LOG.info(f"Patch content already exists in {file_path}, skipping manual application to avoid duplicates")
                return
            
            # Apply all hunks (in reverse order to preserve line numbers)
            for start_line, old_count, new_count, additions in reversed(hunks):
                TaskOrchestrator._apply_hunk_manually(lines, start_line, old_count, additions)
            
            # Write modified content back
            with open(full_path, 'w') as f:
                f.writelines(lines)
    
    @staticmethod
    def _apply_hunk_manually(
        lines: List[str], 
        start_line: int, 
        old_count: int, 
        additions: List[str]
    ) -> None:
        """
        Manually apply a hunk by inserting additions after the specified start line.
        
        Args:
            lines: List of file lines (will be modified in place)
            start_line: 0-indexed line number where the hunk starts
            old_count: Number of lines in the old file (context + deletions)
            additions: List of lines to add (without + prefix)
        """
        # For additions-only patches, insert right after the context lines
        # start_line is 0-indexed, so if patch says line 7, start_line is 6
        # old_count tells us how many lines to skip (the context)
        # We insert after those context lines
        insert_pos = start_line + old_count
        
        # Verify we're not out of bounds
        if insert_pos > len(lines):
            raise ValueError(f"Cannot insert at line {insert_pos + 1}, file only has {len(lines)} lines")
        
        # Check if additions are already present to avoid duplicates
        # Look ahead to see if the additions already exist
        if insert_pos < len(lines):
            # Check if the next few lines match what we're trying to add
            existing_lines = lines[insert_pos:insert_pos + len(additions)]
            if existing_lines == additions:
                # Content already exists, skip insertion
                LOG.debug(f"Content already present at line {insert_pos + 1}, skipping duplicate insertion")
                return
        
        # Insert the new lines (in order, so we insert from end to start when reversed)
        for addition in additions:
            lines.insert(insert_pos, addition)
    
    @staticmethod
    def _get_git_status_snapshot(repo_path: Path, file_paths: List[str]) -> Dict[str, str]:
        """Get a snapshot of file contents before applying patch."""
        snapshot = {}
        for file_path in file_paths:
            full_path = repo_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        snapshot[file_path] = f.read()
                except Exception:
                    snapshot[file_path] = ""
            else:
                snapshot[file_path] = None  # File doesn't exist
        return snapshot
    
    @staticmethod
    def _files_changed(repo_path: Path, file_paths: List[str], before_snapshot: Dict[str, str]) -> bool:
        """Check if any files changed compared to the snapshot."""
        for file_path in file_paths:
            full_path = repo_path / file_path
            before_content = before_snapshot.get(file_path)
            
            if before_content is None:
                # File didn't exist before
                if full_path.exists():
                    return True  # File was created
            else:
                # File existed before
                if not full_path.exists():
                    return True  # File was deleted
                try:
                    with open(full_path, 'r') as f:
                        current_content = f.read()
                    if current_content != before_content:
                        return True  # File content changed
                except Exception:
                    # Can't read file, assume it changed
                    return True
        return False

    # ------------------------------------------------------------------ Refactor helpers
    def list_refactors(self, task_id: str) -> List[RefactorSuggestion]:
        task = self._get_task(task_id)
        return self._load_refactors(task)

    def update_refactor_status(self, task_id: str, suggestion_id: str, status: RefactorSuggestionStatus) -> RefactorSuggestion:
        task = self._get_task(task_id)
        suggestions = self._load_refactors(task)
        suggestion = self._require_refactor(suggestions, suggestion_id)
        suggestion.status = status
        self._persist_refactors(task, suggestions)
        self.logger.record(task.id, "REFACTOR_UPDATED", {"suggestion_id": suggestion.id, "status": status.value})
        return suggestion

    def approve_refactor(self, task_id: str, suggestion_id: str) -> None:
        task = self._get_task(task_id)
        suggestions = self._load_refactors(task)
        suggestion = self._require_refactor(suggestions, suggestion_id)
        if suggestion.status != RefactorSuggestionStatus.PENDING:
            raise ValueError("Refactor suggestion already processed.")

        suggestion.status = RefactorSuggestionStatus.APPROVED
        self._persist_refactors(task, suggestions)
        self._enqueue_refactor_patch(task, suggestion)
        self.logger.record(task.id, "REFACTOR_APPROVED", {"suggestion_id": suggestion.id})

    def reject_refactor(self, task_id: str, suggestion_id: str) -> None:
        task = self._get_task(task_id)
        suggestions = self._load_refactors(task)
        suggestion = self._require_refactor(suggestions, suggestion_id)
        suggestion.status = RefactorSuggestionStatus.REJECTED
        self._persist_refactors(task, suggestions)
        self.logger.record(task.id, "REFACTOR_REJECTED", {"suggestion_id": suggestion.id})

    def _enqueue_refactor_patch(self, task: Task, suggestion: RefactorSuggestion) -> None:
        plan = Plan(
            id=str(uuid4()),
            task_id=task.id,
            steps=[PlanStep(description=suggestion.description, notes=suggestion.rationale, target_files=suggestion.scope)],
        )
        patches = self.patch_engine.draft_patches(plan, repo_path=task.repo_path, kind=PatchKind.REFACTOR)
        queue = self._load_patch_queue(task)
        queue.extend(patches)
        self._persist_patch_queue(task, queue)

    def _load_refactors(self, task: Task) -> List[RefactorSuggestion]:
        raw = task.metadata.get("refactor_suggestions", [])
        return [RefactorSuggestion.from_dict(item) for item in raw]

    def _persist_refactors(self, task: Task, suggestions: List[RefactorSuggestion]) -> None:
        task.metadata["refactor_suggestions"] = [item.to_dict() for item in suggestions]
        task.touch()
        self.store.upsert_task(task)

    @staticmethod
    def _require_refactor(suggestions: List[RefactorSuggestion], suggestion_id: str) -> RefactorSuggestion:
        for suggestion in suggestions:
            if suggestion.id == suggestion_id:
                return suggestion
        raise ValueError(f"Refactor suggestion not found: {suggestion_id}")

    # ------------------------------------------------------------------ Manual override helpers
    def _snapshot_worktree_status(self, task: Task) -> None:
        status = self._get_git_status(task.repo_path)
        task.metadata["worktree_status"] = status
        task.metadata["last_snapshot_commit"] = self._current_commit(task.repo_path)
        self.store.upsert_task(task)

    def _detect_manual_edits(self, task: Task) -> bool:
        current = self._get_git_status(task.repo_path)
        previous = task.metadata.get("worktree_status", "")
        if current != previous:
            self.logger.record(
                task.id,
                "MANUAL_EDIT_DETECTED",
                {"previous_status": previous, "current_status": current},
            )
            return True
        return False

    def _get_git_status(self, repo_path: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to read git status: {exc.stderr}") from exc
        return result.stdout.strip()

    def _current_branch(self, repo_path: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to determine current branch: {exc.stderr}") from exc
        return result.stdout.strip()

    def _current_commit(self, repo_path: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to determine current commit: {exc.stderr}") from exc
        return result.stdout.strip()

    def get_task_status(self, task_id: str) -> Dict[str, str]:
        task = self._get_task(task_id)
        branch = self._current_branch(task.repo_path)
        commit = self._current_commit(task.repo_path)
        git_status = self._get_git_status(task.repo_path)
        patches = self._load_patch_queue(task)
        return {
            "task_id": task.id,
            "repo": str(task.repo_path),
            "branch": branch,
            "expected_branch": task.branch,
            "last_commit": commit,
            "patch_counts": {
                status.value: sum(1 for patch in patches if patch.status == status) for status in PatchStatus
            },
            "git_status": git_status or "clean",
        }


