from __future__ import annotations

import logging
import sys
from dataclasses import asdict
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime
import json

from ..config.settings import AgentSettings, get_settings
from ..domain.models import (
    BoundarySpec,
    BoundarySpecStatus,
    ClarificationStatus,
    Patch,
    PatchKind,
    PatchStatus,
    Plan,
    PlanStep,
    RefactorSuggestion,
    RefactorSuggestionStatus,
    Task,
    TaskStatus,
)
from ..persistence.store import JsonStore
from ..services.context.indexer import ContextIndexer
from ..services.context.semantic_indexer import SemanticIndexer
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
        self.llm_client = self._maybe_create_llm_client()
        self.clarifier = Clarifier(llm_client=self.llm_client)
        self.semantic_indexer = SemanticIndexer(llm_client=self.llm_client, settings=self.settings)
        self.plan_builder = PlanBuilder(llm_client=self.llm_client)
        # Note: BoundaryManager created per-use in generate_plan with context
        self.boundary_manager = BoundaryManager()
        self.refactor_advisor = RefactorAdvisor()
        self.serena_client = self._maybe_create_serena_client()
        self.patch_engine = PatchEngine(
            serena_client=self.serena_client,
            llm_client=self.llm_client,
        )
        self.test_suggester = TestSuggester(llm_client=self.llm_client)

    # ------------------------------------------------------------------ Tasks
    def resolve_repo_root(self, repo_path: Path) -> Path:
        return self._resolve_repo_root(repo_path)

    def _resolve_repo_root(self, repo_path: Path) -> Path:
        """
        Best-effort: if user points at a subdirectory (e.g. ".../src"), resolve to the
        repository root so we don't miss tests/config in sibling folders.
        """
        try:
            candidate = repo_path.resolve()
        except Exception:
            candidate = repo_path

        if candidate.exists() and candidate.is_file():
            candidate = candidate.parent

        # Walk upward to the nearest directory that looks like a VCS root.
        # Note: we don't require git; many repos (or test fixtures) may not have an actual
        # `.git/` directory. In those cases, fall back to common "repo root" marker files.
        root_markers = (
            ".gitignore",
            "pyproject.toml",
            "package.json",
            "go.mod",
            "Cargo.toml",
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "requirements.txt",
            "setup.py",
        )
        for parent in [candidate, *candidate.parents]:
            git_marker = parent / ".git"
            if git_marker.exists():
                return parent
            for marker in root_markers:
                if (parent / marker).exists():
                    return parent
            # Globs for ecosystems where a single root marker is often sufficient.
            if next(parent.glob("*.sln"), None) is not None:
                return parent

        return candidate

    def index_repository(
        self,
        repo_path: Path,
        branch: str,
        include_serena_semantic_tree: bool = False,
    ) -> Dict:
        """
        Index a repository and save the context for later use.
        
        This generates both:
        1. Basic summary (file counts, languages, etc.)
        2. Semantic index (architecture, boundaries, domains, etc.)
        """
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        requested_path = repo_path.resolve()
        resolved_repo_path = self._resolve_repo_root(requested_path)

        sys.stderr.write("Analyzing repository structure...\n")
        summary = self.context_indexer.summarize_repository(
            resolved_repo_path,
            include_serena_semantic_tree=include_serena_semantic_tree,
        )
        
        # Get comprehensive git information
        git_info = {}
        try:
            git_info["current_commit"] = self._current_commit(resolved_repo_path)
            git_info["current_branch"] = self._get_current_branch(resolved_repo_path)
            git_info["remote_url"] = self._get_remote_url(resolved_repo_path)
            git_info["commit_author"] = self._get_commit_author(resolved_repo_path)
            git_info["commit_message"] = self._get_commit_message(resolved_repo_path)
            git_info["commit_date"] = self._get_commit_date(resolved_repo_path)
        except Exception as e:
            self.logger.record("WARNING", "GIT_INFO_FAILED", {"error": str(e)})
            git_info["error"] = str(e)

        # Generate semantic index using LLM
        sys.stderr.write("Generating semantic index (this may take a minute)...\n")
        semantic_index = None
        try:
            semantic_index = self.semantic_indexer.generate_semantic_index(
                repo_path=resolved_repo_path,
                basic_summary=summary
            )
            sys.stderr.write("✓ Semantic index generated successfully\n")
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to generate semantic index: {e}\n")
            self.logger.record("WARNING", "SEMANTIC_INDEX_FAILED", {"error": str(e)})

        index_data = {
            "repo_path": str(resolved_repo_path),
            "requested_path": str(requested_path),
            "repo_name": resolved_repo_path.name,
            "branch": branch,
            "repository_summary": summary,
            "semantic_index": semantic_index,
            "git_info": git_info,
            "indexed_at": datetime.now().isoformat(),
        }

        self.store.save_repository_index(index_data)
        # Also export the semantic tree as a standalone file for easy inspection.
        try:
            tree_payload = (summary or {}).get("serena_semantic_tree") if isinstance(summary, dict) else None
            if tree_payload is not None:
                self.store.serena_tree_file.write_text(json.dumps(tree_payload, indent=2, default=str))
        except Exception:  # pragma: no cover - best-effort export
            LOG.debug("Failed to write serena_semantic_tree.json", exc_info=True)
        self.logger.record(
            "SYSTEM",
            "REPOSITORY_INDEXED",
            {"repo_path": str(resolved_repo_path), "requested_path": str(requested_path), "branch": branch, "has_semantic_index": semantic_index is not None}
        )

        return index_data

    def enrich_repository_index_with_serena_semantic_tree(
        self,
        repo_path: Path,
        *,
        targets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add (or refresh) the Serena semantic tree inside the saved repository index.

        This is useful when you want to keep chat responsive: run a normal index first,
        then run this slower Serena export in the background and persist it.
        """
        index_data = self.store.load_repository_index()

        indexed_repo = Path(index_data.get("repo_path", "")).resolve() if index_data.get("repo_path") else None
        requested_repo = self._resolve_repo_root(repo_path.resolve())
        if indexed_repo and indexed_repo != requested_repo:
            raise ValueError(
                f"Repository index is for {indexed_repo}, but enrichment requested for {requested_repo}. "
                "Re-run index for the target repo first."
            )

        summary = index_data.get("repository_summary") or {}
        if not isinstance(summary, dict):
            summary = {}

        tree_payload = self.context_indexer._export_semantic_tree_with_serena(
            requested_repo,
            targets=targets,
        )

        summary["serena_semantic_tree"] = tree_payload
        index_data["repository_summary"] = summary
        index_data["indexed_at"] = datetime.now().isoformat()

        self.store.save_repository_index(index_data)
        # Standalone convenience file in ~/.spec_agent/
        try:
            self.store.serena_tree_file.write_text(json.dumps(tree_payload, indent=2, default=str))
        except Exception:  # pragma: no cover
            LOG.debug("Failed to write serena_semantic_tree.json", exc_info=True)
        self.logger.record(
            "SYSTEM",
            "REPOSITORY_INDEX_ENRICHED_SERENA_TREE",
            {
                "repo_path": str(requested_repo),
                "has_tree": bool(tree_payload),
                "targets": list(targets or []),
            },
        )

        return {
            "repo_path": str(requested_repo),
            "targets": list(targets or []),
            "serena_semantic_tree": tree_payload,
        }

    def _derive_title_summary(self, description: str) -> tuple[str, str]:
        """
        Derive a human-friendly title + short summary from a free-form description.

        - Title: first non-empty line (trimmed, max 80 chars)
        - Summary: first line (trimmed, max 140 chars)
        """
        raw = (description or "").strip()
        if not raw:
            return ("", "")
        first_line = raw.splitlines()[0].strip()
        title = first_line[:80].rstrip()
        summary = first_line[:140].rstrip()
        return (title, summary)

    def create_task_from_index(
        self,
        description: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        client: str | None = None,
    ) -> Task:
        """
        Create a task using a previously indexed repository.
        """
        index_data = self.store.load_repository_index()

        repo_path = Path(index_data["repo_path"])
        branch = index_data["branch"]
        
        derived_title, derived_summary = self._derive_title_summary(description)
        task = Task(
            id=str(uuid4()),
            repo_path=repo_path,
            branch=branch,
            title=((title or "").strip() or derived_title),
            summary=((summary or "").strip() or derived_summary),
            client=((client or "").strip()),
            description=description,
            status=TaskStatus.CLARIFYING,
        )

        # Use the pre-computed summary from the index
        task.metadata["repository_summary"] = index_data["repository_summary"]
        task.metadata["starting_commit"] = index_data.get("starting_commit")
        task.metadata["description_snapshot"] = task.description

        # Generate clarifications based on the task description
        clarifications = self.clarifier.generate_questions(
            task.id,
            description,
            context_summary=index_data["repository_summary"]
        )
        task.metadata["clarifications"] = [asdict(item) for item in clarifications]
        
        # Snapshot the current worktree status
        self._snapshot_worktree_status(task)

        self._seed_context_from_description(task)
        self.store.upsert_task(task)
        self.logger.record(
            task.id,
            "TASK_CREATED_FROM_INDEX",
            {
                "summary": task.metadata["repository_summary"],
                "clarifications": task.metadata["clarifications"]
            }
        )

        return task

    def create_task(
        self,
        repo_path: Path,
        branch: str,
        description: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        client: str | None = None,
    ) -> Task:
        resolved_repo_path = self._resolve_repo_root(repo_path.resolve())
        derived_title, derived_summary = self._derive_title_summary(description)
        task = Task(
            id=str(uuid4()),
            repo_path=resolved_repo_path,
            branch=branch,
            title=((title or "").strip() or derived_title),
            summary=((summary or "").strip() or derived_summary),
            client=((client or "").strip()),
            description=description,
            status=TaskStatus.CLARIFYING,
        )

        summary = self.context_indexer.summarize_repository(task.repo_path)
        clarifications = self.clarifier.generate_questions(
            task.id,
            description,
            context_summary=summary
        )

        task.metadata["repository_summary"] = summary
        task.metadata["clarifications"] = [asdict(item) for item in clarifications]
        task.metadata["starting_commit"] = self._current_commit(task.repo_path)
        task.metadata["description_snapshot"] = task.description
        self._snapshot_worktree_status(task)

        self._seed_context_from_description(task)
        self.store.upsert_task(task)
        self.logger.record(task.id, "TASK_CREATED", {"summary": summary, "clarifications": task.metadata["clarifications"]})

        return task

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        tasks = self.store.load_tasks()
        if status:
            tasks = [task for task in tasks if task.status == status]
        tasks.sort(key=lambda t: t.created_at)
        return tasks

    def _seed_context_from_description(self, task: Task) -> None:
        """
        Seed the context history with modules referenced in the task description.
        """
        summary = task.metadata.get("repository_summary", {})
        top_modules = summary.get("top_modules", [])
        if not top_modules:
            return

        description_lower = task.description.lower()
        candidates: List[Path] = []
        for entry in top_modules:
            module_name = entry.split(" ")[0].rstrip("/").strip()
            if not module_name:
                continue
            if module_name.lower() in description_lower:
                candidate = (task.repo_path / module_name).resolve()
                if candidate.exists():
                    candidates.append(candidate)

        if candidates:
            self._record_context_step(
                task,
                "task-description",
                included=candidates,
                note="Auto-detected modules mentioned in description",
            )

    def get_cached_repository_index(self) -> Dict[str, Any]:
        """
        Return the most recent repository index payload saved by the index command.
        """
        return self.store.load_repository_index()

    def get_repository_index_for_repo(self, repo_path: Path) -> Dict[str, Any]:
        """
        Return the cached repository index payload for a specific repo (if present).
        """
        return self.store.load_repository_index_for_repo(repo_path.resolve(), None)

    def get_repository_index_for_repo_and_branch(self, repo_path: Path, branch: str) -> Dict[str, Any]:
        """
        Return the cached repository index payload for a specific repo+branch (if present).
        """
        return self.store.load_repository_index_for_repo(repo_path.resolve(), branch)

    def list_clarifications(self, task_id: str) -> List[Dict[str, Any]]:
        task = self._get_task(task_id)
        return task.metadata.get("clarifications", [])

    def has_pending_clarifications(self, task_id: str) -> bool:
        task = self._get_task(task_id)
        clarifications = task.metadata.get("clarifications", [])
        for item in clarifications:
            status = item.get("status") or ClarificationStatus.PENDING.value
            if status == ClarificationStatus.PENDING.value:
                return True
        return False

    def update_clarification(
        self,
        task_id: str,
        clarification_id: str,
        answer: Optional[str],
        status: ClarificationStatus,
    ) -> Dict[str, Any]:
        task = self._get_task(task_id)
        clarifications = task.metadata.get("clarifications", [])
        found = None
        for item in clarifications:
            if item.get("id") == clarification_id:
                item["status"] = status.value
                if answer is not None:
                    item["answer"] = answer
                found = item
                break

        if not found:
            raise ValueError(f"Clarification {clarification_id} not found for task {task_id}")

        task.metadata["clarifications"] = clarifications
        task.touch()
        self.store.upsert_task(task)
        self.logger.record(
            task.id,
            "CLARIFICATION_UPDATED",
            {"clarification_id": clarification_id, "status": status.value, "answer": answer or ""},
        )
        return found

    def restart_clarifications(self, task_id: str, *, reason: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Restart clarifications for a task.

        Intended for use when an engineer rejects a plan due to unclear or missing requirements.
        We generate a fresh set of questions (LLM-powered when available; heuristic fallback otherwise),
        preserve the previous clarification set in history, reset plan artifacts, and move the task
        back to CLARIFYING.
        """
        task = self._get_task(task_id)

        # Preserve prior clarifications for traceability.
        history = task.metadata.get("clarifications_history") or []
        if not isinstance(history, list):
            history = []
        previous = task.metadata.get("clarifications", []) or []
        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "reason": (reason or "").strip(),
                "clarifications": previous,
            }
        )
        task.metadata["clarifications_history"] = history

        # Use the repo index summary when available to produce better questions.
        context_summary = (
            (task.metadata.get("repository_summary") or {})
            if isinstance(task.metadata.get("repository_summary"), dict)
            else None
        )

        augmented_description = task.description.strip()
        if reason and reason.strip():
            augmented_description = f"{augmented_description}\n\nPlan rejected feedback:\n- {reason.strip()}"

        new_items = self.clarifier.generate_questions(
            task_id,
            augmented_description,
            context_summary=context_summary,
        )
        task.metadata["clarifications"] = [asdict(item) for item in new_items]

        # Reset downstream artifacts so the next plan is generated from the clarified inputs.
        self._reset_task_artifacts(task)

        task.status = TaskStatus.CLARIFYING
        task.touch()
        self.store.upsert_task(task)
        self.logger.record(
            task.id,
            "CLARIFICATIONS_RESTARTED",
            {
                "reason": (reason or "").strip(),
                "question_count": len(new_items),
            },
        )
        return task.metadata.get("clarifications", [])

    def update_task_description(
        self,
        task_id: str,
        description: str,
        *,
        reason: Optional[str] = None,
        reset_metadata: bool = True,
    ) -> Task:
        """
        Update a task description safely.

        If reset_metadata is True, we treat this as a new task intent and regenerate
        clarifications, clear downstream artifacts (plan/specs/patches/bounded context),
        and move the task back to CLARIFYING.

        This is also used internally when we detect the description was edited directly
        in tasks.json.
        """
        task = self._get_task(task_id, _skip_description_guard=True)
        new_desc = (description or "").strip()
        if not new_desc:
            raise ValueError("Task description cannot be empty.")

        old_desc = (task.description or "").strip()
        if old_desc == new_desc:
            # Still update snapshot for consistency if it's missing.
            if (task.metadata.get("description_snapshot") or "").strip() != new_desc:
                task.metadata["description_snapshot"] = new_desc
                task.touch()
                self.store.upsert_task(task)
            return task

        history = task.metadata.get("description_history") or []
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "from": old_desc,
                "to": new_desc,
                "reason": (reason or "").strip(),
            }
        )
        task.metadata["description_history"] = history

        task.description = new_desc
        task.metadata["description_snapshot"] = new_desc
        # If title/summary weren't explicitly set, keep them aligned with the description.
        # We treat empty fields as "auto-derived".
        derived_title, derived_summary = self._derive_title_summary(new_desc)
        if not (task.title or "").strip():
            task.title = derived_title
        if not (task.summary or "").strip():
            task.summary = derived_summary

        if reset_metadata:
            self._preserve_clarifications_history(task, reason=reason or "Task description changed")

            context_summary = (
                (task.metadata.get("repository_summary") or {})
                if isinstance(task.metadata.get("repository_summary"), dict)
                else None
            )
            new_items = self.clarifier.generate_questions(
                task_id,
                new_desc,
                context_summary=context_summary,
            )
            task.metadata["clarifications"] = [asdict(item) for item in new_items]

            self._reset_task_artifacts(task, reset_context_steps=True)
            task.status = TaskStatus.CLARIFYING
            task.touch()
            self.store.upsert_task(task)

            # Re-seed context based on the new description (after clearing history).
            self._seed_context_from_description(task)

            self.logger.record(
                task.id,
                "TASK_DESCRIPTION_UPDATED_RESET",
                {
                    "reason": (reason or "").strip(),
                    "old_description": old_desc,
                    "new_description": new_desc,
                    "question_count": len(new_items),
                },
            )
        else:
            task.touch()
            self.store.upsert_task(task)
            self.logger.record(
                task.id,
                "TASK_DESCRIPTION_UPDATED",
                {"reason": (reason or "").strip(), "old_description": old_desc, "new_description": new_desc},
            )

        return task

    def get_context_history(self, task_id: str) -> List[Dict[str, Any]]:
        task = self._get_task(task_id)
        return task.metadata.get("context_steps", [])

    def update_context(
        self,
        task_id: str,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        trigger: str = "manual-context-update",
        note: Optional[str] = None,
        summarize: bool = True,
    ) -> Dict[str, Any]:
        task = self._get_task(task_id)
        include_paths = self._normalize_paths(task.repo_path, include or [])
        exclude_paths = self._normalize_paths(task.repo_path, exclude or [])
        step = self._record_context_step(task, trigger, include_paths, exclude_paths, note=note)

        summary = None
        if summarize and include_paths:
            target_args = [
                str(path.relative_to(task.repo_path))
                if path.is_relative_to(task.repo_path)
                else str(path)
                for path in include_paths
            ]
            summary = self.context_indexer.summarize_targets(task.repo_path, target_args)

        return {"step": step, "summary": summary}

    def infer_scope_targets(self, task_id: str) -> Dict[str, Any]:
        """
        Infer a small set of repo paths to use as frozen scope for bounded indexing.

        Primary signal: Serena semantic tree (code symbols + paths).
        Fallback signals: repo top_directories/namespaces, and lightweight filename globs for config files.
        """
        import re

        task = self._get_task(task_id)
        repo_path = task.repo_path.resolve()
        desc = (task.description or "").strip()

        clarifications = task.metadata.get("clarifications", []) or []
        # Try to extract the "directly affected modules/components" answer to bias scope.
        affected_answer = ""
        for item in clarifications:
            if not isinstance(item, dict):
                continue
            status = (item.get("status") or "").strip().upper()
            if status != "ANSWERED":
                continue
            q = (item.get("question") or "").strip().lower()
            a = (item.get("answer") or "").strip()
            if not a:
                continue
            if any(
                needle in q
                for needle in (
                    "directly affected",
                    "modules or components",
                    "components within the repository",
                    "affected by the",
                    "impacted",
                )
            ):
                affected_answer = a
                break

        answered_text = " ".join(
            (item.get("answer") or "")
            for item in clarifications
            if isinstance(item, dict) and (item.get("status") or "").strip().upper() == "ANSWERED"
        )
        text = f"{desc}\n{answered_text}".strip()

        # Build keyword set (basic, but effective).
        raw_tokens = re.findall(r"[A-Za-z0-9_.-]{3,}", text)
        keywords: set[str] = set()
        for tok in raw_tokens:
            t = tok.strip().strip(".,:;()[]{}\"'").lower()
            if len(t) < 4:
                continue
            # Skip very common noise tokens
            if t in {"this", "that", "with", "from", "into", "only", "should", "need", "have", "been"}:
                continue
            keywords.add(t)

        # Always seed with common config/security terms.
        keywords |= {"oauth", "oauth2", "token", "secret", "secrets", "encrypt", "encrypted", "credential", "credentials", "appsettings"}

        # Focus tokens: when the engineer answers "this is about encrypted service",
        # we should bias scope toward that component and its tests, not the whole repo.
        focus_tokens: set[str] = set()
        if affected_answer:
            # Extract words and also a compact "joined" token.
            words = re.findall(r"[A-Za-z0-9]{3,}", affected_answer.lower())
            joined = re.sub(r"[^a-z0-9]+", "", affected_answer.lower())
            for w in words:
                if len(w) >= 4:
                    focus_tokens.add(w)
            if len(joined) >= 6:
                focus_tokens.add(joined)
            # Common expansions for "encrypted service" phrasing
            if "encrypted" in focus_tokens and "service" in focus_tokens:
                focus_tokens.add("encryptedservice")
            if "encryption" in focus_tokens and "service" in focus_tokens:
                focus_tokens.add("encryptionservice")
            # A few relevant stems
            for stem in ("encrypt", "encrypted", "encryption", "encryptionservice", "encryptedservice"):
                if stem in joined:
                    focus_tokens.add(stem)

        def _is_noise_path(rel_path: str) -> bool:
            p = (rel_path or "").replace("\\", "/").lower()
            return any(
                token in p
                for token in (
                    "/bin/",
                    "/obj/",
                    "/node_modules/",
                    "/dist/",
                    "/build/",
                    "/.git/",
                    "/.venv/",
                    "/.idea/",
                )
            )

        def _load_serena_tree_payload() -> Dict[str, Any] | None:
            summary = task.metadata.get("repository_summary") or {}
            if isinstance(summary, dict) and summary.get("serena_semantic_tree"):
                return summary.get("serena_semantic_tree")
            # If the task was created before background Serena export finished, use latest repository index.
            try:
                index = self.store.load_repository_index()
                if Path(index.get("repo_path", "")).resolve() == repo_path:
                    rs = index.get("repository_summary") or {}
                    if isinstance(rs, dict) and rs.get("serena_semantic_tree"):
                        return rs.get("serena_semantic_tree")
            except Exception:
                return None
            return None

        def _flatten_tree_files(tree_node: Dict[str, Any]) -> list[dict]:
            out: list[dict] = []
            stack = [tree_node]
            while stack:
                node = stack.pop()
                if not isinstance(node, dict):
                    continue
                t = node.get("type")
                if t == "file":
                    out.append(node)
                for child in (node.get("children") or []):
                    stack.append(child)
            return out

        targets: list[str] = []
        sample_files: list[str] = []
        reason = "fallback"

        serena = _load_serena_tree_payload()
        if isinstance(serena, dict) and isinstance(serena.get("tree"), dict):
            reason = "serena-semantic-tree"
            files = _flatten_tree_files(serena["tree"])
            scored: list[tuple[int, str]] = []
            for f in files:
                path = (f.get("path") or "").replace("\\", "/")
                if not path:
                    continue
                if _is_noise_path(path):
                    continue
                p_l = path.lower()
                score = 0

                # If we have a focused component, require at least one focus token match
                # in path/symbols/overview to avoid pulling unrelated modules.
                focus_match = False
                if focus_tokens:
                    if any(tok in p_l for tok in focus_tokens):
                        focus_match = True
                    else:
                        for sym in (f.get("symbols") or []):
                            name = (sym.get("name") or "").lower()
                            if any(tok in name for tok in focus_tokens):
                                focus_match = True
                                break
                    if not focus_match:
                        overview = (f.get("overview") or "").lower()
                        if any(tok in overview for tok in focus_tokens):
                            focus_match = True

                # Path matches are strong.
                for k in keywords:
                    if k in p_l:
                        score += 3
                # Symbol name matches are medium.
                for sym in (f.get("symbols") or []):
                    name = (sym.get("name") or "").lower()
                    for k in keywords:
                        if k and k in name:
                            score += 2
                # Overview matches are weak but helpful.
                overview = (f.get("overview") or "").lower()
                for k in keywords:
                    if k in overview:
                        score += 1

                # Apply strong bias toward focused component (but still allow tests/config
                # if they reference the focused component).
                if focus_tokens:
                    if not focus_match:
                        # Drop unrelated files entirely.
                        continue
                    score += 10

                if score > 0:
                    scored.append((score, path))

            scored.sort(key=lambda x: (-x[0], x[1]))
            top_files = [p for _, p in scored[:30]]
            sample_files = top_files[:12]

            # Convert to directory targets (prefer shallow dirs).
            dir_hits: dict[str, int] = {}
            for score, p in scored[:80]:
                parts = p.split("/")
                if len(parts) >= 2:
                    # When focus exists, use a slightly deeper directory (e.g. src/Foo.Domain/Services)
                    # to avoid scoping the entire module root.
                    depth = 3 if focus_tokens else 2
                    d = "/".join(parts[: min(len(parts) - 1, depth)])
                else:
                    d = p
                if _is_noise_path(d):
                    continue
                dir_hits[d] = dir_hits.get(d, 0) + score

            ranked_dirs = sorted(dir_hits.items(), key=lambda x: (-x[1], x[0]))
            targets = [d for d, _ in ranked_dirs[:8]]

        # Fallback: if we couldn't infer from Serena, use top-level dirs and common config files.
        if not targets:
            summary = task.metadata.get("repository_summary") or {}
            if isinstance(summary, dict):
                top_dirs = summary.get("top_directories") or []
                if top_dirs:
                    targets = [str(d).rstrip("/") + "/" for d in top_dirs[:6]]
                    reason = "serena-top-directories"

        # Add lightweight config file directories (without scanning whole repo).
        config_candidates: list[Path] = []
        for pat in ("**/appsettings*.json", "**/*oauth*.*", "**/*token*.*", "**/*secret*.*"):
            try:
                for p in list(repo_path.glob(pat))[:8]:
                    config_candidates.append(p)
            except Exception:
                continue
        for p in config_candidates[:12]:
            try:
                rel = str(p.relative_to(repo_path)).replace("\\", "/")
            except Exception:
                continue
            if _is_noise_path(rel):
                continue
            parent = rel.rsplit("/", 1)[0] if "/" in rel else rel
            if parent and parent not in targets:
                targets.append(parent)

        # Validate existence and normalize.
        normalized: list[str] = []
        for t in targets:
            tt = str(t).strip().strip("/")
            if not tt:
                continue
            if _is_noise_path(tt):
                continue
            cand = (repo_path / tt).resolve()
            if cand.exists():
                normalized.append(tt)
        # De-dupe preserving order
        seen: set[str] = set()
        final_targets: list[str] = []
        for t in normalized:
            if t in seen:
                continue
            seen.add(t)
            final_targets.append(t)

        return {"targets": final_targets[:12], "sample_files": sample_files, "reason": reason}

    # ------------------------------------------------------------------ Planning
    @staticmethod
    def _collect_directory_candidates_from_summary(summary: Dict[str, Any]) -> List[str]:
        """
        Extract directory paths (relative, POSIX) from the indexed directory tree.

        This is intentionally shallow (whatever the index stored) to avoid rescanning
        large repos during planning.
        """
        root = summary.get("directory_structure")
        if not isinstance(root, dict):
            return []

        results: List[str] = []

        def walk(node: Dict[str, Any]) -> None:
            if not isinstance(node, dict):
                return
            if node.get("type") == "directory":
                rel = str(node.get("path") or "").replace("\\", "/")
                if rel and rel != ".":
                    results.append(rel)
            for child in (node.get("children") or []):
                if isinstance(child, dict):
                    walk(child)

        walk(root)
        # Prefer stable order for debugging and deterministic tests.
        return sorted(set(results), key=lambda p: (p.count("/"), len(p), p))

    def _resolve_plan_targets_to_paths(
        self,
        repo_path: Path,
        raw_targets: List[str],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert plan "target_modules" (often logical names like 'Api', 'Data', 'Pbp.Payments')
        into real filesystem paths for bounded indexing (best-effort).

        Strategy:
        - If raw target already exists as a path under repo_path, keep it.
        - Otherwise, try namespace → path mapping (dots to slashes).
        - Otherwise, fuzzy match against indexed directory names (from directory_structure).
        """
        repo_path = repo_path.resolve()
        directory_candidates = self._collect_directory_candidates_from_summary(summary)
        # Fallback if index didn't include a tree (or was produced by older versions).
        if not directory_candidates:
            try:
                directory_candidates = sorted(
                    {p.name for p in repo_path.iterdir() if p.is_dir()},
                    key=lambda p: (len(p), p),
                )
            except OSError:
                directory_candidates = []

        resolved: List[str] = []
        unresolved: List[str] = []
        matches: Dict[str, List[str]] = {}

        def norm_token(value: str) -> str:
            return (
                (value or "")
                .strip()
                .strip("/")
                .strip("\\")
                .replace("\\", "/")
            )

        def fuzzy_key(value: str) -> str:
            return (
                value.lower()
                .replace("-", "")
                .replace("_", "")
                .replace(".", "")
                .replace("/", "")
                .strip()
            )

        # Precompute candidate name map for faster matching.
        candidate_names = []
        for cand in directory_candidates:
            cand_rel = norm_token(cand)
            cand_name = cand_rel.rsplit("/", 1)[-1]
            candidate_names.append((cand_rel, cand_name, fuzzy_key(cand_name)))

        def _repo_seems_terraform(summary_payload: Dict[str, Any]) -> bool:
            # Summary values can vary (strings like ".tf (123)" / "Terraform", etc.).
            frameworks = summary_payload.get("frameworks") or []
            if any(str(f).lower() == "terraform" for f in frameworks):
                return True
            top_langs = summary_payload.get("top_languages") or []
            if any("terraform" in str(x).lower() for x in top_langs):
                return True
            exts = summary_payload.get("top_file_extensions") or []
            if any(".tf" in str(x).lower() for x in exts):
                return True
            return False

        repo_is_terraform = _repo_seems_terraform(summary or {})

        def default_terraform_paths() -> List[str]:
            """
            When the repo is clearly Terraform-heavy but plan targets are generic labels,
            pick a small, high-signal set of Terraform/IAM files for bounded indexing.
            """
            hits: List[str] = []

            def add_if_exists(p: Path) -> None:
                if not p.exists():
                    return
                try:
                    rel = str(p.relative_to(repo_path)).replace("\\", "/")
                except ValueError:
                    rel = str(p).replace("\\", "/")
                if rel and rel not in hits:
                    hits.append(rel)

            for fname in (
                "main.tf",
                "variables.tf",
                "outputs.tf",
                "providers.tf",
                "provider.tf",
                "versions.tf",
                "locals.tf",
            ):
                add_if_exists(repo_path / fname)

            # Common infra directories
            for dname in ("modules", "terraform", "infra", "infrastructure", "iam"):
                add_if_exists(repo_path / dname)

            # If nothing obvious exists, sample a few *.tf files.
            if not hits:
                tf_files: List[Path] = []
                try:
                    for p in repo_path.rglob("*.tf"):
                        if ".git" in p.parts or ".terraform" in p.parts:
                            continue
                        tf_files.append(p)
                        if len(tf_files) >= 30:
                            break
                except OSError:
                    tf_files = []
                tf_files = sorted(tf_files, key=lambda p: (len(p.parts), len(str(p))))
                for p in tf_files[:12]:
                    add_if_exists(p)

            # Add IAM-y terraform files if present
            iam_files: List[Path] = []
            try:
                for p in repo_path.rglob("*iam*.tf"):
                    if ".git" in p.parts or ".terraform" in p.parts:
                        continue
                    iam_files.append(p)
                    if len(iam_files) >= 20:
                        break
            except OSError:
                iam_files = []
            iam_files = sorted(iam_files, key=lambda p: (len(p.parts), len(str(p))))
            for p in iam_files[:10]:
                add_if_exists(p)

            return hits[:12]

        def heuristic_matches(target: str) -> List[str]:
            """
            Best-effort mapping from logical labels (e.g. "Terraform Configuration")
            to real filesystem paths (e.g. main.tf, variables.tf, iam/*.tf).
            """
            t_lower = (target or "").lower()
            hits: List[str] = []

            def add_if_exists(p: Path) -> None:
                if not p.exists():
                    return
                try:
                    rel = str(p.relative_to(repo_path)).replace("\\", "/")
                except ValueError:
                    rel = str(p).replace("\\", "/")
                if rel and rel not in hits:
                    hits.append(rel)

            is_terraformish = any(k in t_lower for k in ("terraform", "iac", "infra", "infrastructure", "hcl", ".tf"))
            is_iamish = ("iam" in t_lower) or ("role" in t_lower)

            if is_terraformish:
                for fname in (
                    "main.tf",
                    "variables.tf",
                    "outputs.tf",
                    "providers.tf",
                    "provider.tf",
                    "versions.tf",
                    "locals.tf",
                ):
                    add_if_exists(repo_path / fname)
                for dname in ("terraform", "infra", "infrastructure", "modules"):
                    add_if_exists(repo_path / dname)

                # Fallback: include a small sample of *.tf if nothing obvious exists.
                if not hits:
                    tf_files: List[Path] = []
                    try:
                        for p in repo_path.rglob("*.tf"):
                            if ".git" in p.parts or ".terraform" in p.parts:
                                continue
                            tf_files.append(p)
                            if len(tf_files) >= 24:
                                break
                    except OSError:
                        tf_files = []
                    tf_files = sorted(tf_files, key=lambda p: (len(p.parts), len(str(p))))
                    for p in tf_files[:10]:
                        add_if_exists(p)

            if is_iamish:
                add_if_exists(repo_path / "iam")
                iam_files: List[Path] = []
                try:
                    for p in repo_path.rglob("*iam*.tf"):
                        if ".git" in p.parts or ".terraform" in p.parts:
                            continue
                        iam_files.append(p)
                        if len(iam_files) >= 20:
                            break
                except OSError:
                    iam_files = []
                iam_files = sorted(iam_files, key=lambda p: (len(p.parts), len(str(p))))
                for p in iam_files[:10]:
                    add_if_exists(p)

            return hits[:12]

        for raw in raw_targets or []:
            t = norm_token(str(raw))
            if not t:
                continue

            local_matches: List[str] = []

            # 1) Direct path match
            direct = (repo_path / t).resolve()
            if direct.exists():
                try:
                    local_matches.append(str(direct.relative_to(repo_path)).replace("\\", "/"))
                except ValueError:
                    local_matches.append(str(direct).replace("\\", "/"))

            # 2) Namespace path mapping (Pbp.Payments.Auth -> Pbp/Payments/Auth)
            if not local_matches and "." in t and "/" not in t:
                ns_candidate = t.replace(".", "/")
                ns_path = (repo_path / ns_candidate).resolve()
                if ns_path.exists():
                    try:
                        local_matches.append(str(ns_path.relative_to(repo_path)).replace("\\", "/"))
                    except ValueError:
                        local_matches.append(str(ns_path).replace("\\", "/"))

            # 3) Fuzzy directory matching by name
            if not local_matches and directory_candidates:
                t_key = fuzzy_key(t)
                last_segment = t.split(".")[-1] if "." in t else t
                last_key = fuzzy_key(last_segment)
                for cand_rel, cand_name, cand_key in candidate_names:
                    if not cand_rel:
                        continue
                    # Exact-ish
                    if cand_name.lower() == t.lower():
                        local_matches.append(cand_rel)
                        continue
                    # Fuzzy contains (Api -> api, data-api -> dataapi)
                    if t_key and t_key in cand_key:
                        local_matches.append(cand_rel)
                        continue
                    # Namespace last segment hint (Pbp.Payments -> Payments)
                    if last_key and last_key == cand_key:
                        local_matches.append(cand_rel)

            # 4) Heuristic mapping for common logical labels (Terraform, IAM, etc.)
            if not local_matches:
                local_matches.extend(heuristic_matches(t))

            # 5) Terraform repo fallback: plan targets are often abstract labels, so
            # include core terraform/IAM files to avoid an empty scoped index.
            if not local_matches and repo_is_terraform:
                local_matches.extend(default_terraform_paths())

            # Keep a small, deterministic set of matches.
            local_matches = sorted(set(local_matches), key=lambda p: (p.count("/"), len(p), p))[:6]
            matches[t] = local_matches

            if local_matches:
                resolved.extend(local_matches)
            else:
                unresolved.append(t)

        resolved_unique = sorted(set(resolved), key=lambda p: (p.count("/"), len(p), p))
        unresolved_unique = sorted(set(unresolved))

        return {
            "raw_targets": list(raw_targets or []),
            "resolved_targets": resolved_unique,
            "unresolved_targets": unresolved_unique,
            "matches": matches,
            "candidate_directory_count": len(directory_candidates),
        }

    def generate_plan(self, task_id: str, skip_rationale_enhancement: bool = False) -> Dict[str, List[str]]:
        import sys
        
        task = self._get_task(task_id)
        base_summary = task.metadata.get("repository_summary", {})
        bounded_context = task.metadata.get("bounded_context", {}) or {}
        scoped_manual = bounded_context.get("manual")
        # Prefer manual bounded index (post-clarifications) if present; otherwise fall back to plan-targets after plan creation.
        context_summary = dict(base_summary)
        if scoped_manual:
            context_summary["scoped_context"] = scoped_manual
            context_summary["scoped_context_source"] = "bounded_context.manual"

        # Include answered clarifications in the plan prompt so the plan actually
        # reflects user-provided constraints (e.g., "this repo does not have tests").
        augmented_description = task.description.strip()
        clarifications = task.metadata.get("clarifications", []) or []
        answered_lines: List[str] = []
        for item in clarifications:
            if not isinstance(item, dict):
                continue
            status = (item.get("status") or "").strip().upper()
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if status == "ANSWERED" and question and answer:
                answered_lines.append(f"- Q: {question}\n  A: {answer}")
        if answered_lines:
            augmented_description = f"{augmented_description}\n\nClarifications (answered):\n" + "\n".join(answered_lines)
        
        # Show progress
        sys.stderr.write("Building plan...\n")
        plan = self.plan_builder.build_plan(task.id, augmented_description, context_summary)
        sys.stderr.write(f"Plan created with {len(plan.steps)} steps\n")
        
        # Defensive: if no tests are detected in the repo and the change request
        # doesn't explicitly ask for tests, strip any test-related plan steps that
        # the LLM might have suggested anyway.
        if not bool(context_summary.get("has_tests", False)):
            desc_lower = augmented_description.lower()
            wants_tests = any(token in desc_lower for token in ("test", "tests", "pytest", "jest", "integration test", "unit test"))
            if not wants_tests and plan.steps:
                filtered = []
                for step in plan.steps:
                    text = f"{step.description} {step.notes or ''}".lower()
                    if "test" in text:
                        continue
                    filtered.append(step)
                plan.steps = filtered

        # Create BoundaryManager with LLM client and context for this plan
        sys.stderr.write("Detecting boundaries...\n")
        boundary_manager = BoundaryManager(
            llm_client=self.llm_client,
            context_summary=context_summary
        )
        specs = boundary_manager.required_specs(plan)
        sys.stderr.write(f"Found {len(specs)} boundary specs\n")

        sys.stderr.write("Generating refactor suggestions...\n")
        refactors = self.refactor_advisor.suggest(plan)
        sys.stderr.write("Plan generation complete!\n")

        previous_statuses = {
            spec.get("boundary_name"): spec.get("status", BoundarySpecStatus.PENDING.value)
            for spec in task.metadata.get("boundary_specs", [])
        }
        for spec in specs:
            prev_status = previous_statuses.get(spec.boundary_name)
            if prev_status:
                try:
                    spec.status = BoundarySpecStatus(prev_status)
                except ValueError:
                    LOG.debug("Ignoring unknown spec status '%s' for %s", prev_status, spec.boundary_name)

        # Serialize preliminary plan artifacts.
        plan_preview = {
            "id": plan.id,
            "steps": [step.to_dict() for step in plan.steps],
            "risks": plan.risks,
            "refactors": plan.refactor_suggestions,
        }
        serialized_specs = [
            {
                "id": spec.id,
                "task_id": spec.task_id,
                "boundary_name": spec.boundary_name,
                "human_description": spec.human_description,
                "diagram_text": spec.diagram_text,
                "machine_spec": spec.machine_spec,
                "status": spec.status.value,
                "plan_step": spec.plan_step,
            }
            for spec in specs
        ]

        pending_specs = [spec["boundary_name"] for spec in serialized_specs if spec["status"] == BoundarySpecStatus.PENDING.value]

        # Resolve plan targets to paths and build a frozen scoped index (allowlist).
        # Note: the final plan regeneration happens when the preliminary plan is approved
        # in chat (see build_final_plan_with_frozen_scope()).
        plan_targets = sorted(
            {
                target
                for step in plan.steps
                for target in (step.target_files or [])
                if target
            }
        )
        plan_targets_resolution = None
        resolved_targets: list[str] = []

        if plan_targets:
            plan_targets_resolution = self._resolve_plan_targets_to_paths(
                task.repo_path,
                plan_targets,
                base_summary or {},
            )
            resolved_targets = list(plan_targets_resolution.get("resolved_targets") or [])

        if resolved_targets:
            include_paths: List[Path] = []
            for rel in resolved_targets:
                candidate = (task.repo_path / rel).resolve()
                if candidate.exists():
                    include_paths.append(candidate)
            if include_paths:
                self._record_context_step(
                    task,
                    "plan-targets",
                    included=include_paths,
                    note="Auto-selected from plan targets (resolved to filesystem paths)",
                )

            bounded_summary = self.context_indexer.summarize_targets(
                task.repo_path,
                resolved_targets,
                include_file_list=True,
            )
            bounded_summary["plan_targets_resolution"] = plan_targets_resolution
            task.metadata.setdefault("bounded_context", {})
            task.metadata["bounded_context"]["plan_targets"] = bounded_summary
            # If no manual scope is present, this plan is a preliminary plan that can be
            # finalized later by regenerating with a frozen allowlist.
            task.metadata["plan_stage"] = "PRELIMINARY"

        # If we already have a frozen manual bounded scope (from clarifications),
        # treat the plan as FINAL immediately (no second pass required).
        if scoped_manual:
            scope_meta = (scoped_manual.get("scope") or {}) if isinstance(scoped_manual, dict) else {}
            frozen = bool(scope_meta.get("frozen"))
            agg = (scoped_manual.get("aggregate") or {}) if isinstance(scoped_manual, dict) else {}
            file_count = agg.get("file_count", 0) if isinstance(agg, dict) else 0
            if frozen and isinstance(file_count, int) and file_count > 0:
                task.metadata["plan_stage"] = "FINAL"
        # Default stage if not otherwise set.
        task.metadata.setdefault("plan_stage", "PRELIMINARY")

        task.metadata["plan_preview"] = plan_preview
        task.metadata["boundary_specs"] = serialized_specs
        task.metadata["pending_specs"] = pending_specs
        task.metadata["refactor_suggestions"] = [item.to_dict() for item in refactors]
        task.metadata["plan_approved"] = False
        task.metadata.pop("plan_approved_at", None)
        for key in ("patch_queue", "patch_queue_state", "test_suggestions"):
            task.metadata.pop(key, None)

        task.status = TaskStatus.SPEC_PENDING if pending_specs else TaskStatus.PLANNING
        self._snapshot_worktree_status(task)
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "PLAN_GENERATED",
            {
                "plan": plan_preview,
                "pending_specs": pending_specs,
                "refactor_suggestions": task.metadata["refactor_suggestions"],
                "plan_targets_indexed": bool(
                    (plan_targets_resolution or {}).get("resolved_targets")
                    if plan_targets_resolution
                    else plan_targets
                ),
                "plan_targets_resolution": plan_targets_resolution,
            },
        )

        return_payload = {
            "plan": plan_preview,
            "pending_specs": pending_specs,
            "refactor_suggestions": task.metadata["refactor_suggestions"],
            # Prefer manual bounded index when available; otherwise show plan-targets scope.
            "bounded_context": (task.metadata.get("bounded_context", {}) or {}).get("manual")
            or (task.metadata.get("bounded_context", {}) or {}).get("plan_targets"),
        }
        if plan_targets_resolution:
            return_payload["plan_targets_resolution"] = plan_targets_resolution
        return return_payload

    def build_final_plan_with_frozen_scope(self, task_id: str) -> Dict[str, Any]:
        """
        After a preliminary plan is approved (in chat), freeze scope and regenerate a final plan
        using the frozen allowlist from bounded_context.plan_targets (or bounded_context.manual).
        """
        import sys

        task = self._get_task(task_id)
        base_summary = task.metadata.get("repository_summary", {}) or {}
        bounded_context = task.metadata.get("bounded_context", {}) or {}

        # Prefer manual bounded scope if present; it is already frozen (allowlist).
        manual = bounded_context.get("manual")
        plan_targets_scope = bounded_context.get("plan_targets")
        scoped = manual or plan_targets_scope
        if not isinstance(scoped, dict) or not scoped:
            raise ValueError("No scoped context available to freeze. Provide scope paths during chat or via bounded-index.")

        aggregate = (scoped.get("aggregate") or {}) if isinstance(scoped, dict) else {}
        scoped_files = aggregate.get("file_count", 0) if isinstance(aggregate, dict) else 0
        if not isinstance(scoped_files, int) or scoped_files <= 0:
            raise ValueError("Scoped context is empty; cannot build a final plan. Provide real filesystem paths for scope.")

        # Use the same augmented description logic as generate_plan (include answered clarifications).
        augmented_description = task.description.strip()
        clarifications = task.metadata.get("clarifications", []) or []
        answered_lines: List[str] = []
        for item in clarifications:
            if not isinstance(item, dict):
                continue
            status = (item.get("status") or "").strip().upper()
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if status == "ANSWERED" and question and answer:
                answered_lines.append(f"- Q: {question}\n  A: {answer}")
        if answered_lines:
            augmented_description = f"{augmented_description}\n\nClarifications (answered):\n" + "\n".join(answered_lines)

        sys.stderr.write(f"Rebuilding final plan with frozen scoped context ({scoped_files} files)...\n")
        second_pass_context = dict(base_summary)
        second_pass_context["scoped_context"] = scoped
        second_pass_context["scoped_context_source"] = (
            "bounded_context.manual" if manual else "bounded_context.plan_targets"
        )

        plan = self.plan_builder.build_plan(task.id, augmented_description, second_pass_context)
        sys.stderr.write(f"Final plan created with {len(plan.steps)} steps\n")

        sys.stderr.write("Detecting boundaries (final plan)...\n")
        boundary_manager = BoundaryManager(
            llm_client=self.llm_client,
            context_summary=second_pass_context,
        )
        specs = boundary_manager.required_specs(plan)
        sys.stderr.write(f"Found {len(specs)} boundary specs (final plan)\n")

        sys.stderr.write("Generating refactor suggestions (final plan)...\n")
        refactors = self.refactor_advisor.suggest(plan)

        # Preserve preliminary plan for auditability.
        if task.metadata.get("plan_preview"):
            task.metadata["plan_preview_preliminary"] = task.metadata.get("plan_preview")

        plan_preview = {
            "id": plan.id,
            "steps": [step.to_dict() for step in plan.steps],
            "risks": plan.risks,
            "refactors": plan.refactor_suggestions,
        }
        serialized_specs = [
            {
                "id": spec.id,
                "task_id": spec.task_id,
                "boundary_name": spec.boundary_name,
                "human_description": spec.human_description,
                "diagram_text": spec.diagram_text,
                "machine_spec": spec.machine_spec,
                "status": spec.status.value,
                "plan_step": spec.plan_step,
            }
            for spec in specs
        ]
        pending_specs = [
            spec["boundary_name"]
            for spec in serialized_specs
            if spec["status"] == BoundarySpecStatus.PENDING.value
        ]

        task.metadata["plan_preview"] = plan_preview
        task.metadata["boundary_specs"] = serialized_specs
        task.metadata["pending_specs"] = pending_specs
        task.metadata["refactor_suggestions"] = [item.to_dict() for item in refactors]
        task.metadata["plan_stage"] = "FINAL"
        task.metadata["plan_approved"] = False
        task.metadata.pop("plan_approved_at", None)
        for key in ("patch_queue", "patch_queue_state", "test_suggestions"):
            task.metadata.pop(key, None)

        task.status = TaskStatus.SPEC_PENDING if pending_specs else TaskStatus.PLANNING
        self._snapshot_worktree_status(task)
        task.touch()
        self.store.upsert_task(task)

        return {"status": "FINAL_BUILT", "plan": plan_preview, "pending_specs": pending_specs}

    def bounded_index_task(
        self,
        task_id: str,
        targets: List[str],
        *,
        include_serena_semantic_tree: bool = False,
    ) -> Dict[str, any]:
        """
        Run a bounded index over a subset of repo paths and store the summary.
        """
        task = self._get_task(task_id)
        summary = self.context_indexer.summarize_targets(
            task.repo_path,
            targets,
            include_serena_semantic_tree=include_serena_semantic_tree,
            include_file_list=True,
        )
        aggregate = summary.get("aggregate", {}) if isinstance(summary, dict) else {}
        file_count = aggregate.get("file_count", 0) if isinstance(aggregate, dict) else 0
        targets_indexed = list((summary.get("targets", {}) or {}).keys()) if isinstance(summary, dict) else []

        # Defensive: don't persist an empty bounded index. In practice, this usually
        # means the provided targets were invalid or were module labels rather than paths.
        if isinstance(file_count, int) and file_count > 0 and targets_indexed:
            task.metadata.setdefault("bounded_context", {})
            task.metadata["bounded_context"]["manual"] = summary
            include_paths = [
                (task.repo_path / Path(target)).resolve()
                if not Path(target).is_absolute()
                else Path(target).resolve()
                for target in targets
            ]
            self._record_context_step(
                task,
                "bounded-index",
                included=include_paths,
                note="Engineer requested scoped index",
            )
            task.touch()
            self.store.upsert_task(task)
            self.logger.record(
                task.id,
                "BOUNDED_INDEX_CREATED",
                {"targets": targets_indexed},
            )
        return summary

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

        found_spec = None
        for spec in specs:
            if spec["id"] == spec_id:
                spec["status"] = "APPROVED"
                found_spec = spec
                break

        if not found_spec:
            raise ValueError(f"Boundary spec not found: {spec_id}")

        task.metadata["boundary_specs"] = specs
        self._update_pending_specs(task)
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "SPEC_APPROVED",
            {"spec_id": spec_id, "boundary_name": found_spec.get("boundary_name")}
        )

        return {"spec_id": spec_id, "status": "APPROVED"}

    def skip_spec(self, task_id: str, spec_id: str) -> Dict:
        """
        Skip (override) a boundary specification.
        """
        task = self._get_task(task_id)
        specs = task.metadata.get("boundary_specs", [])

        found_spec = None
        for spec in specs:
            if spec["id"] == spec_id:
                spec["status"] = "SKIPPED"
                found_spec = spec
                break

        if not found_spec:
            raise ValueError(f"Boundary spec not found: {spec_id}")

        task.metadata["boundary_specs"] = specs
        self._update_pending_specs(task)
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "SPEC_SKIPPED",
            {"spec_id": spec_id, "boundary_name": found_spec.get("boundary_name")}
        )

        return {"spec_id": spec_id, "status": "SKIPPED"}

    def approve_all_specs(self, task_id: str) -> Dict[str, Any]:
        """
        Approve all remaining pending boundary specs for a task.
        """
        specs = self.get_boundary_specs(task_id)
        pending_ids = [
            s.get("id")
            for s in (specs or [])
            if (s.get("status") or "") == BoundarySpecStatus.PENDING.value and s.get("id")
        ]
        for spec_id in pending_ids:
            self.approve_spec(task_id, str(spec_id))
        return {"approved_count": len(pending_ids), "approved_spec_ids": pending_ids}

    def skip_all_specs(self, task_id: str) -> Dict[str, Any]:
        """
        Skip all remaining pending boundary specs for a task.
        """
        specs = self.get_boundary_specs(task_id)
        pending_ids = [
            s.get("id")
            for s in (specs or [])
            if (s.get("status") or "") == BoundarySpecStatus.PENDING.value and s.get("id")
        ]
        for spec_id in pending_ids:
            self.skip_spec(task_id, str(spec_id))
        return {"skipped_count": len(pending_ids), "skipped_spec_ids": pending_ids}

    def update_spec_fields(
        self,
        task_id: str,
        spec_id: str,
        *,
        description: Optional[str] = None,
        diagram_text: Optional[str] = None,
        machine_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task = self._get_task(task_id)
        specs = task.metadata.get("boundary_specs", [])
        target = None
        for spec in specs:
            if spec["id"] == spec_id:
                target = spec
                break

        if not target:
            raise ValueError(f"Boundary spec not found: {spec_id}")

        if description is not None:
            target["human_description"] = description
        if diagram_text is not None:
            target["diagram_text"] = diagram_text
        if machine_spec is not None:
            target["machine_spec"] = machine_spec

        # Editing a spec re-opens approval
        target["status"] = BoundarySpecStatus.PENDING.value
        task.metadata["boundary_specs"] = specs
        self._update_pending_specs(task)
        task.touch()
        self.store.upsert_task(task)
        self.logger.record(
            task.id,
            "SPEC_UPDATED",
            {"spec_id": spec_id, "description_changed": description is not None},
        )
        return target

    def regenerate_spec(self, task_id: str, spec_id: str) -> Dict[str, Any]:
        """
        Regenerate a boundary spec from its associated plan step using the BoundaryManager.
        """
        task = self._get_task(task_id)
        specs = task.metadata.get("boundary_specs", [])
        target = None
        for spec in specs:
            if spec["id"] == spec_id:
                target = spec
                break

        if not target:
            raise ValueError(f"Boundary spec not found: {spec_id}")

        plan_step_description = target.get("plan_step")
        if not plan_step_description:
            raise ValueError("This boundary spec is not linked to a plan step and cannot be regenerated automatically.")

        plan_preview = task.metadata.get("plan_preview", {})
        plan_steps = [
            PlanStep.from_dict(step) if isinstance(step, dict) else PlanStep(description=str(step))
            for step in plan_preview.get("steps", [])
        ]
        plan_step = next((step for step in plan_steps if step.description == plan_step_description), None)
        if not plan_step:
            raise ValueError("Unable to locate the original plan step for this boundary spec.")

        boundary_manager = BoundaryManager(
            llm_client=self.llm_client,
            context_summary=task.metadata.get("repository_summary"),
        )
        new_spec = boundary_manager.generate_spec_for_step(task.id, plan_step)

        target.update(
            {
                "boundary_name": new_spec.boundary_name,
                "human_description": new_spec.human_description,
                "diagram_text": new_spec.diagram_text,
                "machine_spec": new_spec.machine_spec,
                "status": BoundarySpecStatus.PENDING.value,
                "plan_step": plan_step.description,
            }
        )
        task.metadata["boundary_specs"] = specs
        self._update_pending_specs(task)
        task.touch()
        self.store.upsert_task(task)
        self.logger.record(task.id, "SPEC_REGENERATED", {"spec_id": spec_id})
        return target

    def approve_plan(self, task_id: str) -> Dict:
        """
        Approve the entire implementation plan.

        This approves the plan at a high level rather than requiring
        approval of individual boundary specifications.
        """
        task = self._get_task(task_id)

        unresolved = [
            spec for spec in task.metadata.get("boundary_specs", [])
            if spec.get("status") == BoundarySpecStatus.PENDING.value
        ]
        if unresolved:
            raise ValueError("All boundary specs must be approved or skipped before approving the plan.")

        # Mark plan as approved
        task.metadata["plan_approved"] = True
        task.metadata["plan_approved_at"] = task.updated_at.isoformat()

        # Update task status
        task.status = TaskStatus.IMPLEMENTING
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "PLAN_APPROVED",
            {
                "plan_steps": len(task.metadata.get("plan_preview", {}).get("steps", [])),
                "boundary_specs": len(task.metadata.get("boundary_specs", []))
            }
        )

        return {"status": "APPROVED", "task_id": task_id}

    def export_approved_plan_markdown(self, task_id: str, *, output_path: Path | None = None) -> Path:
        """
        Export the approved plan to a Markdown file under the target repository.

        Default path: <repo>/docs/plans/<slugified-description>.md (with a short task id suffix if needed)
        """
        task = self._get_task(task_id)
        if not task.metadata.get("plan_approved", False):
            raise ValueError("Plan must be approved before exporting Markdown.")

        plan_preview = task.metadata.get("plan_preview", {}) or {}
        steps = plan_preview.get("steps", []) or []
        risks = plan_preview.get("risks", []) or []
        refactors = plan_preview.get("refactors", []) or []
        specs = task.metadata.get("boundary_specs", []) or []

        bounded_context = task.metadata.get("bounded_context", {}) or {}
        bounded = bounded_context.get("manual") or bounded_context.get("plan_targets") or {}
        scoped_aggregate = (bounded.get("aggregate") or {}) if isinstance(bounded, dict) else {}

        repo_path = task.repo_path.resolve()
        docs_dir = repo_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        plans_dir = docs_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            raw_title = (task.description or "").strip() or f"task-{task.id}"
            title_line = raw_title.splitlines()[0].strip() if raw_title else f"task-{task.id}"

            def _slugify(value: str) -> str:
                import re

                s = (value or "").strip().lower()
                s = re.sub(r"[^a-z0-9]+", "-", s)
                s = re.sub(r"-{2,}", "-", s).strip("-")
                return s or f"task-{task.id}"

            slug = _slugify(title_line)[:80].rstrip("-")
            candidate = plans_dir / f"{slug}.md"
            if candidate.exists():
                candidate = plans_dir / f"{slug}-{task.id[:8]}.md"
            output_path = candidate

        def _fmt_list(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items) if items else "_None_"

        lines: list[str] = []
        title = (task.description or "").strip().splitlines()[0].strip() if (task.description or "").strip() else "Spec Agent Plan"
        if len(title) > 120:
            title = title[:117].rstrip() + "..."
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"- **Task ID**: `{task.id}`")
        lines.append(f"- **Repo**: `{repo_path}`")
        lines.append(f"- **Branch**: `{task.branch}`")
        if task.metadata.get("plan_approved_at"):
            lines.append(f"- **Approved at**: `{task.metadata.get('plan_approved_at')}`")
        lines.append("")
        lines.append("## Change request")
        lines.append("")
        lines.append(task.description.strip() or "_(missing description)_")
        lines.append("")

        repo_summary = task.metadata.get("repository_summary", {}) or {}
        repo_files = repo_summary.get("file_count", "unknown")
        scoped_files = scoped_aggregate.get("file_count")
        if isinstance(scoped_files, int):
            lines.append("## Scope")
            lines.append("")
            lines.append(f"- **Total files (scoped)**: {scoped_files} (repo: {repo_files})")
            targets = list(((bounded.get("targets") or {}) if isinstance(bounded, dict) else {}).keys())
            if targets:
                lines.append(f"- **Targets**: {', '.join(f'`{t}`' for t in targets[:25])}{' …' if len(targets) > 25 else ''}")
            lines.append("")

        lines.append("## Plan steps")
        lines.append("")
        if steps:
            for idx, step in enumerate(steps, start=1):
                if isinstance(step, dict):
                    desc = (step.get("description") or "").strip() or "(missing step description)"
                    lines.append(f"{idx}. {desc}")
                    targets = step.get("target_files") or []
                    if targets:
                        lines.append(f"   - Targets: {', '.join(f'`{t}`' for t in targets)}")
                    notes = (step.get("notes") or "").strip()
                    if notes:
                        lines.append(f"   - Notes: {notes}")
                else:
                    lines.append(f"{idx}. {str(step)}")
        else:
            lines.append("_No plan steps generated._")
        lines.append("")

        lines.append("## Risks")
        lines.append("")
        lines.append(_fmt_list([str(r) for r in risks if str(r).strip()]))
        lines.append("")

        lines.append("## Refactor suggestions")
        lines.append("")
        lines.append(_fmt_list([str(r) for r in refactors if str(r).strip()]))
        lines.append("")

        lines.append("## Boundary specs")
        lines.append("")
        if specs:
            for spec in specs:
                name = (spec.get("boundary_name") or "Unknown").strip()
                status = (spec.get("status") or "").strip()
                human_description = (spec.get("human_description") or "").strip()
                diagram_text = (spec.get("diagram_text") or "").rstrip()
                machine_spec = spec.get("machine_spec") or {}
                actors = machine_spec.get("actors") or []
                interfaces = machine_spec.get("interfaces") or []
                invariants = machine_spec.get("invariants") or []
                plan_step = (spec.get("plan_step") or "").strip()

                lines.append(f"### {name}{f' ({status})' if status else ''}")
                lines.append("")

                if plan_step:
                    lines.append(f"- **Plan step**: {plan_step}")
                    lines.append("")

                if human_description:
                    lines.append("**Description**")
                    lines.append("")
                    lines.append(human_description)
                    lines.append("")

                if diagram_text:
                    lines.append("**Mermaid diagram**")
                    lines.append("")
                    lines.append("```mermaid")
                    lines.append(diagram_text)
                    lines.append("```")
                    lines.append("")

                lines.append("**Machine-readable contract**")
                lines.append("")
                lines.append("- Actors:")
                lines.extend([f"  - {actor}" for actor in actors] or ["  - (none)"])
                lines.append("- Interfaces:")
                lines.extend([f"  - {interface}" for interface in interfaces] or ["  - (none)"])
                lines.append("- Invariants:")
                lines.extend([f"  - {invariant}" for invariant in invariants] or ["  - (none)"])
                lines.append("")

                lines.append("**Implementation constraints (for Cursor/Claude)**")
                lines.append("")
                lines.append("- Preserve the actor boundaries above (do not fold responsibilities across services).")
                lines.append("- Ensure invariants hold across all happy-path and failure-path flows.")
                lines.append("- If an interface is renamed/expanded, update all call sites and configuration consistently.")
                lines.append("")
        else:
            lines.append("_None_")
        lines.append("")

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        task.metadata["plan_markdown_path"] = str(output_path)
        task.touch()
        self.store.upsert_task(task)
        return output_path

    def generate_patches(self, task_id: str, skip_rationale_enhancement: bool = False) -> Dict:
        """
        Generate patches for an approved plan.

        This should be called after plan approval.
        """
        import sys

        task = self._get_task(task_id)

        # Check if plan is approved
        if not task.metadata.get("plan_approved", False):
            raise ValueError("Plan must be approved before generating patches")

        # Retrieve plan and boundary specs from metadata
        context_summary = task.metadata.get("repository_summary", {})
        plan_preview = task.metadata.get("plan_preview", {})
        boundary_specs_data = task.metadata.get("boundary_specs", [])

        # Reconstruct plan object from stored data
        from ..domain.models import Plan, PlanStep
        plan = Plan(
            id=plan_preview.get("id", str(uuid4())),
            task_id=task_id,
            steps=[
                PlanStep.from_dict(step) if isinstance(step, dict) else PlanStep(description=str(step))
                for step in plan_preview.get("steps", [])
            ],
            risks=plan_preview.get("risks", []),
            refactor_suggestions=plan_preview.get("refactors", [])
        )

        # Reconstruct boundary specs from stored data
        from ..domain.models import BoundarySpecStatus
        specs = [
            BoundarySpec(
                id=spec_data["id"],
                task_id=spec_data.get("task_id", task_id),
                boundary_name=spec_data["boundary_name"],
                human_description=spec_data["human_description"],
                diagram_text=spec_data["diagram_text"],
                machine_spec=spec_data["machine_spec"],
                status=BoundarySpecStatus(spec_data["status"]),
                plan_step=spec_data.get("plan_step"),
            )
            for spec_data in boundary_specs_data
        ]

        sys.stderr.write("Generating patches...\n")
        try:
            patches = self.patch_engine.draft_patches(
                plan,
                repo_path=task.repo_path,
                boundary_specs=specs,
                skip_rationale_enhancement=skip_rationale_enhancement,
                repo_context=context_summary,
            )
            sys.stderr.write(f"Generated {len(patches)} patches\n")
        except Exception as exc:
            import traceback
            sys.stderr.write(f"Warning: Patch generation failed: {exc}\n")
            sys.stderr.write(f"Traceback:\n{traceback.format_exc()}\n")
            sys.stderr.write("Continuing without patches.\n")
            LOG.error("Patch generation failed", exc_info=True)
            patches = []  # Continue without patches

        # Generate test suggestions with patches (Epic 4.2)
        tests_skipped_reason: str | None = None
        if not bool(context_summary.get("has_tests", False)):
            tests_skipped_reason = "no tests detected in repository"
            sys.stderr.write(f"Skipping test suggestions ({tests_skipped_reason}).\n")
            tests = []
        else:
            sys.stderr.write("Generating test suggestions...\n")
            tests = self.test_suggester.suggest(
                plan=plan,
                patches=patches,
                boundary_specs=specs,
                repo_context=context_summary,
                repo_path=task.repo_path,
            )
            sys.stderr.write(f"Generated {len(tests)} test suggestions\n")

        # Store rationale history for each patch (Epic 4.1)
        rationale_history = task.metadata.get("rationale_history", [])
        for patch in patches:
            rationale_history.append({
                "patch_id": patch.id,
                "step_reference": patch.step_reference,
                "rationale": patch.rationale,
                "alternatives": patch.alternatives,
                "timestamp": patch.id,  # Using patch ID as timestamp proxy for now
            })
        task.metadata["rationale_history"] = rationale_history

        # Store patches and test suggestions in task metadata
        task.metadata["patch_queue"] = [patch.step_reference for patch in patches]
        task.metadata["patch_queue_state"] = [patch.to_dict() for patch in patches]
        task.metadata["test_suggestions"] = [suggestion.description for suggestion in tests]
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "PATCHES_GENERATED",
            {
                "patch_count": len(patches),
                "patch_queue": task.metadata["patch_queue"],
                "test_suggestions": task.metadata["test_suggestions"]
            }
        )

        return {
            "patch_count": len(patches),
            "patches": task.metadata["patch_queue"],
            "test_count": len(tests),
            "test_suggestions": task.metadata["test_suggestions"],
            "tests_skipped_reason": tests_skipped_reason,
        }

    # ------------------------------------------------------------------ Helpers
    def _get_task(self, task_id: str, *, _skip_description_guard: bool = False) -> Task:
        for task in self.store.load_tasks():
            if task.id == task_id:
                if _skip_description_guard:
                    return task
                return self._ensure_description_consistency(task)
        raise ValueError(f"Task not found: {task_id}")

    def _ensure_description_consistency(self, task: Task) -> Task:
        """
        Guardrail: if someone edits tasks.json and changes the description for an
        existing task, regenerate clarifications and clear stale metadata so we
        don't ask nonsense questions based on the old intent.
        """
        current = (task.description or "").strip()
        snapshot = (task.metadata.get("description_snapshot") or "").strip()

        if not current:
            return task

        if not snapshot:
            task.metadata["description_snapshot"] = current
            task.touch()
            self.store.upsert_task(task)
            return task

        if snapshot == current:
            return task

        # Only auto-reset if the task is still in early workflow stages.
        early_states = {
            TaskStatus.CREATED,
            TaskStatus.CLARIFYING,
            TaskStatus.PLANNING,
            TaskStatus.SPEC_PENDING,
        }
        reset = task.status in early_states

        return self.update_task_description(
            task.id,
            current,
            reason="Auto-detected description change (tasks.json edit)",
            reset_metadata=reset,
        )

    def _preserve_clarifications_history(self, task: Task, *, reason: str) -> None:
        history = task.metadata.get("clarifications_history") or []
        if not isinstance(history, list):
            history = []
        previous = task.metadata.get("clarifications", []) or []
        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "reason": (reason or "").strip(),
                "clarifications": previous,
            }
        )
        task.metadata["clarifications_history"] = history

    def _reset_task_artifacts(self, task: Task, *, reset_context_steps: bool = False) -> None:
        # Reset downstream artifacts so the next plan/specs/patches are generated
        # from the current description + clarifications.
        for key in (
            "plan_preview",
            "refactor_suggestions",
            "boundary_specs",
            "pending_specs",
            "patches",
            "patch_queue",
            "patch_queue_state",
            "test_suggestions",
        ):
            task.metadata.pop(key, None)

        # Preserve engineer-provided bounded scope across plan resets/rejections, but
        # drop auto-generated plan-target scoping (it can be regenerated).
        bounded = task.metadata.get("bounded_context")
        if isinstance(bounded, dict) and bounded:
            manual = bounded.get("manual")
            if manual:
                task.metadata["bounded_context"] = {"manual": manual}
            else:
                task.metadata.pop("bounded_context", None)
        else:
            task.metadata.pop("bounded_context", None)

        if reset_context_steps:
            task.metadata.pop("context_steps", None)

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
        return self._load_patch_queue(task)

    def get_next_pending_patch(self, task_id: str) -> Patch | None:
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
            subprocess.run(
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
            subprocess.run(
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
            subprocess.run(
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
    def _record_context_step(
        self,
        task: Task,
        trigger: str,
        included: Optional[List[Path]] = None,
        excluded: Optional[List[Path]] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        included_rel = self._relative_paths(task.repo_path, included or [])
        excluded_rel = self._relative_paths(task.repo_path, excluded or [])
        step = {
            "id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
            "note": note,
            "included": included_rel,
            "excluded": excluded_rel,
        }
        steps = task.metadata.setdefault("context_steps", [])
        steps.append(step)
        task.touch()
        self.store.upsert_task(task)
        self.logger.record(task.id, "CONTEXT_UPDATED", step)
        return step

    @staticmethod
    def _relative_paths(repo_path: Path, paths: List[Path]) -> List[str]:
        results: List[str] = []
        for path in paths:
            try:
                results.append(str(path.resolve().relative_to(repo_path)))
            except ValueError:
                results.append(str(path.resolve()))
        return results

    @staticmethod
    def _normalize_paths(repo_path: Path, raw_paths: List[str]) -> List[Path]:
        normalized: List[Path] = []
        for raw in raw_paths:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (repo_path / candidate).resolve()
            else:
                candidate = candidate.resolve()
            normalized.append(candidate)
        return normalized

    def _snapshot_worktree_status(self, task: Task) -> None:
        status = self._get_git_status(task.repo_path)
        task.metadata["worktree_status"] = status
        task.metadata["last_snapshot_commit"] = self._current_commit(task.repo_path)
        self.store.upsert_task(task)

    def has_manual_edits(self, task_id: str) -> bool:
        task = self._get_task(task_id)
        return self._detect_manual_edits(task)

    def acknowledge_manual_edits(self, task_id: str) -> None:
        task = self._get_task(task_id)
        self._snapshot_worktree_status(task)

    def _update_pending_specs(self, task: Task) -> None:
        specs = task.metadata.get("boundary_specs", [])
        task.metadata["pending_specs"] = [
            spec.get("boundary_name")
            for spec in specs
            if spec.get("status") == BoundarySpecStatus.PENDING.value
        ]

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

    def _get_current_branch(self, repo_path: Path) -> str:
        """Get the current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _get_remote_url(self, repo_path: Path) -> str:
        """Get the git remote URL (origin)."""
        try:
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _get_commit_author(self, repo_path: Path) -> str:
        """Get the author of the latest commit."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%an <%ae>"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _get_commit_message(self, repo_path: Path) -> str:
        """Get the message of the latest commit."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%s"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _get_commit_date(self, repo_path: Path) -> str:
        """Get the date of the latest commit."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%ci"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

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
