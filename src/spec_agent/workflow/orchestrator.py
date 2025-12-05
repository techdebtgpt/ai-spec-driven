from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from ..config.settings import AgentSettings, get_settings
from ..domain.models import BoundarySpec, Patch, Plan, Task, TaskStatus, TestSuggestion
from ..persistence.store import JsonStore
from ..services.context.indexer import ContextIndexer
from ..services.context.retriever import ContextRetriever
from ..services.integrations.serena_client import SerenaToolClient
from ..services.llm.openai_client import OpenAILLMClient
from ..services.planning.clarifier import Clarifier
from ..services.planning.plan_builder import PlanBuilder
from ..services.patches.engine import PatchEngine
from ..services.specs.boundary_manager import BoundaryManager
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

        patches = self.patch_engine.draft_patches(plan, repo_path=task.repo_path)
        tests = self.test_suggester.suggest(plan)

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
        task.metadata["serena_patches"] = [
            {
                "step_reference": patch.step_reference,
                "diff": patch.diff,
                "rationale": patch.rationale,
                "alternatives": patch.alternatives,
            }
            for patch in patches
        ]
        task.metadata["test_suggestions"] = [suggestion.description for suggestion in tests]
        task.status = TaskStatus.SPEC_PENDING if specs else TaskStatus.PLANNING
        task.touch()
        self.store.upsert_task(task)

        self.logger.record(
            task.id,
            "PLAN_GENERATED",
            {
                "plan": task.metadata["plan_preview"],
                "pending_specs": task.metadata["pending_specs"],
                "patch_queue": task.metadata["patch_queue"],
                "test_suggestions": task.metadata["test_suggestions"],
            },
        )

        return {
            "plan": task.metadata["plan_preview"],
            "pending_specs": task.metadata["pending_specs"],
            "patch_queue": task.metadata["patch_queue"],
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


