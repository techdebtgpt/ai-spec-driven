from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from ...domain.models import Patch, Plan
from ..integrations.serena_client import SerenaToolClient, SerenaToolError


LOG = logging.getLogger(__name__)


class PatchEngine:
    """
    Breaks implementation work into incremental patch steps.
    """

    def __init__(self, serena_client: Optional[SerenaToolClient] = None) -> None:
        self.serena_client = serena_client

    def draft_patches(self, plan: Plan, repo_path: Path | None = None) -> List[Patch]:
        if self.serena_client:
            if repo_path is None:
                raise ValueError("repo_path is required when Serena integration is enabled.")
            return self._draft_with_serena(plan, repo_path)

        return [self._placeholder_patch(plan, index, step.description) for index, step in enumerate(plan.steps, start=1)]

    def _draft_with_serena(self, plan: Plan, repo_path: Path) -> List[Patch]:
        patches: List[Patch] = []
        for index, step in enumerate(plan.steps, start=1):
            try:
                proposal = self.serena_client.request_patch(repo_path, step.description, plan.id)
            except SerenaToolError as exc:
                LOG.warning("Serena patch generation failed for '%s'. Falling back to placeholder. (%s)", step.description, exc)
                patches.append(self._placeholder_patch(plan, index, step.description))
                continue

            patches.append(
                Patch(
                    id=str(uuid4()),
                    task_id=plan.task_id,
                    step_reference=step.description,
                    diff=proposal.diff,
                    rationale=proposal.rationale,
                    alternatives=proposal.alternatives,
                )
            )
        return patches

    @staticmethod
    def _placeholder_patch(plan: Plan, index: int, description: str) -> Patch:
        diff = f"--- step-{index}.txt\n+++ step-{index}.txt\n@@\n- placeholder\n+ implementation details TBD\n"
        rationale = f"Implements plan step {index}: {description}"
        alternatives = [
            "Manual refactor before applying patch.",
            "Defer change until boundary spec is approved.",
        ]
        return Patch(
            id=str(uuid4()),
            task_id=plan.task_id,
            step_reference=description,
            diff=diff,
            rationale=rationale,
            alternatives=alternatives,
        )


