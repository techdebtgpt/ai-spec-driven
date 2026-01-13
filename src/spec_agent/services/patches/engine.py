from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from ...domain.models import Patch, PatchKind, Plan
from ..integrations.serena_client import SerenaToolClient, SerenaToolError
from ..review.rationale_enhancer import RationaleEnhancer


LOG = logging.getLogger(__name__)


class PatchEngine:
    """
    Breaks implementation work into incremental patch steps.
    """

    def __init__(
        self, 
        serena_client: Optional[SerenaToolClient] = None,
        llm_client: Optional[object] = None,
    ) -> None:
        self.serena_client = serena_client
        self.rationale_enhancer = RationaleEnhancer(llm_client=llm_client)

    def draft_patches(
        self,
        plan: Plan,
        repo_path: Path | None = None,
        *,
        kind: PatchKind = PatchKind.IMPLEMENTATION,
        skip_rationale_enhancement: bool = False,
    ) -> List[Patch]:
        if self.serena_client:
            if repo_path is None:
                raise ValueError("repo_path is required when Serena integration is enabled.")
            return self._draft_with_serena(
                plan,
                repo_path,
                kind=kind,
                skip_rationale_enhancement=skip_rationale_enhancement,
            )

        return [
            self._placeholder_patch(plan, index, step.description, kind=kind)
            for index, step in enumerate(plan.steps, start=1)
        ]

    def _draft_with_serena(
        self,
        plan: Plan,
        repo_path: Path,
        *,
        kind: PatchKind,
        skip_rationale_enhancement: bool = False,
    ) -> List[Patch]:
        import sys
        patches: List[Patch] = []
        total_steps = len(plan.steps)
        for index, step in enumerate(plan.steps, start=1):
            sys.stderr.write(f"Generating patch {index}/{total_steps}: {step.description[:50]}...\n")
            try:
                proposal = self.serena_client.request_patch(
                    repo_path,
                    step.description,
                    plan.id
                )
                
                # Validate that we got a real diff (not empty or error)
                if not proposal.diff or proposal.diff.strip() == "":
                    raise SerenaToolError(
                        f"Serena returned empty diff for step '{step.description}'. "
                        f"Rationale: {proposal.rationale}"
                    )
                
                # Check if rationale indicates an error (but allow connection success messages)
                rationale_lower = proposal.rationale.lower()
                if (proposal.rationale.startswith(("Error", "Failed", "Serena integration failed", "Serena MCP integration failed")) 
                    and "connected successfully" not in rationale_lower):
                    raise SerenaToolError(
                        f"Serena integration error for step '{step.description}': {proposal.rationale}"
                    )
                
            except SerenaToolError as exc:
                LOG.error("Serena patch generation failed for '%s': %s", step.description, exc)
                raise  # Fail fast - no fallback to placeholder

            # Create initial patch
            patch = Patch(
                id=str(uuid4()),
                task_id=plan.task_id,
                step_reference=step.description,
                diff=proposal.diff,
                rationale=proposal.rationale,
                alternatives=proposal.alternatives,
                kind=kind,
            )
            
            # Enhance rationale with design decisions, trade-offs, and constraints (Epic 4.1)
            if not skip_rationale_enhancement:
                try:
                    import sys
                    sys.stderr.write(f"  Enhancing rationale for patch {index}...\n")
                    patch = self.rationale_enhancer.enhance_rationale(
                        patch=patch,
                        plan_step=step,
                        plan=plan,
                    )
                except Exception as exc:
                    LOG.warning("Rationale enhancement failed for patch %s: %s", patch.id, exc)
                    # Continue with original rationale if enhancement fails
            
            patches.append(patch)
        return patches

    @staticmethod
    def _placeholder_patch(plan: Plan, index: int, description: str, *, kind: PatchKind) -> Patch:
        diff = f"--- step-{index}.txt\n+++ step-{index}.txt\n@@\n- placeholder\n+ implementation details TBD\n"
        rationale = f"{kind.value.title()} patch {index}: {description}"
        alternatives = [
            "Manual refactor before applying patch.",
            "Defer change until after plan review.",
        ]
        return Patch(
            id=str(uuid4()),
            task_id=plan.task_id,
            step_reference=description,
            diff=diff,
            rationale=rationale,
            alternatives=alternatives,
            kind=kind,
        )


