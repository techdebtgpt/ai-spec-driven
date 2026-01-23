from __future__ import annotations

import logging
import re
from pathlib import Path
from textwrap import dedent
from typing import List, Optional
from uuid import uuid4

from ...domain.models import BoundarySpec, Patch, PatchKind, Plan
from ..integrations.serena_client import SerenaToolClient, SerenaToolError
from ..review.rationale_enhancer import RationaleEnhancer


LOG = logging.getLogger(__name__)

EXTERNAL_PATCH_SENTINEL = "EXTERNAL_EDIT_REQUIRED"


class PatchEngine:
    """
    Breaks implementation work into incremental patch steps.
    """

    def __init__(
        self,
        serena_client: Optional[SerenaToolClient] = None,
        llm_client: Optional[object] = None,
        *,
        prefer_external_edits: bool = False,
    ) -> None:
        self.serena_client = serena_client
        self.prefer_external_edits = prefer_external_edits
        self.rationale_enhancer = RationaleEnhancer(llm_client=llm_client)

    def draft_patches(
        self, 
        plan: Plan, 
        repo_path: Path | None = None, 
        *,
        kind: PatchKind = PatchKind.IMPLEMENTATION,
        boundary_specs: List[BoundarySpec] | None = None,
        skip_rationale_enhancement: bool = False,
    ) -> List[Patch]:
        # If external edits are preferred, emit manual instructions even when Serena is available.
        if self.prefer_external_edits:
            return [
                self._placeholder_patch(plan, index, step.description, kind=kind)
                for index, step in enumerate(plan.steps, start=1)
            ]

        if self.serena_client:
            if repo_path is None:
                raise ValueError("repo_path is required when Serena integration is enabled.")
            return self._draft_with_serena(
                plan, 
                repo_path, 
                kind=kind, 
                boundary_specs=boundary_specs or [],
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
        boundary_specs: List[BoundarySpec],
        skip_rationale_enhancement: bool = False,
    ) -> List[Patch]:
        import sys
        patches: List[Patch] = []
        total_steps = len(plan.steps)
        for index, step in enumerate(plan.steps, start=1):
            sys.stderr.write(f"Generating patch {index}/{total_steps}: {step.description[:50]}...\n")
            try:
                # Find relevant boundary specs for this step
                relevant_specs = [
                    spec for spec in boundary_specs
                    if spec.status.value == "APPROVED"  # Only use approved specs
                ]
                proposal = self.serena_client.request_patch(
                    repo_path, 
                    step.description, 
                    plan.id,
                    boundary_specs=relevant_specs
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
                        boundary_specs=relevant_specs,
                    )
                except Exception as exc:
                    LOG.warning("Rationale enhancement failed for patch %s: %s", patch.id, exc)
                    # Continue with original rationale if enhancement fails
            
            patches.append(patch)
        return patches

    def _placeholder_patch(self, plan: Plan, index: int, description: str, *, kind: PatchKind) -> Patch:
        """
        When Serena is unavailable we still emit actionable guidance.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-") or "step"
        if self.prefer_external_edits:
            instructions = dedent(
                f"""\
                Step {index}: {description}

                Please implement this step in your editor (Cursor/Claude/IDE). You can use Spec Agent MCP tools in Cursor/Claude to fetch patch details if needed. After making the changes, run any relevant tests or checks, then sync the diff back so Spec Agent can capture it: `./spec-agent sync-external {plan.task_id} --patch-id <PATCH_ID>`
                """
            ).strip()
            diff = f"{EXTERNAL_PATCH_SENTINEL}\n{instructions}\n"
            rationale = (
                f"{kind.value.title()} patch {index}: external edit required."
            )
            alternatives = [
                "Apply the change externally, then run `./spec-agent sync-external <task-id> --patch-id <patch-id>`.",
                "If this step is obsolete, reject it from the patch queue.",
            ]
        else:
            filename = f".spec_agent/manual_steps/{plan.id[:8]}-{index:02d}-{slug[:32]}.md"
            instructions = dedent(
                f"""\
                # Manual patch {index}: {description}

                Please implement this step in your editor (Cursor/Claude/etc.). You can use Spec Agent MCP tools in Cursor/Claude to fetch patch details if needed. After making the changes, run the repo's relevant tests to verify behavior, then sync the diff back to Spec Agent: `./spec-agent sync-external {plan.task_id}`

                Include any notes about files touched or follow-up tasks inside this file if helpful.
                """
            )
            diff_body = "\n".join(f"+{line}" for line in instructions.splitlines())
            diff = f"--- /dev/null\n+++ b/{filename}\n@@\n{diff_body}\n"
            rationale = (
                f"{kind.value.title()} patch {index}: manual instructions generated."
            )
            alternatives = [
                "Use your editor to apply the change, then run `./spec-agent sync-external <task-id>`.",
                "Break the step into smaller manual commits if it spans multiple boundaries.",
            ]

        patch = Patch(
            id=str(uuid4()),
            task_id=plan.task_id,
            step_reference=description,
            diff=diff,
            rationale=rationale,
            alternatives=alternatives,
            kind=kind,
        )

        if diff.startswith(EXTERNAL_PATCH_SENTINEL):
            patch.rationale += (
                f"\n\nNext steps:\n"
                f"- Apply this step externally in your editor.\n"
                f"- Run the relevant tests/checks.\n"
                f"- Sync the result: `./spec-agent sync-external {plan.task_id} --patch-id {patch.id}`"
            )

        return patch
