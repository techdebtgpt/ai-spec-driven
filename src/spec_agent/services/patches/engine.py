from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from ...domain.models import BoundarySpec, Patch, PatchKind, Plan, PlanStep
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
        self.llm_client = llm_client
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
        if self.llm_client:
            if repo_path is None:
                raise ValueError("repo_path is required for LLM-based patch generation.")
            return self._draft_with_llm(
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

    def _draft_with_llm(
        self,
        plan: Plan,
        repo_path: Path,
        *,
        kind: PatchKind,
        boundary_specs: List[BoundarySpec],
        skip_rationale_enhancement: bool = False,
    ) -> List[Patch]:
        """
        Generate patches using direct LLM calls.

        This method uses the LLM to analyze the plan steps and generate
        unified diffs for each implementation step.
        """
        import sys
        patches: List[Patch] = []
        total_steps = len(plan.steps)

        for index, step in enumerate(plan.steps, start=1):
            sys.stderr.write(f"Generating patch {index}/{total_steps} with LLM: {step.description[:50]}...\n")

            try:
                # Build context for LLM
                boundary_context = self._build_boundary_context(boundary_specs)
                plan_context = self._build_plan_context(plan, step)

                # Generate diff using LLM
                diff, rationale, alternatives = self._generate_diff_with_llm(
                    step=step,
                    plan_context=plan_context,
                    boundary_context=boundary_context,
                    repo_path=repo_path,
                    kind=kind,
                )

                # Validate diff
                if not diff or diff.strip() == "":
                    LOG.warning("LLM returned empty diff for step '%s', using placeholder", step.description)
                    patch = self._placeholder_patch(plan, index, step.description, kind=kind)
                    patches.append(patch)
                    continue

                # Create patch
                patch = Patch(
                    id=str(uuid4()),
                    task_id=plan.task_id,
                    step_reference=step.description,
                    diff=diff,
                    rationale=rationale,
                    alternatives=alternatives,
                    kind=kind,
                )

                # Enhance rationale if requested
                if not skip_rationale_enhancement:
                    try:
                        sys.stderr.write(f"  Enhancing rationale for patch {index}...\n")
                        relevant_specs = [s for s in boundary_specs if s.status.value == "APPROVED"]
                        patch = self.rationale_enhancer.enhance_rationale(
                            patch=patch,
                            plan_step=step,
                            plan=plan,
                            boundary_specs=relevant_specs,
                        )
                    except Exception as exc:
                        LOG.warning("Rationale enhancement failed for patch %s: %s", patch.id, exc)

                patches.append(patch)

            except Exception as exc:
                LOG.error("LLM patch generation failed for '%s': %s", step.description, exc)
                # Fallback to placeholder on error
                patch = self._placeholder_patch(plan, index, step.description, kind=kind)
                patch.rationale += f"\n\n[Error during LLM generation: {exc}]"
                patches.append(patch)

        return patches

    def _build_boundary_context(self, boundary_specs: List[BoundarySpec]) -> str:
        """Build context string from boundary specifications."""
        if not boundary_specs:
            return ""

        context = "\n\nBoundary Specifications:\n"
        for spec in boundary_specs:
            context += f"- {spec.boundary_name}: {spec.human_description}\n"
            if spec.machine_spec:
                actors = spec.machine_spec.get("actors", [])
                interfaces = spec.machine_spec.get("interfaces", [])
                invariants = spec.machine_spec.get("invariants", [])
                if actors:
                    context += f"  Actors: {', '.join(actors)}\n"
                if interfaces:
                    context += f"  Interfaces: {', '.join(interfaces)}\n"
                if invariants:
                    context += f"  Invariants: {', '.join(invariants[:3])}\n"
        return context

    def _build_plan_context(self, plan: Plan, step: PlanStep) -> str:
        """Build context string from plan information."""
        context = f"Task: {plan.task_id}\n"
        context += f"Current Step: {step.description}\n"
        if step.notes:
            context += f"Notes: {step.notes}\n"
        if step.target_files:
            context += f"Target Files: {', '.join(step.target_files)}\n"
        if plan.risks:
            context += f"\nRisks: {', '.join(plan.risks[:3])}\n"
        return context

    def _generate_diff_with_llm(
        self,
        step: PlanStep,
        plan_context: str,
        boundary_context: str,
        repo_path: Path,
        kind: PatchKind,
    ) -> tuple[str, str, List[str]]:
        """
        Use LLM to generate a unified diff for the given plan step.

        Returns:
            (diff, rationale, alternatives)
        """
        # Build the prompt
        prompt = f"""You are a senior software engineer implementing a code change.

{plan_context}{boundary_context}

Repository path: {repo_path}

Your task is to generate a unified diff (git diff format) for this implementation step:
{step.description}

Guidelines:
1. Generate a valid unified diff in git format
2. Start with `--- a/path/to/file` and `+++ b/path/to/file`
3. Include proper hunk headers `@@ -start,count +start,count @@`
4. Use `-` for removed lines and `+` for added lines
5. Include context lines (unchanged lines) around changes
6. Make focused, minimal changes that accomplish the step
7. Follow best practices for the language/framework being used
8. Consider error handling and edge cases
9. Respect any boundary specifications and constraints mentioned above

Respond with JSON in this exact format:
{{
  "diff": "the complete unified diff here",
  "rationale": "1-2 sentences explaining why this change is needed and what it does",
  "alternatives": ["Alternative approach 1", "Alternative approach 2"]
}}

Return ONLY valid JSON, no markdown formatting, no code blocks, no extra text."""

        try:
            import json

            response_text = self.llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior software engineer. Generate code changes as unified diffs. "
                            "Return only valid JSON with diff, rationale, and alternatives fields."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_output_tokens=2000,
            )

            # Clean up JSON if wrapped in markdown
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = '\n'.join(lines)

            # Parse JSON response
            data = json.loads(cleaned)
            diff = data.get("diff", "")
            rationale = data.get("rationale", f"{kind.value.title()} for: {step.description}")
            alternatives = data.get("alternatives", [
                "Manual implementation with different approach",
                "Defer this change to a later iteration",
            ])

            return diff, rationale, alternatives

        except Exception as exc:
            LOG.error("Failed to generate diff with LLM: %s", exc)
            raise

    @staticmethod
    def _placeholder_patch(plan: Plan, index: int, description: str, *, kind: PatchKind) -> Patch:
        diff = f"--- step-{index}.txt\n+++ step-{index}.txt\n@@\n- placeholder\n+ implementation details TBD\n"
        rationale = f"{kind.value.title()} patch {index}: {description}"
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
            kind=kind,
        )


