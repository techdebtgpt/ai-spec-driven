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
        repo_context: Optional[dict] = None,
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
                repo_context=repo_context,
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
        repo_context: Optional[dict] = None,
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

                # Read target files from repository
                file_contents = self._read_target_files(repo_path, step.target_files)

                # Generate diff using LLM
                diff, rationale, alternatives = self._generate_diff_with_llm(
                    step=step,
                    plan_context=plan_context,
                    boundary_context=boundary_context,
                    repo_path=repo_path,
                    kind=kind,
                    file_contents=file_contents,
                    repo_context=repo_context,
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

    def _read_target_files(self, repo_path: Path, target_files: List[str]) -> dict[str, str]:
        """
        Read target files from the repository.

        Returns a dict mapping file paths to their contents.
        """
        file_contents = {}

        for target in target_files:
            if not target:
                continue

            # Try to find the file (handle various naming conventions)
            potential_paths = [
                repo_path / target,
                repo_path / f"{target}.java",
                repo_path / f"{target}.py",
                repo_path / f"{target}.ts",
                repo_path / f"{target}.js",
            ]

            # Also try searching for the file in common directories
            for ext in ['.java', '.py', '.ts', '.js', '.go', '.rs']:
                potential_paths.append(repo_path / "src" / "main" / "java" / f"{target.replace('.', '/')}{ext}")
                potential_paths.append(repo_path / "src" / f"{target.replace('.', '/')}{ext}")

            found_file = None
            for path in potential_paths:
                if path.exists() and path.is_file():
                    found_file = path
                    break

            if found_file:
                try:
                    content = found_file.read_text(encoding='utf-8', errors='ignore')
                    # Limit file size to avoid token limits
                    if len(content) > 10000:
                        content = content[:10000] + "\n... (file truncated)"
                    file_contents[target] = content
                    LOG.debug(f"Read {len(content)} chars from {found_file}")
                except Exception as e:
                    LOG.warning(f"Failed to read {found_file}: {e}")
                    file_contents[target] = f"[Could not read file: {e}]"
            else:
                LOG.info(f"Target file '{target}' not found - will create new file")
                file_contents[target] = "[New file - no existing content]"

        return file_contents

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
        file_contents: dict[str, str],
        repo_context: Optional[dict] = None,
    ) -> tuple[str, str, List[str]]:
        """
        Use LLM to generate a unified diff for the given plan step.

        Returns:
            (diff, rationale, alternatives)
        """
        # Build repo context section
        repo_info = ""
        if repo_context:
            languages = repo_context.get("top_languages", [])
            frameworks = repo_context.get("frameworks", [])
            if languages:
                repo_info += f"\nRepository language(s): {', '.join(str(l) for l in languages[:3])}"
            if frameworks:
                repo_info += f"\nFrameworks detected: {', '.join(frameworks[:5])}"

        # Build file contents section
        files_section = ""
        if file_contents:
            files_section = "\n\nCurrent file contents:\n"
            for filename, content in file_contents.items():
                files_section += f"\n--- File: {filename} ---\n"
                files_section += f"{content}\n"
                files_section += f"--- End of {filename} ---\n"

        # Build the prompt
        prompt = f"""You are a senior software engineer implementing a code change.

{plan_context}{boundary_context}{repo_info}

Repository path: {repo_path}

Your task is to generate a unified diff (git diff format) for this implementation step:
{step.description}
{files_section}

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
10. If modifying an existing file, base your changes on the current file content shown above
11. For new files (marked as "[New file - no existing content]"), create the entire file

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


