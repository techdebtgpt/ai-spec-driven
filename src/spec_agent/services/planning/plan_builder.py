from __future__ import annotations

import json
import logging
from typing import List, Optional
from uuid import uuid4

from ...domain.models import Plan, PlanStep

LOG = logging.getLogger(__name__)


class PlanBuilder:
    """
    Produces high-level implementation plans once clarifications are resolved.

    When an LLM client is provided, generates plans using AI analysis.
    Falls back to template-based plans when LLM is unavailable or fails.
    """

    def __init__(self, llm_client: Optional[object] = None) -> None:
        """
        Initialize PlanBuilder with optional LLM client.

        Args:
            llm_client: Optional OpenAILLMClient for AI-powered plan generation
        """
        self.llm_client = llm_client

    def build_plan(self, task_id: str, description: str, context_summary: dict) -> Plan:
        """
        Generate an implementation plan for the given task.

        Attempts LLM-based generation first if available, with fallback to template.
        """
        if self.llm_client:
            try:
                LOG.info("Generating plan using LLM for task %s", task_id)
                return self._build_plan_with_llm(task_id, description, context_summary)
            except Exception as exc:
                LOG.warning("LLM plan generation failed: %s, falling back to template", exc)
                return self._build_plan_template(task_id, description, context_summary)
        else:
            LOG.debug("No LLM client available, using template plan")
            return self._build_plan_template(task_id, description, context_summary)

    def _build_plan_with_llm(self, task_id: str, description: str, context_summary: dict) -> Plan:
        """
        Generate plan using LLM analysis.
        """
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(description, context_summary)

        # Call LLM with increased token limit for plan generation
        response = self.llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2000
        )

        # Parse JSON response
        plan_data = self._parse_llm_response(response)

        # Build Plan object from parsed data
        return self._create_plan_from_data(task_id, plan_data)

    def _create_system_prompt(self) -> str:
        """
        Define the LLM's role and output format.
        """
        return """You are a software architect helping senior engineers plan code changes in legacy codebases.

Your task is to analyze a change request and generate a high-level implementation plan.

The plan should include:
1. **Steps**: 3-7 high-level implementation steps, each identifying affected modules/components
2. **Risks**: Potential issues or concerns (technical debt, breaking changes, performance, etc.)
3. **Refactorings**: Opportunities to improve code quality while making changes

Focus on architectural and component-level changes, not specific file paths.
If the repository has no existing tests (or no test harness is detected) and the change request
does not explicitly require adding tests, do NOT add plan steps about creating tests.
Return ONLY valid JSON in this exact format:

{
  "steps": [
    {
      "description": "Clear, actionable step description",
      "target_modules": ["Module or component name"],
      "notes": "Optional implementation notes or warnings"
    }
  ],
  "risks": ["Risk description"],
  "refactor_suggestions": ["Refactoring opportunity"]
}

Do not include any explanatory text before or after the JSON."""

    def _create_user_prompt(self, description: str, context_summary: dict) -> str:
        """
        Build the user prompt with task details and repository context.
        """
        # Extract top languages
        top_languages = context_summary.get("top_languages", [])
        languages_str = ", ".join(top_languages) if top_languages else "unknown"

        # Extract top modules
        top_modules = context_summary.get("top_modules", [])
        modules_str = ", ".join(top_modules[:5]) if top_modules else "not analyzed"

        has_tests = bool(context_summary.get("has_tests", False))
        tests_str = "present" if has_tests else "not detected"

        # Extract top hotspots (large files)
        hotspots = context_summary.get("hotspots", [])
        hotspot_names = [h.get("path", "") for h in hotspots[:5]]
        hotspots_str = ", ".join(hotspot_names) if hotspot_names else "none detected"

        scoped = context_summary.get("scoped_context") or {}
        scoped_hint = ""
        total_files_line = f"- Total Files: {context_summary.get('file_count', 'unknown')}"
        if isinstance(scoped, dict) and scoped:
            agg = scoped.get("aggregate") or {}
            scoped_files = agg.get("file_count")
            if isinstance(scoped_files, int):
                repo_files = context_summary.get("file_count", "unknown")
                total_files_line = f"- Total Files (scoped): {scoped_files} (repo: {repo_files})"

            impact = scoped.get("impact") or {}
            targets = list((scoped.get("targets") or {}).keys())
            top_dirs = impact.get("top_directories") or []
            namespaces = impact.get("namespaces") or []
            scoped_hint = "\n\nScoped Context (post-clarifications):\n"
            if targets:
                scoped_hint += f"- Targets: {', '.join(targets[:8])}{' ...' if len(targets) > 8 else ''}\n"
            if top_dirs:
                scoped_hint += f"- Impacted directories: {', '.join(top_dirs[:10])}\n"
            if namespaces:
                scoped_hint += f"- Impacted namespaces: {', '.join(namespaces[:10])}\n"

        return f"""Change Request: {description}

Repository Context:
- {total_files_line}
- Main Languages: {languages_str}
- Tests: {tests_str}
- Legacy Hotspots (large files): {hotspots_str}
- Key Modules: {modules_str}
{scoped_hint}

Guidance:
- If tests are not detected and the change request does not explicitly ask to add tests, omit test-related steps and suggestions.

Generate a high-level implementation plan for this change request."""

    def _parse_llm_response(self, response: str) -> dict:
        """
        Parse and validate the LLM's JSON response.
        """
        try:
            # Try to extract JSON if LLM included extra text
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                if response.startswith("json"):
                    response = response[4:].strip()

            plan_data = json.loads(response)

            # Validate required fields
            if "steps" not in plan_data:
                raise ValueError("LLM response missing 'steps' field")
            if not isinstance(plan_data["steps"], list):
                raise ValueError("'steps' must be a list")

            return plan_data

        except json.JSONDecodeError as exc:
            LOG.error("Failed to parse LLM response as JSON: %s", exc)
            LOG.debug("LLM response was: %s", response)
            raise ValueError(f"Invalid JSON response from LLM: {exc}") from exc

    def _create_plan_from_data(self, task_id: str, plan_data: dict) -> Plan:
        """
        Convert parsed LLM data into a Plan domain model.
        """
        # Build PlanStep objects from steps data
        steps: List[PlanStep] = []
        for step_data in plan_data.get("steps", []):
            step = PlanStep(
                description=step_data.get("description", ""),
                target_files=step_data.get("target_modules", []),
                notes=step_data.get("notes", "")
            )
            steps.append(step)

        # Extract risks and refactor suggestions
        risks = plan_data.get("risks", [])
        refactor_suggestions = plan_data.get("refactor_suggestions", [])

        return Plan(
            id=str(uuid4()),
            task_id=task_id,
            steps=steps,
            risks=risks,
            refactor_suggestions=refactor_suggestions,
        )

    def _build_plan_template(self, task_id: str, description: str, context_summary: dict) -> Plan:
        """
        Fallback template-based plan generation (original implementation).
        """
        steps: List[PlanStep] = [
            PlanStep(
                description="Index relevant directories and update dependency graph cache.",
                target_files=context_summary.get("top_modules", []),
            ),
            PlanStep(
                description="Draft boundary-aware implementation steps based on clarified scope.",
                notes="Ensure no plan step crosses subsystems without spec approval.",
            ),
            PlanStep(
                description="Prepare incremental patches (<30 LOC each) and rationale outlines.",
            ),
        ]

        risks = [
            "Repository size may exceed configured indexing threshold."
            if context_summary.get("file_count", 0) > 50_000
            else "Legacy hotspots need manual review before editing.",
            "Boundary contracts might be missing or outdated.",
        ]

        return Plan(
            id=str(uuid4()),
            task_id=task_id,
            steps=steps,
            risks=risks,
            refactor_suggestions=["Extract helper modules when a patch grows beyond the LOC budget."],
        )


