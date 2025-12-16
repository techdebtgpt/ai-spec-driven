from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional
from uuid import uuid4

from ...domain.models import BoundarySpec, Plan, PlanStep

LOG = logging.getLogger(__name__)


class BoundaryManager:
    """
    Detects cross-boundary changes and drafts human + machine-readable specs.

    When an LLM client is provided, generates context-aware boundary specifications.
    Falls back to template-based specs when LLM is unavailable or fails.
    """

    def __init__(self, llm_client: Optional[object] = None, context_summary: Optional[Dict] = None) -> None:
        """
        Initialize BoundaryManager with optional LLM client and repo context.

        Args:
            llm_client: Optional OpenAILLMClient for AI-powered spec generation
            context_summary: Optional repository context for richer specs
        """
        self.llm_client = llm_client
        self.context_summary = context_summary or {}

    def required_specs(self, plan: Plan) -> List[BoundarySpec]:
        """
        Generate boundary specifications for plan steps that cross boundaries.

        Note: Boundary detection (determining WHICH steps cross boundaries) is done
        by section 2.3. This currently uses a
        simple keyword check as a placeholder.
        """
        boundary_specs: List[BoundarySpec] = []

        for step in plan.steps:
            # TODO: Replace with real boundary detection from section 2.3
            if self._is_boundary_crossing(step):
                boundary_specs.append(self.generate_spec_for_step(plan.task_id, step))

        return boundary_specs

    def generate_spec_for_step(self, task_id: str, step: PlanStep) -> BoundarySpec:
        """
        Generate (or regenerate) a boundary specification for a single plan step.
        """
        if self.llm_client:
            try:
                LOG.info("Generating boundary spec using LLM for step: %s", step.description)
                return self._generate_spec_with_llm(task_id, step)
            except Exception as exc:
                LOG.warning("LLM spec generation failed: %s, falling back to template", exc)
                return self._generate_spec_template(task_id, step)

        LOG.debug("No LLM client available, using template spec")
        return self._generate_spec_template(task_id, step)

    def _is_boundary_crossing(self, step: PlanStep) -> bool:
        """
        Placeholder for boundary detection (section 2.3).

        Currently uses simple keyword matching. Will be replaced with proper
        boundary detection logic by the team member assigned to section 2.3.
        """
        # Simple heuristic: look for keywords that suggest boundary crossings
        text = f"{step.description} {step.notes or ''}".lower()
        boundary_keywords = ["boundary", "api", "database", "service", "layer", "module", "component"]
        return any(keyword in text for keyword in boundary_keywords)

    def _generate_spec_with_llm(self, task_id: str, step: PlanStep) -> BoundarySpec:
        """
        Generate boundary specification using LLM analysis.
        """
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(step)

        # Call LLM with increased token limit for diagram generation
        response = self.llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1500
        )

        # Parse JSON response
        spec_data = self._parse_llm_response(response)

        # Build BoundarySpec object
        return BoundarySpec(
            id=str(uuid4()),
            task_id=task_id,
            boundary_name=spec_data.get("boundary_name", "UnnamedBoundary"),
            human_description=spec_data.get("human_description", ""),
            diagram_text=spec_data.get("diagram_text", ""),
            machine_spec=spec_data.get("machine_spec", {}),
            plan_step=step.description,
        )

    def _create_system_prompt(self) -> str:
        """
        Define the LLM's role for boundary specification generation.
        """
        return """You are a software architect helping define clear boundaries between system components.

When a code change crosses a boundary between modules/subsystems, you create a specification that defines:
1. **Human Description**: What this boundary represents and why it matters (2-3 sentences)
2. **Mermaid Diagram**: Visual representation of the interaction (sequenceDiagram or graph)
3. **Machine Spec**: Formal contract with actors, interfaces, and invariants

Your goal: Ensure engineers understand the architectural contract before implementing changes.

Return ONLY valid JSON in this format:
{
  "boundary_name": "Short name (e.g., 'API-Database', 'Frontend-Backend')",
  "human_description": "2-3 sentences explaining what this boundary controls and why it's important",
  "diagram_text": "Valid Mermaid diagram (sequenceDiagram or graph TD)",
  "machine_spec": {
    "actors": ["Component1", "Component2"],
    "interfaces": ["method or endpoint name"],
    "invariants": ["Rule that must never be broken"]
  }
}

Do not include any explanatory text before or after the JSON."""

    def _create_user_prompt(self, step: PlanStep) -> str:
        """
        Build the user prompt with plan step details and repository context.
        """
        # Extract context information
        languages = ", ".join(self.context_summary.get("top_languages", [])[:3]) or "unknown"
        modules = ", ".join(self.context_summary.get("top_modules", [])[:5]) or "not analyzed"

        return f"""Plan Step: {step.description}
Target Modules: {", ".join(step.target_files) if step.target_files else "not specified"}
Notes: {step.notes or "none"}

Repository Context:
- Languages: {languages}
- Key Modules: {modules}

This step crosses a boundary between system components.

Generate a boundary specification that defines:
1. What this boundary represents and why it matters
2. How the components should interact (Mermaid diagram showing the flow)
3. The formal contract (actors, interfaces, invariants)

Focus on making the boundary maintainable and loosely coupled."""

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

            spec_data = json.loads(response)

            # Validate required fields
            required_fields = ["boundary_name", "human_description", "diagram_text", "machine_spec"]
            for field in required_fields:
                if field not in spec_data:
                    raise ValueError(f"LLM response missing required field: {field}")

            return spec_data

        except json.JSONDecodeError as exc:
            LOG.error("Failed to parse LLM response as JSON: %s", exc)
            LOG.debug("LLM response was: %s", response)
            raise ValueError(f"Invalid JSON response from LLM: {exc}") from exc

    def _generate_spec_template(self, task_id: str, step: PlanStep) -> BoundarySpec:
        """
        Fallback template-based spec generation (original implementation).
        """
        return BoundarySpec(
            id=str(uuid4()),
            task_id=task_id,
            boundary_name="AutoDetectedBoundary",
            human_description=f"This boundary is related to: {step.description}. Defines responsibilities between system components to maintain separation of concerns.",
            diagram_text="""sequenceDiagram
    participant ComponentA
    participant ComponentB
    ComponentA->>ComponentB: request
    ComponentB-->>ComponentA: response
""",
            machine_spec={
                "name": "auto-detected-boundary",
                "actors": ["ComponentA", "ComponentB"],
                "interfaces": ["request", "response"],
                "invariants": [
                    "Components must communicate through defined interfaces only.",
                    "Changes to one component should not require changes to the other."
                ],
            },
            plan_step=step.description,
        )

