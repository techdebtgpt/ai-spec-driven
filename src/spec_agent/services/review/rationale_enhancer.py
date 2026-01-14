from __future__ import annotations

import logging
from typing import List, Optional

from ...domain.models import BoundarySpec, Patch, Plan, PlanStep

LOG = logging.getLogger(__name__)


class RationaleEnhancer:
    """
    Enhances patch rationales with design decisions, trade-offs, constraints, and alternatives.
    
    Epic 4.1: Rationale-Based Code Review
    """

    def __init__(self, llm_client: Optional[object] = None) -> None:
        """
        Initialize RationaleEnhancer with optional LLM client.
        
        Args:
            llm_client: Optional OpenAILLMClient for AI-powered rationale enhancement
        """
        self.llm_client = llm_client

    def enhance_rationale(
        self,
        patch: Patch,
        plan_step: PlanStep,
        plan: Plan,
        boundary_specs: List[BoundarySpec],
        repo_context: Optional[dict] = None,
    ) -> Patch:
        """
        Enhance a patch's rationale with design decisions, trade-offs, and constraints.
        
        Args:
            patch: The patch to enhance
            plan_step: The plan step this patch implements
            plan: The full plan context
            boundary_specs: Relevant boundary specifications
            repo_context: Optional repository context summary
            
        Returns:
            Enhanced patch with improved rationale and alternatives
        """
        if self.llm_client:
            try:
                LOG.info("Enhancing rationale using LLM for patch %s", patch.id)
                return self._enhance_with_llm(patch, plan_step, plan, boundary_specs, repo_context)
            except Exception as exc:
                LOG.warning("LLM rationale enhancement failed: %s, using template", exc)
                return self._enhance_template(patch, plan_step, plan, boundary_specs)
        else:
            LOG.debug("No LLM client available, using template rationale")
            return self._enhance_template(patch, plan_step, plan, boundary_specs)

    def _enhance_with_llm(
        self,
        patch: Patch,
        plan_step: PlanStep,
        plan: Plan,
        boundary_specs: List[BoundarySpec],
        repo_context: Optional[dict],
    ) -> Patch:
        """Enhance rationale using LLM analysis."""
        # Build context for LLM
        boundary_context = ""
        if boundary_specs:
            boundary_context = "\n\nBoundary Specifications:\n"
            for spec in boundary_specs:
                boundary_context += f"- {spec.boundary_name}: {spec.human_description}\n"
                if spec.machine_spec:
                    actors = spec.machine_spec.get("actors", [])
                    interfaces = spec.machine_spec.get("interfaces", [])
                    invariants = spec.machine_spec.get("invariants", [])
                    if actors:
                        boundary_context += f"  Actors: {', '.join(actors)}\n"
                    if interfaces:
                        boundary_context += f"  Interfaces: {', '.join(interfaces)}\n"
                    if invariants:
                        boundary_context += f"  Invariants: {', '.join(invariants[:3])}\n"

        plan_context = f"Plan: {plan.task_id}\n"
        plan_context += f"Step: {plan_step.description}\n"
        if plan_step.notes:
            plan_context += f"Notes: {plan_step.notes}\n"
        if plan.risks:
            plan_context += f"Risks: {', '.join(plan.risks[:3])}\n"

        repo_info = ""
        if repo_context:
            repo_info = "\nRepository Context:\n"
            repo_info += f"- Primary language: {repo_context.get('primary_language', 'unknown')}\n"
            if repo_context.get("modules"):
                repo_info += f"- Modules: {', '.join(repo_context.get('modules', [])[:5])}\n"

        prompt = f"""Analyze this code change and provide a comprehensive rationale.

{plan_context}{boundary_context}{repo_info}

Code Change (unified diff):
```
{patch.diff[:2000]}
```

Current Rationale:
{patch.rationale}

Provide an enhanced rationale that includes:

1. **Why this change is needed**: Explain the purpose and motivation
2. **Design decisions**: What design choices were made and why
3. **Trade-offs**: What trade-offs were considered (performance vs maintainability, simplicity vs flexibility, etc.)
4. **Constraints**: What constraints influenced the design (boundary specs, existing patterns, technical limitations)
5. **Alternatives considered**: List at least 1-2 alternative approaches that were considered and why they weren't chosen

Format your response as JSON:
{{
  "rationale": "Enhanced rationale with all sections above",
  "alternatives": ["Alternative 1: ...", "Alternative 2: ..."],
  "design_decisions": ["Decision 1: ...", "Decision 2: ..."],
  "trade_offs": ["Trade-off 1: ...", "Trade-off 2: ..."],
  "constraints": ["Constraint 1: ...", "Constraint 2: ..."]
}}

Return only valid JSON, no markdown, no explanations."""

        try:
            import json

            response_text = self.llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior software engineer providing code review rationale. "
                            "Analyze code changes and explain design decisions, trade-offs, and alternatives. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_output_tokens=800,
            )
            
            # Clean up JSON if wrapped in markdown
            if response_text.strip().startswith("```"):
                lines = response_text.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = '\n'.join(lines)

            enhanced_data = json.loads(response_text.strip())

            # Create enhanced patch
            enhanced_rationale = enhanced_data.get("rationale", patch.rationale)
            if not isinstance(enhanced_rationale, str):
                enhanced_rationale = str(enhanced_rationale)
            enhanced_alternatives = enhanced_data.get("alternatives", patch.alternatives)
            if not isinstance(enhanced_alternatives, list):
                enhanced_alternatives = [
                    str(enhanced_alternatives)
                ] if enhanced_alternatives is not None else patch.alternatives

            # Store additional metadata in a structured format
            enhanced_metadata = {
                "design_decisions": [
                    str(item) for item in enhanced_data.get("design_decisions", []) or []
                ],
                "trade_offs": [
                    str(item) for item in enhanced_data.get("trade_offs", []) or []
                ],
                "constraints": [
                    str(item) for item in enhanced_data.get("constraints", []) or []
                ],
                "original_rationale": patch.rationale,
            }

            # Create new patch with enhanced rationale
            enhanced_patch = Patch(
                id=patch.id,
                task_id=patch.task_id,
                step_reference=patch.step_reference,
                diff=patch.diff,
                rationale=enhanced_rationale,
                alternatives=enhanced_alternatives,
                status=patch.status,
                kind=patch.kind,
            )
            
            # Store metadata (we'll need to extend Patch model or store separately)
            # For now, include in rationale
            if enhanced_metadata["design_decisions"]:
                enhanced_patch.rationale += "\n\n**Design Decisions:**\n" + "\n".join(f"- {d}" for d in enhanced_metadata["design_decisions"])
            if enhanced_metadata["trade_offs"]:
                enhanced_patch.rationale += "\n\n**Trade-offs:**\n" + "\n".join(f"- {t}" for t in enhanced_metadata["trade_offs"])
            if enhanced_metadata["constraints"]:
                enhanced_patch.rationale += "\n\n**Constraints:**\n" + "\n".join(f"- {c}" for c in enhanced_metadata["constraints"])

            return enhanced_patch

        except Exception as exc:
            LOG.error("Failed to enhance rationale with LLM: %s", exc)
            return self._enhance_template(patch, plan_step, plan, boundary_specs)

    def _enhance_template(
        self,
        patch: Patch,
        plan_step: PlanStep,
        plan: Plan,
        boundary_specs: List[BoundarySpec],
    ) -> Patch:
        """Enhance rationale using template-based approach."""
        # Build template rationale
        rationale_parts = [patch.rationale]
        
        rationale_parts.append("\n\n**Why this change is needed:**")
        rationale_parts.append(f"This change implements the plan step: {plan_step.description}")
        if plan_step.notes:
            rationale_parts.append(f"Additional context: {plan_step.notes}")
        
        if boundary_specs:
            rationale_parts.append("\n**Constraints from boundary specifications:**")
            for spec in boundary_specs:
                rationale_parts.append(f"- {spec.boundary_name}: Must respect {spec.human_description}")
                if spec.machine_spec:
                    invariants = spec.machine_spec.get("invariants", [])
                    if invariants:
                        rationale_parts.append(f"  Invariants: {', '.join(invariants[:2])}")
        
        if plan.risks:
            rationale_parts.append("\n**Risks considered:**")
            for risk in plan.risks[:3]:
                rationale_parts.append(f"- {risk}")
        
        # Ensure we have at least one alternative
        if not patch.alternatives:
            patch.alternatives = [
                "Manual implementation with custom approach",
                "Defer this change to a later iteration",
            ]
        
        enhanced_rationale = "\n".join(rationale_parts)
        
        return Patch(
            id=patch.id,
            task_id=patch.task_id,
            step_reference=patch.step_reference,
            diff=patch.diff,
            rationale=enhanced_rationale,
            alternatives=patch.alternatives,
            status=patch.status,
            kind=patch.kind,
        )

    def answer_followup_question(
        self,
        patch: Patch,
        question: str,
        plan_step: PlanStep,
        plan: Plan,
    ) -> str:
        """
        Answer a follow-up question about a patch's rationale.
        
        Epic 4.1: Support for follow-up questions in CLI/MCP.
        
        Args:
            patch: The patch being questioned
            question: The follow-up question
            plan_step: The plan step context
            plan: The full plan context
            
        Returns:
            Answer to the follow-up question
        """
        if self.llm_client:
            try:
                prompt = f"""Answer this follow-up question about a code change:

Patch Rationale:
{patch.rationale}

Plan Step: {plan_step.description}
Alternatives Considered: {', '.join(patch.alternatives[:3])}

Question: {question}

Provide a clear, concise answer that references the rationale, design decisions, and alternatives."""

                return self.llm_client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior software engineer explaining code changes. Answer questions clearly and reference the rationale.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_output_tokens=500,
                )
            except Exception as exc:
                LOG.error("Failed to answer follow-up question: %s", exc)
                return f"Error generating answer: {exc}. Please review the patch rationale manually."
        else:
            return "LLM client not available. Please review the patch rationale: " + patch.rationale
