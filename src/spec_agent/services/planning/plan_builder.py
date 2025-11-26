from __future__ import annotations

from typing import List
from uuid import uuid4

from ...domain.models import Plan, PlanStep


class PlanBuilder:
    """
    Produces high-level implementation plans once clarifications are resolved.
    """

    def build_plan(self, task_id: str, description: str, context_summary: dict) -> Plan:
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


