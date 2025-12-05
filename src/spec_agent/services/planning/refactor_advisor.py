from __future__ import annotations

from typing import List
from uuid import uuid4

from ...domain.models import Plan, RefactorSuggestion


class RefactorAdvisor:
    """
    Heuristic advisor that flags potential refactors for each plan step.

    This keeps refactor suggestions explicit and separate from the functional
    patches, fulfilling the Epic 3 requirement without requiring a full static
    analysis engine yet.
    """

    def suggest(self, plan: Plan) -> List[RefactorSuggestion]:
        suggestions: List[RefactorSuggestion] = []
        for step in plan.steps:
            scope = step.target_files or []
            description = step.description.lower()

            if "refactor" in description or "extract" in description:
                suggestions.append(
                    RefactorSuggestion(
                        id=str(uuid4()),
                        task_id=plan.task_id,
                        description=f"Break down work for: {step.description}",
                        rationale="Step already references a refactor; track it explicitly.",
                        scope=scope,
                    )
                )
            elif len(step.description) > 120:
                suggestions.append(
                    RefactorSuggestion(
                        id=str(uuid4()),
                        task_id=plan.task_id,
                        description=f"Extract helper to keep implementation concise for: {step.description[:80]}...",
                        rationale="Long plan steps often hide multiple responsibilities.",
                        scope=scope,
                    )
                )
            elif scope:
                suggestions.append(
                    RefactorSuggestion(
                        id=str(uuid4()),
                        task_id=plan.task_id,
                        description=f"Review {', '.join(scope[:2])} for duplication before applying changes.",
                        rationale="Touching critical files warrants a design pass to prevent drift.",
                        scope=scope,
                    )
                )
        return suggestions

