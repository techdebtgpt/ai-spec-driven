from __future__ import annotations

from typing import List
from uuid import uuid4

from ...domain.models import Plan, TestSuggestion


class TestSuggester:
    """
    Provides candidate test updates based on the approved plan.
    """

    def suggest(self, plan: Plan) -> List[TestSuggestion]:
        suggestions: List[TestSuggestion] = []
        for step in plan.steps:
            suggestion = TestSuggestion(
                id=str(uuid4()),
                task_id=plan.task_id,
                description=f"Verify behavior for: {step.description}",
                suggestion_type="UNIT" if step.target_files else "INTEGRATION",
                related_files=step.target_files,
                skeleton_code="def test_placeholder():\n    assert True",
            )
            suggestions.append(suggestion)
        return suggestions


