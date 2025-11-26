from __future__ import annotations

from typing import List
from uuid import uuid4

from ...domain.models import ClarificationItem


class Clarifier:
    """
    Runs a clarity assessment and generates targeted questions.

    In this iteration we rely on keyword heuristics so the CLI workflow is
    functional before hooking up an LLM.
    """

    def generate_questions(self, task_id: str, description: str) -> List[ClarificationItem]:
        questions: List[str] = []
        if "how" not in description.lower():
            questions.append("What is the expected end-state or observable behavior?")
        if "test" not in description.lower():
            questions.append("Which tests (unit/integration) must be updated or created?")
        if "boundary" in description.lower():
            questions.append("Should this change stay within the current subsystem boundary?")

        return [
            ClarificationItem(id=str(uuid4()), task_id=task_id, question=question)
            for question in questions
        ]


