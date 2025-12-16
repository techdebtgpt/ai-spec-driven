from __future__ import annotations

import json
import logging
from typing import List, Optional
from uuid import uuid4

from ...domain.models import ClarificationItem

LOG = logging.getLogger(__name__)


class Clarifier:
    """
    Runs a clarity assessment and generates targeted questions.

    When an LLM client is provided, generates context-aware clarification questions.
    Falls back to keyword-based heuristics when LLM is unavailable or fails.
    """

    def __init__(self, llm_client: Optional[object] = None) -> None:
        """
        Initialize Clarifier with optional LLM client.

        Args:
            llm_client: Optional OpenAILLMClient for AI-powered question generation
        """
        self.llm_client = llm_client

    def generate_questions(
        self,
        task_id: str,
        description: str,
        context_summary: Optional[dict] = None
    ) -> List[ClarificationItem]:
        """
        Generate clarification questions for the given task description.

        Attempts LLM-based generation first if available, with fallback to heuristics.
        """
        if self.llm_client:
            try:
                LOG.info("Generating clarification questions using LLM for task %s", task_id)
                return self._generate_with_llm(task_id, description, context_summary)
            except Exception as exc:
                LOG.warning("LLM question generation failed: %s, falling back to heuristics", exc)
                return self._generate_template(task_id, description)
        else:
            LOG.debug("No LLM client available, using heuristic questions")
            return self._generate_template(task_id, description)

    def _generate_with_llm(
        self,
        task_id: str,
        description: str,
        context_summary: Optional[dict]
    ) -> List[ClarificationItem]:
        """Generate clarification questions using LLM analysis."""
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(description, context_summary)

        # Call LLM
        response = self.llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1000
        )

        # Parse JSON response
        questions = self._parse_llm_response(response)

        # Build ClarificationItem objects
        return [
            ClarificationItem(id=str(uuid4()), task_id=task_id, question=question)
            for question in questions
        ]

    def _create_system_prompt(self) -> str:
        """Define the LLM's role for clarification question generation."""
        return """You are a technical requirements analyst helping engineers clarify feature requests.

Your task is to generate 3-5 targeted clarifying questions that will help create a better implementation plan.

Focus on questions about:
1. **Scope**: What's in/out of scope? Edge cases? Boundaries?
2. **Technical Approach**: Preferred patterns, libraries, or existing code to follow?
3. **Dependencies**: What systems, modules, or APIs are affected?
4. **Testing**: What testing strategy? Which tests need updating? (If no tests are detected in the repo, do NOT ask about tests unless the request explicitly mentions them.)
5. **Risks**: Known constraints, performance concerns, or compatibility issues?

Guidelines:
- Ask specific, answerable questions (not open-ended philosophy)
- Prioritize questions that would most impact the implementation plan
- Consider the repository context (languages, modules) when asking
- Avoid asking about things already clear in the description

Return ONLY valid JSON in this format:
{
  "questions": [
    "Specific question 1?",
    "Specific question 2?",
    "Specific question 3?"
  ]
}

Do not include any explanatory text before or after the JSON."""

    def _create_user_prompt(self, description: str, context_summary: Optional[dict]) -> str:
        """Build the user prompt with task description and repository context."""
        # Extract context information
        languages = ""
        modules = ""
        tests_str = "unknown"

        if context_summary:
            top_languages = context_summary.get("top_languages", [])
            languages = ", ".join(top_languages[:3]) if top_languages else "unknown"

            top_modules = context_summary.get("top_modules", [])
            modules = ", ".join(top_modules[:5]) if top_modules else "not analyzed"
            tests_str = "present" if bool(context_summary.get("has_tests", False)) else "not detected"

        return f"""Task Description: {description}

Repository Context:
- Languages: {languages}
- Key Modules: {modules}
- Tests: {tests_str}

Generate clarifying questions that will help create a detailed, accurate implementation plan."""

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse and validate the LLM's JSON response."""
        try:
            # Try to extract JSON if LLM included extra text
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                if response.startswith("json"):
                    response = response[4:].strip()

            data = json.loads(response)

            # Validate structure
            if "questions" not in data:
                raise ValueError("LLM response missing 'questions' field")
            if not isinstance(data["questions"], list):
                raise ValueError("'questions' must be a list")

            questions = data["questions"]

            # Validate each question
            if not questions:
                raise ValueError("No questions returned")
            if len(questions) > 10:
                LOG.warning("LLM returned %d questions, truncating to 10", len(questions))
                questions = questions[:10]

            return questions

        except json.JSONDecodeError as exc:
            LOG.error("Failed to parse LLM response as JSON: %s", exc)
            LOG.debug("LLM response was: %s", response)
            raise ValueError(f"Invalid JSON response from LLM: {exc}") from exc

    def _generate_template(self, task_id: str, description: str) -> List[ClarificationItem]:
        """Fallback template-based question generation (original implementation)."""
        questions: List[str] = []

        if "how" not in description.lower():
            questions.append("What is the expected end-state or observable behavior?")
        if "test" not in description.lower():
            questions.append("Which tests (unit/integration) must be updated or created?")
        if "boundary" in description.lower():
            questions.append("Should this change stay within the current subsystem boundary?")

        # Always ask about scope if description is short
        if len(description.split()) < 10:
            questions.insert(0, "Can you provide more details about what this change should accomplish?")

        return [
            ClarificationItem(id=str(uuid4()), task_id=task_id, question=question)
            for question in questions
        ]


