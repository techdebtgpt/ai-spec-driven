from __future__ import annotations

from typing import Dict, List, Protocol


class LLMClientError(RuntimeError):
    """Raised when an LLM provider returns an invalid response."""


class LLMClientProtocol(Protocol):
    """Minimal interface for LLM providers used by Spec Agent."""

    def complete(self, system_prompt: str, user_prompt: str, *, max_output_tokens: int = 800) -> str:  # pragma: no cover
        ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> str:  # pragma: no cover
        ...
