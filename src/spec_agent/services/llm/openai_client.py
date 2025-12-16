from __future__ import annotations

import logging
from typing import Dict, List, Optional

try:
    # Optional at dev-time: editors/CI environments may not have extras installed.
    # If you want LLM features, install dependencies (e.g. `pip install -e .`).
    from openai import OpenAI, OpenAIError  # type: ignore[reportMissingImports]
except ModuleNotFoundError as exc:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    OpenAIError = Exception  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = exc


LOG = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when an LLM provider returns an invalid response."""


class OpenAILLMClient:
    """
    Thin wrapper around the OpenAI Responses API used throughout the agent.

    The wrapper keeps the rest of the codebase agnostic of the specific OpenAI
    SDK surface area and centralizes text extraction + error handling.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "OpenAI SDK not installed. Install project dependencies (e.g. `pip install -e .`) "
                "or add `openai` to your environment."
            ) from _OPENAI_IMPORT_ERROR

        if not api_key:
            raise ValueError("OpenAI API key must be provided when enabling the LLM.")

        client_kwargs = {"api_key": api_key, "timeout": timeout_seconds}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str, *, max_output_tokens: int = 800) -> str:
        """
        Execute a short, deterministic call against the configured OpenAI model.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, temperature=0.2, max_output_tokens=max_output_tokens)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> str:
        """
        Generic chat completion helper so downstream services do not shell out
        to the SDK directly.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except OpenAIError as exc:
            raise LLMClientError(f"OpenAI request failed: {exc}") from exc

        text = self._extract_text(response)
        if not text:
            raise LLMClientError("OpenAI response did not include usable text.")
        return text.strip()

    @staticmethod
    def _extract_text(response: object) -> str:
        """
        Pull the human-readable text content out of the Chat Completions API response.
        """

        try:
            # Standard OpenAI chat completions response structure
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                    return first_choice.message.content or ""

            LOG.debug("OpenAI response missing expected structure: %s", response)
            return ""

        except Exception as exc:
            LOG.error("Failed to extract text from OpenAI response: %s", exc)
            return ""
