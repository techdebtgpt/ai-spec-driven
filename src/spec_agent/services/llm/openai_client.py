from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAI, OpenAIError


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

        try:
            response = self._client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_output_tokens=max_output_tokens,
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
        Pull the human-readable text content out of the Responses API payload.
        """

        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            content = getattr(item, "content", None)
            if not content:
                continue
            for block in content:
                if getattr(block, "type", None) == "text":
                    value = getattr(getattr(block, "text", None), "value", "")
                    if value:
                        chunks.append(value)
        extracted = "\n".join(chunk for chunk in chunks if chunk).strip()
        if not extracted:
            LOG.debug("OpenAI response missing text content: %s", response)
        return extracted

