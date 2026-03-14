"""
Google Gemini LLM class.

Wrapper for Google Gemini models using the Google Generative AI API.
"""

from __future__ import annotations

from src.llm.providers.base import BaseLLM


class GeminiLLM(BaseLLM):
    """Wrapper around :class:`vanna.integrations.google.GeminiLlmService`."""

    def __init__(self, model: str, api_key: str):
        try:
            from vanna.integrations.google import GeminiLlmService
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Google Gemini LLM service not available. "
                "Install with: pip install google-generativeai"
            ) from e

        self.service = GeminiLlmService(model=model, api_key=api_key)
