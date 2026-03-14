"""
Ollama LLM class.

Wrapper for Ollama local LLM models.
"""

from __future__ import annotations

from src.llm.providers.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Wrapper around :class:`vanna.integrations.ollama.OllamaLlmService`."""

    def __init__(self, model: str, base_url: str):
        try:
            from vanna.integrations.ollama import OllamaLlmService
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Ollama LLM service not available. Install with: pip install vanna[ollama]"
            ) from e

        self.service = OllamaLlmService(model=model, base_url=base_url)
