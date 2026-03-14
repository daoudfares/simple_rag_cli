"""
Anthropic LLM class.

Wrapper for Anthropic models using the Anthropic API.
"""

from __future__ import annotations

from src.llm.providers.base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Wrapper around :class:`vanna.integrations.anthropic.AnthropicLlmService`."""

    def __init__(self, model: str, api_key: str):
        try:
            from vanna.integrations.anthropic import AnthropicLlmService
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Anthropic LLM service not available. Install with: pip install vanna[anthropic]"
            ) from e

        self.service = AnthropicLlmService(model=model, api_key=api_key)
