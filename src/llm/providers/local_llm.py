"""
Local LLM class.

Wrapper for Local LLM models such as when running a local
LM Studio server or calling the OpenAI API directly.
"""

from __future__ import annotations

from src.llm.providers.base import BaseLLM


class LocalLLM(BaseLLM):
    """Wrapper around :class:`vanna.integrations.openai.OpenAILlmService`.

    This covers ``local-llm`` models such as when running a local
    LM Studio server or calling the OpenAI API directly.
    """

    def __init__(self, model: str, base_url: str, api_key: str):
        try:
            from vanna.integrations.openai import OpenAILlmService
        except ImportError as e:  # pragma: no cover - dependency may be missing
            raise ImportError(
                "Local LLM service not available. Install with: pip install vanna[openai]"
            ) from e

        self.service = OpenAILlmService(model=model, base_url=base_url, api_key=api_key)
