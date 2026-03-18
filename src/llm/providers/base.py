"""
Base LLM class for the application.

Defines a common abstract base that all LLM provider subclasses
inherit from, allowing the rest of the code to remain provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """A response from an LLM service containing both text and optional metadata."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class for all LLM providers used in this project.

    Subclasses must initialise ``self.service`` with an object that
    actually implements the provider API (typically one of the classes
    from ``vanna.integrations``).  The base class provides a simple
    proxy so caller code can remain provider-agnostic — any attribute
    not found on the wrapper is forwarded to ``self.service``.
    """

    service: Any

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any):
        """Constructor should set ``self.service``.

        The signature is intentionally loose; each provider subclass
        documents its own parameters.
        """
        ...

    # ── Transparent proxy ────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying service.

        This allows the Vanna Agent to call any method on the LLM
        service (e.g. ``stream_request``, ``generate``) without the
        wrapper having to explicitly re-implement each one.
        """
        # __getattr__ is only called when normal lookup fails, so
        # this won't shadow attributes set on the instance itself.
        if "service" in self.__dict__:
            return getattr(self.__dict__["service"], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate text using the underlying service.

        This helper keeps the public interface consistent, but callers
        who need provider-specific functionality can still reach into
        ``.service``.

        Most Vanna services expose a ``generate`` method, a ``submit_prompt``
        method, or in Vanna 2.0+, a ``send_request`` method. This helper
        tries all of them.
        """
        if not hasattr(self, "service"):
            raise RuntimeError("LLM service has not been initialised")

        import asyncio
        import re

        res = None
        # 1. Try ``generate`` (shorthand for simple text generation)
        if hasattr(self.service, "generate"):
            res = self.service.generate(prompt, **kwargs)
            if asyncio.iscoroutine(res):
                res = await res

        # 2. Try ``send_request`` (the standard Vanna 2.0 execution method)
        elif hasattr(self.service, "send_request"):
            # Prepare request according to Vanna 2.0 specs
            try:
                from vanna.core.llm.models import LlmRequest
                from vanna.core.user import User

                # Vanna 2.0 LlmRequest expects messages and user
                dummy_user = User(id="default", name="Default User")
                messages = [{"role": "user", "content": prompt}]
                request = LlmRequest(prompt=prompt, messages=messages, user=dummy_user)
                res = self.service.send_request(request)
            except (ImportError, TypeError):
                # Expected failures: missing module or wrong call signature.
                import logging
                logging.getLogger(__name__).debug(
                    "Failed to create complex LlmRequest, falling back", exc_info=True,
                )
                res = self.service.send_request(prompt, **kwargs)
            except Exception:
                # Unexpected failure — log at warning and try the simpler call.
                import logging
                logging.getLogger(__name__).warning(
                    "Unexpected error creating LlmRequest, falling back to simple call",
                    exc_info=True,
                )
                res = self.service.send_request(prompt, **kwargs)

            if asyncio.iscoroutine(res):
                res = await res

        # 3. Fall back to ``submit_prompt`` (the standard Vanna legacy execution method)
        elif hasattr(self.service, "submit_prompt"):
            res = self.service.submit_prompt(prompt, **kwargs)
            if asyncio.iscoroutine(res):
                res = await res

        if res is None:
            raise AttributeError(
                f"The underlying LLM service '{type(self.service).__name__}' "
                "has neither 'generate', 'send_request', nor 'submit_prompt' methods."
            )

        # Extract content and metadata
        text = ""
        metadata = {}

        # If it's a Vanna LlmResponse or similar object
        if hasattr(res, "text") and isinstance(res.text, str):
            text = res.text
            # Try to extract metadata from object attributes if they exist
            for attr in ["usage", "metadata", "finish_reason", "model"]:
                if hasattr(res, attr):
                    metadata[attr] = getattr(res, attr)
        else:
            text = str(res)

        # Robust check for string representation of an object (e.g. from logs/local LLM)
        # Pattern: content='...' tool_calls=... usage=...
        content_match = re.match(r"^content=['\"](.*?)['\"]\s+tool_calls=", text, re.DOTALL)
        if content_match:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Parsing LLM response from its string representation — "
                "this is fragile and may break if the format changes."
            )
            actual_text = content_match.group(1)
            # Try to extract metadata from the string representation
            # finish_reason='stop'
            fr_match = re.search(r"finish_reason=['\"](.*?)['\"]", text)
            if fr_match:
                metadata["finish_reason"] = fr_match.group(1)

            # usage={'prompt_tokens': 283, ...}
            usage_match = re.search(r"usage=(\{.*?\})", text)
            if usage_match:
                try:
                    # Very rough attempt to parse the dict-like string
                    import ast
                    metadata["usage"] = ast.literal_eval(usage_match.group(1))
                except Exception:
                    metadata["usage_raw"] = usage_match.group(1)

            return LLMResponse(actual_text, metadata=metadata)

        return LLMResponse(text, metadata=metadata)
