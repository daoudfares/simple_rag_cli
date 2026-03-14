"""
Base LLM class for the application.

Defines a common abstract base that all LLM provider subclasses
inherit from, allowing the rest of the code to remain provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate text using the underlying service.

        This helper keeps the public interface consistent, but callers
        who need provider-specific functionality can still reach into
        ``.service``.
        """
        if not hasattr(self, "service"):
            raise RuntimeError("LLM service has not been initialised")
        # most vanna services expose a ``generate`` method; delegate to it
        return self.service.generate(prompt, **kwargs)
