"""
Base connection factory interface.

Provides a common interface for all database connection factories.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseConnectionFactory(ABC):
    """Abstract base class for all database connection factories."""

    backend: str

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    @abstractmethod
    def connect(self) -> Any:
        """Open and return a new database connection."""
        pass
