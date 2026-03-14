"""
Central registry for database connections and trainers.

Allows for dynamic registration of database backends, supporting the
Open/Closed principle.
"""

from typing import Any

from src.database.connections.base import BaseConnectionFactory
from src.training.base import BaseTrainer


class DatabaseRegistry:
    """Registry for database connection factories and trainers."""

    _connections: dict[str, type[BaseConnectionFactory]] = {}
    _trainers: dict[str, type[BaseTrainer]] = {}

    @classmethod
    def register_connection(cls, db_type: str, factory_cls: type[BaseConnectionFactory]) -> None:
        """Register a connection factory for a database type."""
        cls._connections[db_type.lower()] = factory_cls

    @classmethod
    def register_trainer(cls, db_type: str, trainer_cls: type[BaseTrainer]) -> None:
        """Register a trainer for a database type."""
        cls._trainers[db_type.lower()] = trainer_cls

    @classmethod
    def get_connection_factory(cls, db_type: str, config: dict[str, Any]) -> BaseConnectionFactory:
        """Get a connection factory instance for the given type and config."""
        factory_cls = cls._connections.get(db_type.lower())
        if not factory_cls:
            raise ValueError(f"No connection factory registered for database type '{db_type}'")
        return factory_cls(config)

    @classmethod
    def get_trainer(
        cls, db_type: str, connection_factory: BaseConnectionFactory, agent_memory: Any
    ) -> BaseTrainer:
        """Get a trainer instance for the given type."""
        trainer_cls = cls._trainers.get(db_type.lower())
        if not trainer_cls:
            raise ValueError(f"No trainer registered for database type '{db_type}'")
        return trainer_cls(connection_factory, agent_memory)

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Return list of supported database types."""
        return list(cls._connections.keys())
