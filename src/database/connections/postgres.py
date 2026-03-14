"""
PostgreSQL connection factory.

Provides a single source of truth for creating psycopg2 connections.
"""

import logging
from typing import Any

from src.database.connections.base import BaseConnectionFactory
from src.database.registry import DatabaseRegistry

logger = logging.getLogger(__name__)


class PostgresConnectionFactory(BaseConnectionFactory):
    """Creates PostgreSQL connections from centralized configuration."""

    backend = "postgres"

    def connect(self) -> Any:
        """Open and return a new psycopg2 connection."""
        # import inside method so we don't require psycopg2 unless the class is used
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2 is required for Postgres connections; "
                "install with `pip install 'vanna[postgres]'`"
            ) from e

        logger.info(
            "Connecting to PostgreSQL (%s@%s:%s/%s)",
            self._config.get("user"),
            self._config.get("host"),
            self._config.get("port", 5432),
            self._config.get("database"),
        )

        conn_args: dict[str, Any] = {
            "host": self._config.get("host"),
            "database": self._config.get("database"),
            "user": self._config.get("user"),
            "password": self._config.get("password"),
        }
        if "port" in self._config:
            conn_args["port"] = int(self._config.get("port", 5432))

        return psycopg2.connect(**conn_args)

    @property
    def database(self) -> str | None:
        return self._config.get("database")

    @property
    def host(self) -> str | None:
        return self._config.get("host")

    @property
    def port(self) -> int:
        return int(self._config.get("port", 5432))


DatabaseRegistry.register_connection("postgresql", PostgresConnectionFactory)
DatabaseRegistry.register_connection("postgres", PostgresConnectionFactory)
