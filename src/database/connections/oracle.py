"""
Oracle connection factory.

Provides a single source of truth for creating oracledb connections.
"""

import logging
from typing import Any

from src.database.connections.base import BaseConnectionFactory
from src.database.registry import DatabaseRegistry

logger = logging.getLogger(__name__)


class OracleConnectionFactory(BaseConnectionFactory):
    """Creates Oracle connections from centralized configuration."""

    backend = "oracle"

    def connect(self) -> Any:
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "oracledb is required for Oracle connections; "
                "install with `pip install 'vanna[oracle]'`"
            ) from e

        logger.info(
            "Connecting to Oracle (%s@%s)",
            self._config.get("user"),
            self._config.get("dsn"),
        )

        return oracledb.connect(
            user=self._config.get("user"),
            password=self._config.get("password"),
            dsn=self._config.get("dsn"),
        )

    @property
    def dsn(self) -> str | None:
        return self._config.get("dsn")

    @property
    def database(self) -> str | None:
        return self._config.get("database")


DatabaseRegistry.register_connection("oracle", OracleConnectionFactory)
