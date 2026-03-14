"""
Snowflake connection factory.

Provides a single source of truth for creating Snowflake connections,
eliminating credential duplication across modules.
"""

import logging
from typing import Any

from src.database.connections.base import BaseConnectionFactory
from src.database.registry import DatabaseRegistry
from src.security.key_management import RSAKeyLoadError, get_snowflake_key_bytes

logger = logging.getLogger(__name__)


class SnowflakeConnectionFactory(BaseConnectionFactory):
    """Creates Snowflake connections from centralised configuration."""

    backend = "snowflake"

    def connect(self) -> Any:
        """Open and return a new Snowflake connection."""
        try:
            import snowflake.connector
        except ImportError as e:
            raise ImportError(
                "snowflake-connector-python is required for Snowflake connections; "
                "install with `pip install 'vanna[snowflake]'`"
            ) from e

        try:
            private_key_path = self._config.get("private_key_path", "")
            private_key_bytes = get_snowflake_key_bytes(private_key_path)
        except RSAKeyLoadError as e:
            logger.error("Failed to load RSA key for Snowflake connection: %s", e)
            raise ValueError(f"Snowflake requires a valid RSA key: {e}") from e

        logger.info(
            "Connecting to Snowflake (%s / %s.%s)",
            self._config.get("account"),
            self._config.get("database"),
            self._config.get("schema"),
        )
        return snowflake.connector.connect(
            account=self._config.get("account"),
            user=self._config.get("user"),
            private_key=private_key_bytes,
            database=self._config.get("database"),
            schema=self._config.get("schema"),
            warehouse=self._config.get("warehouse"),
            role=self._config.get("role"),
        )

    @property
    def database(self) -> str | None:
        return self._config.get("database")

    @property
    def schema(self) -> str | None:
        return self._config.get("schema")


DatabaseRegistry.register_connection("snowflake", SnowflakeConnectionFactory)
