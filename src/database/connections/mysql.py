"""
MySQL connection factory.

Provides a single source of truth for creating PyMySQL connections.
"""

import logging
from typing import Any

from src.database.connections.base import BaseConnectionFactory
from src.database.registry import DatabaseRegistry

logger = logging.getLogger(__name__)


class MySQLConnectionFactory(BaseConnectionFactory):
    """Creates MySQL connections from centralized configuration."""

    backend = "mysql"

    def connect(self) -> Any:
        try:
            import pymysql
        except ImportError as e:
            raise ImportError(
                "PyMySQL is required for MySQL connections; "
                "install with `pip install 'vanna[mysql]'`"
            ) from e

        logger.info(
            "Connecting to MySQL (%s@%s:%s/%s)",
            self._config.get("user"),
            self._config.get("host"),
            self._config.get("port", 3306),
            self._config.get("database"),
        )

        conn_args: dict[str, Any] = {
            "host": self._config.get("host"),
            "user": self._config.get("user"),
            "password": self._config.get("password"),
            "database": self._config.get("database"),
            "port": int(self._config.get("port", 3306)),
        }
        return pymysql.connect(**conn_args)

    @property
    def database(self) -> str | None:
        return self._config.get("database")

    @property
    def host(self) -> str | None:
        return self._config.get("host")

    @property
    def port(self) -> int:
        return int(self._config.get("port", 3306))


DatabaseRegistry.register_connection("mysql", MySQLConnectionFactory)
