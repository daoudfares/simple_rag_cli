"""
Database tool factory.

Creates a ``RunSqlTool`` backed by the appropriate database runner
based on the named database profile from the configuration.
"""

import logging

from vanna.integrations.mysql import MySQLRunner
from vanna.integrations.oracle import OracleRunner
from vanna.integrations.postgres import PostgresRunner
from vanna.integrations.snowflake import SnowflakeRunner
from vanna.tools import RunSqlTool

from src.config.config_loader import get_database_config
from src.security.key_management import RSAKeyLoadError, get_snowflake_key_bytes

logger = logging.getLogger(__name__)


def _create_postgres_tool(config: dict) -> RunSqlTool:
    """Create Postgres RunSqlTool from config."""
    port = int(config.get("port", 5432))
    logger.info("Creating Postgres RunSqlTool for %s", config.get("database"))
    return RunSqlTool(
        sql_runner=PostgresRunner(
            host=config["host"],
            port=port,
            database=config["database"],
            user=config["user"],
            password=config.get("password"),
        )
    )


def _create_mysql_tool(config: dict) -> RunSqlTool:
    """Create MySQL RunSqlTool from config."""
    port = int(config.get("port", 3306))
    logger.info("Creating MySQL RunSqlTool for %s", config.get("database"))
    return RunSqlTool(
        sql_runner=MySQLRunner(
            host=config["host"],
            port=port,
            database=config["database"],
            user=config["user"],
            password=config.get("password"),
        )
    )


def _create_oracle_tool(config: dict) -> RunSqlTool:
    """Create Oracle RunSqlTool from config."""
    logger.info("Creating Oracle RunSqlTool for %s", config.get("dsn"))
    return RunSqlTool(
        sql_runner=OracleRunner(
            user=config["user"],
            password=config.get("password"),
            dsn=config["dsn"],
        )
    )


def _create_snowflake_tool(config: dict) -> RunSqlTool:
    """Create Snowflake RunSqlTool from config."""
    logger.info("Creating Snowflake RunSqlTool for %s.%s", config["database"], config["schema"])
    try:
        p_key_bytes = get_snowflake_key_bytes(config["private_key_path"])
    except RSAKeyLoadError as e:
        logger.error("Failed to load RSA key for Snowflake: %s", e)
        raise ValueError(f"Snowflake requires a valid RSA key: {e}") from e

    return RunSqlTool(
        sql_runner=SnowflakeRunner(
            account=config["account"],
            username=config["user"],
            private_key_content=p_key_bytes,
            database=config["database"],
            schema=config["schema"],
            warehouse=config["warehouse"],
            role=config["role"],
            private_key=p_key_bytes,
        )
    )


_DB_TYPE_FACTORIES = {
    "postgresql": _create_postgres_tool,
    "mysql": _create_mysql_tool,
    "oracle": _create_oracle_tool,
    "snowflake": _create_snowflake_tool,
}


def get_db_tool(db_name: str) -> RunSqlTool:
    """Return a RunSqlTool for the named database profile.

    Args:
        db_name: Name of the database profile (e.g. ``snowflake``,
                 ``postgres_local``).  Must exist in ``secrets.toml``.

    Returns:
        A RunSqlTool instance configured for the selected backend.

    Raises:
        ValueError: If the profile does not exist or the ``type`` is unknown.
    """
    config = get_database_config(db_name)
    db_type = config["type"]  # guaranteed by config_loader validation

    factory = _DB_TYPE_FACTORIES.get(db_type)
    if factory is None:
        raise ValueError(
            f"Unknown database type '{db_type}' in profile '{db_name}'. "
            f"Valid types: {', '.join(_DB_TYPE_FACTORIES)}"
        )

    return factory(config)
