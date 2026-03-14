"""
Unit tests for the database management helpers.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestDatabaseManagement:
    """Tests for get_db_tool with named profiles."""

    SNOWFLAKE_SAMPLE = {
        "type": "snowflake",
        "account": "acct",
        "user": "u",
        "role": "r",
        "warehouse": "w",
        "database": "db",
        "schema": "sch",
        "private_key_path": "fake_key.p8",
    }

    POSTGRES_SAMPLE = {
        "type": "postgresql",
        "host": "localhost",
        "port": "5433",
        "database": "pgdb",
        "user": "pguser",
        "password": "pgpass",
    }

    MYSQL_SAMPLE = {
        "type": "mysql",
        "host": "localhost",
        "port": "3306",
        "database": "mydb",
        "user": "myuser",
        "password": "mypass",
    }

    ORACLE_SAMPLE = {
        "type": "oracle",
        "user": "orauser",
        "password": "orapass",
        "dsn": "host:1521/sid",
    }

    def test_get_db_tool_snowflake(self):
        """get_db_tool should create a Snowflake tool for a snowflake profile."""
        from src.database import database_management as dbm

        with (
            patch(
                "src.database.database_management.get_database_config",
                return_value=self.SNOWFLAKE_SAMPLE,
            ),
            patch("src.database.database_management.get_snowflake_key_bytes", return_value=b"key"),
            patch("src.database.database_management.SnowflakeRunner") as mock_runner,
            patch("src.database.database_management.RunSqlTool") as mock_tool,
        ):
            mock_runner.return_value = MagicMock()
            mock_tool.return_value = "tool"

            tool = dbm.get_db_tool("snowflake")
            assert tool == "tool"

            mock_runner.assert_called_once_with(
                account="acct",
                username="u",
                private_key_content=b"key",
                database="db",
                schema="sch",
                warehouse="w",
                role="r",
                private_key=b"key",
            )
            mock_tool.assert_called_once()

    def test_get_db_tool_postgres(self):
        """get_db_tool should create a Postgres tool for a postgresql profile."""
        from src.database import database_management as dbm

        with (
            patch(
                "src.database.database_management.get_database_config",
                return_value=self.POSTGRES_SAMPLE,
            ),
            patch("src.database.database_management.PostgresRunner") as mock_runner,
            patch("src.database.database_management.RunSqlTool") as mock_tool,
        ):
            mock_runner.return_value = MagicMock()
            mock_tool.return_value = "pgtool"

            tool = dbm.get_db_tool("postgres_local")
            assert tool == "pgtool"

            mock_runner.assert_called_once_with(
                host="localhost",
                port=5433,
                database="pgdb",
                user="pguser",
                password="pgpass",
            )
            mock_tool.assert_called_once()

    def test_get_db_tool_mysql(self):
        """get_db_tool should create a MySQL tool for a mysql profile."""
        from src.database import database_management as dbm

        with (
            patch(
                "src.database.database_management.get_database_config",
                return_value=self.MYSQL_SAMPLE,
            ),
            patch("src.database.database_management.MySQLRunner") as mock_runner,
            patch("src.database.database_management.RunSqlTool") as mock_tool,
        ):
            mock_runner.return_value = MagicMock()
            mock_tool.return_value = "mytool"

            tool = dbm.get_db_tool("mysql_prod")
            assert tool == "mytool"

            mock_runner.assert_called_once_with(
                host="localhost",
                port=3306,
                database="mydb",
                user="myuser",
                password="mypass",
            )
            mock_tool.assert_called_once()

    def test_get_db_tool_oracle(self):
        """get_db_tool should create an Oracle tool for an oracle profile."""
        from src.database import database_management as dbm

        with (
            patch(
                "src.database.database_management.get_database_config",
                return_value=self.ORACLE_SAMPLE,
            ),
            patch("src.database.database_management.OracleRunner") as mock_runner,
            patch("src.database.database_management.RunSqlTool") as mock_tool,
        ):
            mock_runner.return_value = MagicMock()
            mock_tool.return_value = "oratool"

            tool = dbm.get_db_tool("oracle_dev")
            assert tool == "oratool"

            mock_runner.assert_called_once_with(
                user="orauser",
                password="orapass",
                dsn="host:1521/sid",
            )
            mock_tool.assert_called_once()

    def test_get_db_tool_unknown_type_raises(self):
        """get_db_tool should raise ValueError for unknown database type."""
        from src.database import database_management as dbm

        with (
            patch(
                "src.database.database_management.get_database_config",
                return_value={"type": "unknown_db"},
            ),
            pytest.raises(ValueError, match="Unknown database type"),
        ):
            dbm.get_db_tool("bad_profile")

    def test_get_db_tool_unknown_profile_raises(self):
        """get_db_tool should raise ValueError for non-existent profile."""
        from src.database import database_management as dbm

        with (
            patch(
                "src.database.database_management.get_database_config",
                side_effect=ValueError("not found"),
            ),
            pytest.raises(ValueError, match="not found"),
        ):
            dbm.get_db_tool("nonexistent")
