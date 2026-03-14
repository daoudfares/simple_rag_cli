"""
Basic tests for application factory behaviour with named profiles.
"""

from unittest.mock import MagicMock, patch

import app as app_module


class TestCreateApp:
    """Tests for create_app with the new named-profile API."""

    def test_create_app_snowflake(self):
        """create_app should work with a snowflake database profile."""
        with (
            patch("app.get_llm", return_value=MagicMock()),
            patch("app.get_db_tool", return_value=MagicMock()),
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={
                    "type": "snowflake",
                    "account": "a",
                    "user": "u",
                    "database": "db",
                    "schema": "sch",
                },
            ),
            patch("src.database.registry.DatabaseRegistry.get_connection_factory") as mock_reg,
        ):
            result = app_module.create_app(llm_name="x", db_name="sf")
            mock_reg.assert_called_once()
            assert "agent" in result
            assert "connection_factory" in result

    def test_create_app_postgres(self):
        """create_app should work with a postgresql database profile."""
        with (
            patch("app.get_llm", return_value=MagicMock()),
            patch("app.get_db_tool", return_value=MagicMock()),
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={
                    "type": "postgresql",
                    "host": "h",
                    "user": "u",
                    "password": "p",
                    "database": "d",
                },
            ),
            patch("src.database.registry.DatabaseRegistry.get_connection_factory") as mock_reg,
        ):
            result = app_module.create_app(llm_name="x", db_name="pg")
            mock_reg.assert_called_once()
            assert "connection_factory" in result

    def test_create_app_mysql(self):
        """create_app should work with a mysql database profile."""
        with (
            patch("app.get_llm", return_value=MagicMock()),
            patch("app.get_db_tool", return_value=MagicMock()),
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={
                    "type": "mysql",
                    "host": "h",
                    "user": "u",
                    "password": "p",
                    "database": "d",
                },
            ),
            patch("src.database.registry.DatabaseRegistry.get_connection_factory") as mock_reg,
        ):
            result = app_module.create_app(llm_name="x", db_name="my")
            mock_reg.assert_called_once()
            assert "connection_factory" in result

    def test_create_app_oracle(self):
        """create_app should work with an oracle database profile."""
        with (
            patch("app.get_llm", return_value=MagicMock()),
            patch("app.get_db_tool", return_value=MagicMock()),
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={
                    "type": "oracle",
                    "user": "u",
                    "password": "p",
                    "dsn": "x",
                    "database": "orcl",
                },
            ),
            patch("src.database.registry.DatabaseRegistry.get_connection_factory") as mock_reg,
        ):
            result = app_module.create_app(llm_name="x", db_name="ora")
            mock_reg.assert_called_once()
            assert "connection_factory" in result

    def test_llm_name_forwarded(self):
        """create_app should forward llm_name to get_llm."""
        with (
            patch("app.get_llm") as mock_get_llm,
            patch("app.get_db_tool", return_value=MagicMock()),
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={"type": "snowflake", "database": "db", "schema": "s"},
            ),
            patch(
                "src.database.registry.DatabaseRegistry.get_connection_factory",
                return_value=MagicMock(),
            ),
        ):
            mock_get_llm.return_value = MagicMock()
            app_module.create_app(llm_name="gemini", db_name="sf")
            mock_get_llm.assert_called_once_with("gemini")

    def test_db_name_forwarded(self):
        """create_app should forward db_name to get_db_tool."""
        with (
            patch("app.get_llm", return_value=MagicMock()),
            patch("app.get_db_tool") as mock_get_db,
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={"type": "snowflake", "database": "db", "schema": "s"},
            ),
            patch(
                "src.database.registry.DatabaseRegistry.get_connection_factory",
                return_value=MagicMock(),
            ),
        ):
            mock_get_db.return_value = MagicMock()
            app_module.create_app(llm_name="x", db_name="my_snowflake")
            mock_get_db.assert_called_once_with("my_snowflake")
