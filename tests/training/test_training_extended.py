"""
Extended tests for the training module — covering train_agent and
_build_training_context to increase overall code coverage.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Importing trainer modules triggers DatabaseRegistry.register_trainer() calls.
import src.training.mysql  # noqa: F401
import src.training.oracle  # noqa: F401
import src.training.postgres  # noqa: F401
from src.training.snowflake import (
    SNOWFLAKE_DOCUMENTATION,
    TRAINING_EXAMPLES,
)
from src.training.trainer import train_agent


def _make_mock_memory():
    """Create a mock memory that passes pydantic validation."""
    mock_memory = MagicMock()
    mock_memory.save_text_memory = AsyncMock()
    return mock_memory


class TestBuildTrainingContext:
    """Tests for _build_training_context helper."""

    def test_returns_tool_context(self):
        """_build_training_context should return a ToolContext with admin user."""
        from vanna.tools.agent_memory import AgentMemory

        # Use a spec'd mock that passes isinstance checks
        mock_memory = MagicMock(spec=AgentMemory)
        from src.training.base import _build_training_context

        ctx = _build_training_context(mock_memory)

        assert ctx.user.id == "admin@example.com"
        assert ctx.user.email == "admin@example.com"
        assert "admin" in ctx.user.group_memberships
        assert ctx.conversation_id  # non-empty UUID
        assert ctx.request_id  # non-empty UUID


class TestTrainAgentSnowflake:
    """Tests for train_agent with Snowflake backend."""

    @pytest.fixture()
    def mock_memory(self):
        return _make_mock_memory()

    @pytest.fixture()
    def snowflake_factory(self):
        """Simulate a Snowflake connection factory."""
        factory = MagicMock()
        factory.backend = "snowflake"
        factory.schema = "HUB"

        conn = MagicMock()
        conn.database = "TEST_DB"
        conn.schema = "TEST_SCHEMA"
        cursor = MagicMock()

        tables = [("USERS", "BASE TABLE")]
        columns = [
            ("ID", "NUMBER", "NO", None, None, 38, 0),
            ("NAME", "TEXT", "YES", None, 255, None, None),
        ]
        views = []

        cursor.execute = MagicMock()
        cursor.fetchall = MagicMock(side_effect=[tables, columns, views])
        conn.cursor.return_value = cursor
        factory.connect.return_value = conn
        return factory

    def test_train_agent_snowflake_happy_path(self, mock_memory, snowflake_factory):
        """train_agent should extract schema, add examples, and save docs."""
        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, snowflake_factory))

        # DDL + 6 examples + 1 doc = 8 calls
        assert mock_memory.save_text_memory.call_count == 8

        first_call = mock_memory.save_text_memory.call_args_list[0]
        ddl = first_call.kwargs.get("content", first_call[1].get("content", ""))
        assert "CREATE TABLE USERS" in ddl
        assert "ID NUMBER(38) NOT NULL" in ddl
        assert "NAME VARCHAR(255)" in ddl

    def test_train_agent_saves_all_examples(self, mock_memory, snowflake_factory):
        """train_agent should save one entry per TRAINING_EXAMPLES."""
        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, snowflake_factory))

        saved_contents = [
            call.kwargs.get("content", call[1].get("content", ""))
            for call in mock_memory.save_text_memory.call_args_list
        ]
        for example in TRAINING_EXAMPLES:
            assert any(example["question"] in content for content in saved_contents), (
                f"Missing training example: {example['question']}"
            )

    def test_train_agent_saves_documentation(self, mock_memory, snowflake_factory):
        """train_agent should save Snowflake documentation."""
        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, snowflake_factory))

        last_call = mock_memory.save_text_memory.call_args_list[-1]
        content = last_call.kwargs.get("content", last_call[1].get("content", ""))
        assert content == SNOWFLAKE_DOCUMENTATION


class TestTrainAgentPostgres:
    """Tests for train_agent with Postgres backend."""

    def test_train_agent_postgres_backend(self):
        mock_memory = _make_mock_memory()
        factory = MagicMock()
        factory.backend = "postgres"
        factory.schema = "public"

        conn = MagicMock()
        conn.database = "testdb"
        cursor = MagicMock()
        tables = [("orders", "BASE TABLE")]
        columns = [
            ("id", "integer", "NO", None, None, 32, 0),
            ("amount", "numeric", "YES", None, None, 10, 2),
        ]
        cursor.fetchall = MagicMock(side_effect=[tables, columns])
        conn.cursor.return_value = cursor
        factory.connect.return_value = conn

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, factory))

        assert mock_memory.save_text_memory.call_count == 1
        first_call = mock_memory.save_text_memory.call_args_list[0]
        ddl = first_call.kwargs.get("content", first_call[1].get("content", ""))
        assert "CREATE TABLE orders" in ddl


class TestTrainAgentOracle:
    """Tests for train_agent with Oracle backend."""

    def test_train_agent_oracle_backend(self):
        mock_memory = _make_mock_memory()
        factory = MagicMock()
        factory.backend = "oracle"
        factory.dsn = "host:1521/sid"
        factory.schema = "PUBLIC"

        conn = MagicMock()
        cursor = MagicMock()
        tables = [("EMPLOYEES", "TABLE")]
        columns = [
            ("EMP_ID", "NUMBER", "N", None, None, 10, 0),
            ("EMP_NAME", "VARCHAR2", "Y", None, 100, None, None),
        ]
        cursor.fetchall = MagicMock(side_effect=[tables, columns])
        conn.cursor.return_value = cursor
        factory.connect.return_value = conn

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, factory))

        assert mock_memory.save_text_memory.call_count == 4
        first_call = mock_memory.save_text_memory.call_args_list[0]
        ddl = first_call.kwargs.get("content", first_call[1].get("content", ""))
        assert "Oracle DSN" in ddl
        assert "CREATE TABLE EMPLOYEES" in ddl


class TestTrainAgentMySQL:
    """Tests for train_agent with MySQL backend."""

    def test_train_agent_mysql_backend(self):
        mock_memory = _make_mock_memory()
        factory = MagicMock()
        factory.backend = "mysql"
        factory.schema = "test_db"

        conn = MagicMock()
        cursor = MagicMock()
        tables = [("products", "BASE TABLE")]
        columns = [
            ("id", "int", "NO", None, None, 11, 0),
            ("name", "varchar", "YES", None, 255, None, None),
        ]
        cursor.fetchall = MagicMock(side_effect=[tables, columns])
        conn.cursor.return_value = cursor
        factory.connect.return_value = conn

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, factory))

        # DDL (1) + Examples (2) + Documentation (1) = 4 calls
        assert mock_memory.save_text_memory.call_count == 4
        first_call = mock_memory.save_text_memory.call_args_list[0]
        ddl = first_call.kwargs.get("content", first_call[1].get("content", ""))
        assert "CREATE TABLE products" in ddl


class TestTrainAgentErrors:
    """Tests for error handling in train_agent."""

    def test_connection_failure_logs_error(self):
        mock_memory = _make_mock_memory()
        factory = MagicMock()
        factory.backend = "snowflake"
        factory.schema = "HUB"
        factory.connect.side_effect = RuntimeError("Connection refused")

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            with pytest.raises(
                RuntimeError, match="Database connection or schema extraction failed"
            ):
                asyncio.run(train_agent(mock_memory, factory))

        # Failed extraction stops the process; no examples or docs saved
        assert mock_memory.save_text_memory.call_count == 0

    def test_unsupported_backend_logs_error(self):
        mock_memory = _make_mock_memory()
        factory = MagicMock()
        factory.backend = "unknown_db"
        factory.schema = "PUBLIC"
        factory.connect.return_value = MagicMock()

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            with pytest.raises(ValueError, match="Unsupported backend for training"):
                asyncio.run(train_agent(mock_memory, factory))

        assert mock_memory.save_text_memory.call_count == 0


class TestTrainAgentTableLimit:
    """Tests for table limit feature."""

    def test_respects_table_limit_env_var(self):
        mock_memory = _make_mock_memory()
        factory = MagicMock()
        factory.backend = "snowflake"
        factory.schema = "HUB"

        conn = MagicMock()
        conn.database = "DB"
        conn.schema = "SCH"
        cursor = MagicMock()

        tables = [(f"TABLE_{i}", "BASE TABLE") for i in range(10)]
        columns = [("COL", "TEXT", "YES", None, None, None, None)]
        views = []
        cursor.fetchall = MagicMock(side_effect=[tables, columns, columns, views])
        conn.cursor.return_value = cursor
        factory.connect.return_value = conn

        with (
            patch.dict(os.environ, {"SNOWFLAKE_TRAIN_TABLE_LIMIT": "2"}),
            patch("src.training.base._build_training_context") as mock_ctx,
        ):
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, factory))

        first_call = mock_memory.save_text_memory.call_args_list[0]
        ddl = first_call.kwargs.get("content", first_call[1].get("content", ""))
        assert "TABLE_0" in ddl
        assert "TABLE_1" in ddl
        assert "TABLE_2" not in ddl


class TestTrainAgentViews:
    """Tests for Snowflake view extraction."""

    def test_extracts_snowflake_views(self):
        mock_memory = _make_mock_memory()
        factory = MagicMock()
        factory.backend = "snowflake"
        factory.schema = "HUB"

        conn = MagicMock()
        conn.database = "DB"
        conn.schema = "SCH"
        cursor = MagicMock()

        tables = [("T1", "BASE TABLE")]
        columns = [("ID", "NUMBER", "NO", None, None, 10, 0)]
        views = [("V_REPORT",)]
        view_columns = [("COL1", "VARCHAR"), ("COL2", "NUMBER")]

        cursor.fetchall = MagicMock(side_effect=[tables, columns, views, view_columns])
        conn.cursor.return_value = cursor
        factory.connect.return_value = conn

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            asyncio.run(train_agent(mock_memory, factory))

        first_call = mock_memory.save_text_memory.call_args_list[0]
        ddl = first_call.kwargs.get("content", first_call[1].get("content", ""))
        assert "V_REPORT" in ddl
        assert "VIEWS" in ddl
