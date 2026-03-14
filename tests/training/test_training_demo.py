"""
Verification test for demo mode table limiting.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.training.mysql import MySQLTrainer
from src.training.oracle import OracleTrainer
from src.training.postgres import PostgresTrainer


class TestDemoMode:
    @pytest.fixture
    def mock_agent_memory(self):
        return MagicMock()

    @pytest.fixture
    def mock_connection_factory(self):
        factory = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        factory.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.database = "test_db"
        return factory, mock_conn, mock_cursor

    def test_postgres_trainer_demo_limit(self, mock_agent_memory, mock_connection_factory):
        factory, conn, cursor = mock_connection_factory
        # Mock 10 tables
        tables = [(f"table_{i}", "BASE TABLE") for i in range(10)]
        cursor.fetchall.side_effect = [tables] + [[]] * 5  # 5 tables processed

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            trainer = PostgresTrainer(mock_agent_memory, factory, demo=True)
            ddl, count = trainer.extract_schema()

        # Should be limited to 5
        assert count == 5
        assert "table_0" in ddl
        assert "table_4" in ddl
        assert "table_5" not in ddl

    def test_mysql_trainer_demo_limit(self, mock_agent_memory, mock_connection_factory):
        factory, conn, cursor = mock_connection_factory
        # Mock 10 tables
        tables = [(f"table_{i}", "BASE TABLE") for i in range(10)]
        cursor.fetchall.side_effect = [tables] + [[]] * 5

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            trainer = MySQLTrainer(mock_agent_memory, factory, demo=True)
            ddl, count = trainer.extract_schema()

        assert count == 5
        assert "table_4" in ddl
        assert "table_5" not in ddl

    def test_oracle_trainer_demo_limit(self, mock_agent_memory, mock_connection_factory):
        factory, conn, cursor = mock_connection_factory
        # Mock 10 tables
        tables = [(f"table_{i}", "TABLE") for i in range(10)]
        cursor.fetchall.side_effect = [tables] + [[]] * 5

        factory.dsn = "test_dsn"
        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            trainer = OracleTrainer(mock_agent_memory, factory, demo=True)
            ddl, count = trainer.extract_schema()

        assert count == 5
        assert "table_4" in ddl
        assert "table_5" not in ddl

    def test_trainer_no_demo_limit(self, mock_agent_memory, mock_connection_factory):
        factory, conn, cursor = mock_connection_factory
        # Mock 10 tables
        tables = [(f"table_{i}", "BASE TABLE") for i in range(10)]
        cursor.fetchall.side_effect = [tables] + [[]] * 10

        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            trainer = PostgresTrainer(mock_agent_memory, factory, demo=False)
            ddl, count = trainer.extract_schema()

        # Should NOT be limited
        assert count == 10
        assert "table_9" in ddl
