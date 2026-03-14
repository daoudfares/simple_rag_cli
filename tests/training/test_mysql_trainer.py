"""
Tests for MySQL schema extraction.
"""

from unittest.mock import MagicMock, patch

from vanna.tools.agent_memory import AgentMemory

from src.training.mysql import MySQLTrainer


class TestMySQLTrainer:
    @patch("src.database.connections.mysql.MySQLConnectionFactory")
    def test_extract_schema_mysql(self, mock_factory_cls):
        # Mock connection and cursor
        mock_factory = mock_factory_cls.return_value
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_factory.connect.return_value = mock_conn
        mock_conn.database = "mysql_db"

        # Mock tables result
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = [
            None,  # table_query
            None,  # column_query for table1
        ]
        mock_cursor.fetchall.side_effect = [
            [("table1", "BASE TABLE")],  # tables
            [("id", "int", "NO", "NULL", None, 11, 0)],  # columns
        ]

        # Use a spec'd mock for agent_memory to avoid Pydantic issues
        agent_memory = MagicMock(spec=AgentMemory)
        trainer = MySQLTrainer(agent_memory, mock_factory)

        # Patch _build_training_context to avoid further pydantic issues inside it
        with patch("src.training.base._build_training_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            ddl, count = trainer.extract_schema()

        assert count == 1
        assert "CREATE TABLE table1" in ddl
        assert "id int NOT NULL" in ddl
        assert mock_conn.close.called
