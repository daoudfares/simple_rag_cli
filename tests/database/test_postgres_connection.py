"""
Unit tests for the PostgreSQL connection factory.
"""

from unittest.mock import MagicMock

import pytest

from src.database.connections.postgres import PostgresConnectionFactory


class TestPostgresConnectionFactory:
    SAMPLE_CONFIG = {
        "host": "db.example.com",
        "port": "5432",
        "database": "testdb",
        "user": "testuser",
        "password": "secret",
    }

    def test_connect_passes_correct_params(self):
        """connect() should forward host/port/database/user/password to psycopg2.connect."""
        factory = PostgresConnectionFactory(config=self.SAMPLE_CONFIG)

        # make a fake psycopg2 module so the import inside connect() succeeds
        import sys
        import types

        fake = types.SimpleNamespace(connect=MagicMock())
        sys.modules["psycopg2"] = fake

        try:
            factory.connect()
            fake.connect.assert_called_once_with(
                host="db.example.com",
                port=5432,
                database="testdb",
                user="testuser",
                password="secret",
            )
        finally:
            del sys.modules["psycopg2"]

    def test_properties(self):
        factory = PostgresConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.database == "testdb"
        assert factory.host == "db.example.com"
        assert factory.port == 5432

    def test_psycopg2_import_missing_raises(self):
        """connect() should raise ImportError with helpful message if psycopg2 not installed."""
        factory = PostgresConnectionFactory(config=self.SAMPLE_CONFIG)

        import sys

        # Hide psycopg2 by setting it to None
        original_module = sys.modules.get("psycopg2")
        sys.modules["psycopg2"] = None

        try:
            with pytest.raises(ImportError, match="psycopg2 is required"):
                factory.connect()
        finally:
            if original_module:
                sys.modules["psycopg2"] = original_module
            else:
                del sys.modules["psycopg2"]
