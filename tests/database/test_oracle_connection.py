"""
Unit tests for the Oracle connection factory.
"""

from unittest.mock import MagicMock

import pytest

from src.database.connections.oracle import OracleConnectionFactory


class TestOracleConnectionFactory:
    SAMPLE_CONFIG = {
        "user": "orauser",
        "password": "opass",
        "dsn": "host:1521/sid",
        "database": "orcl",
    }

    def test_connect_passes_correct_params(self):
        factory = OracleConnectionFactory(config=self.SAMPLE_CONFIG)
        import sys
        import types

        fake = types.SimpleNamespace(connect=MagicMock())
        sys.modules["oracledb"] = fake

        try:
            factory.connect()
            fake.connect.assert_called_once_with(
                user="orauser",
                password="opass",
                dsn="host:1521/sid",
            )
        finally:
            del sys.modules["oracledb"]

    def test_dsn_property(self):
        factory = OracleConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.dsn == "host:1521/sid"

    def test_database_property(self):
        factory = OracleConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.database == "orcl"

    def test_oracledb_import_missing_raises(self):
        """connect() should raise ImportError with helpful message if oracledb not installed."""
        factory = OracleConnectionFactory(config=self.SAMPLE_CONFIG)

        import sys

        original_module = sys.modules.get("oracledb")
        sys.modules["oracledb"] = None

        try:
            with pytest.raises(ImportError, match="oracledb is required"):
                factory.connect()
        finally:
            if original_module:
                sys.modules["oracledb"] = original_module
            else:
                del sys.modules["oracledb"]
