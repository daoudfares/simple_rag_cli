"""
Unit tests for the MySQL connection factory.
"""

from unittest.mock import MagicMock

import pytest

from src.database.connections.mysql import MySQLConnectionFactory


class TestMySQLConnectionFactory:
    SAMPLE_CONFIG = {
        "host": "mysql.host",
        "port": "3306",
        "database": "mysqldb",
        "user": "mysqluser",
        "password": "mpass",
    }

    def test_connect_passes_correct_params(self):
        factory = MySQLConnectionFactory(config=self.SAMPLE_CONFIG)
        import sys
        import types

        fake = types.SimpleNamespace(connect=MagicMock())
        sys.modules["pymysql"] = fake

        try:
            factory.connect()
            fake.connect.assert_called_once_with(
                host="mysql.host",
                user="mysqluser",
                password="mpass",
                database="mysqldb",
                port=3306,
            )
        finally:
            del sys.modules["pymysql"]

    def test_database_property(self):
        factory = MySQLConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.database == "mysqldb"

    def test_host_property(self):
        factory = MySQLConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.host == "mysql.host"

    def test_port_property(self):
        factory = MySQLConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.port == 3306

    def test_pymysql_import_missing_raises(self):
        """connect() should raise ImportError with helpful message if pymysql not installed."""
        factory = MySQLConnectionFactory(config=self.SAMPLE_CONFIG)

        import sys

        original_module = sys.modules.get("pymysql")
        sys.modules["pymysql"] = None

        try:
            with pytest.raises(ImportError, match="PyMySQL is required"):
                factory.connect()
        finally:
            if original_module:
                sys.modules["pymysql"] = original_module
            else:
                del sys.modules["pymysql"]
