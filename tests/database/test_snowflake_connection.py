"""
Unit tests for SnowflakeConnectionFactory.
"""

from unittest.mock import patch

import pytest

from src.database.connections.snowflake import SnowflakeConnectionFactory


class TestSnowflakeConnectionFactory:
    """Tests for the Snowflake connection factory."""

    SAMPLE_CONFIG = {
        "account": "test-account",
        "user": "test@user.com",
        "role": "TEST_ROLE",
        "warehouse": "TEST_WH",
        "database": "TEST_DB",
        "schema": "TEST_SCHEMA",
        "private_key_path": "dummy_key.p8",
    }

    def test_connect_passes_correct_params(self):
        """connect() should pass all config fields to snowflake.connector.connect."""
        factory = SnowflakeConnectionFactory(
            config=self.SAMPLE_CONFIG,
        )

        try:
            with (
                patch(
                    "src.database.connections.snowflake.get_snowflake_key_bytes",
                    return_value=b"fake-key",
                ),
                patch("snowflake.connector.connect") as mock_conn,
            ):
                factory.connect()

            mock_conn.assert_called_once_with(
                account="test-account",
                user="test@user.com",
                private_key=b"fake-key",
                database="TEST_DB",
                schema="TEST_SCHEMA",
                warehouse="TEST_WH",
                role="TEST_ROLE",
            )
        except ImportError:
            pytest.skip("snowflake-connector-python not installed")

    def test_database_property(self):
        """database property should return the config database."""
        factory = SnowflakeConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.database == "TEST_DB"

    def test_schema_property(self):
        """schema property should return the config schema."""
        factory = SnowflakeConnectionFactory(self.SAMPLE_CONFIG)
        assert factory.schema == "TEST_SCHEMA"
