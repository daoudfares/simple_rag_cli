"""
Tests for System Prompt builder.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.system_prompt import VannaSystemPromptBuilder


@pytest.mark.asyncio
class TestSystemPromptBuilder:
    async def test_build_system_prompt_snowflake(self):
        builder = VannaSystemPromptBuilder(
            db_type="snowflake", database="sf_db", schema="sf_schema"
        )
        user = MagicMock()
        tools = []

        # Mock super().build_system_prompt
        with patch(
            "vanna.core.system_prompt.DefaultSystemPromptBuilder.build_system_prompt",
            new_callable=AsyncMock,
        ) as mock_super:
            mock_super.return_value = "Base prompt"
            prompt = await builder.build_system_prompt(user, tools)

            assert "DATABASE CONTEXT:" in prompt
            assert "SNOWFLAKE" in prompt
            assert "QUALIFY" in prompt
            assert "sf_db" in prompt
            assert "sf_schema" in prompt

    async def test_build_system_prompt_postgres(self):
        builder = VannaSystemPromptBuilder(db_type="postgresql", database="pg_db")
        user = MagicMock()
        tools = []

        with patch(
            "vanna.core.system_prompt.DefaultSystemPromptBuilder.build_system_prompt",
            new_callable=AsyncMock,
        ) as mock_super:
            mock_super.return_value = "Base prompt"
            prompt = await builder.build_system_prompt(user, tools)

            assert "POSTGRESQL" in prompt
            assert "pg_catalog" in prompt

    async def test_build_system_prompt_mysql(self):
        builder = VannaSystemPromptBuilder(db_type="mysql", database="my_db")
        user = MagicMock()
        tools = []

        with patch(
            "vanna.core.system_prompt.DefaultSystemPromptBuilder.build_system_prompt",
            new_callable=AsyncMock,
        ) as mock_super:
            mock_super.return_value = "Base prompt"
            prompt = await builder.build_system_prompt(user, tools)

            assert "MYSQL" in prompt
            assert "backticks" in prompt


