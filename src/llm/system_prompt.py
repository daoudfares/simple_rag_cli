"""
Custom system prompt builder for Vanna AI agent.

Extends the default prompt builder to inject database context
(backend type, database name, schema) so the LLM generates
SQL in the correct dialect.
"""

from typing import TYPE_CHECKING

from vanna.core.system_prompt import DefaultSystemPromptBuilder

if TYPE_CHECKING:
    from vanna.core.tool.models import ToolSchema
    from vanna.core.user import User


class VannaSystemPromptBuilder(DefaultSystemPromptBuilder):
    """System prompt builder that injects database context.

    Ensures the LLM knows which database backend it is connected to
    (e.g. Snowflake, PostgreSQL) so it generates SQL in the correct dialect.
    """

    def __init__(
        self,
        db_type: str,
        database: str,
        schema: str = "",
    ) -> None:
        super().__init__()
        self._db_type = db_type
        self._database = database
        self._schema = schema

    async def build_system_prompt(self, user: "User", tools: list["ToolSchema"]) -> str | None:
        """Build a system prompt with database context prepended."""
        base_prompt = await super().build_system_prompt(user, tools)

        db_context_parts = [
            "",
            "=" * 60,
            "DATABASE CONTEXT:",
            "=" * 60,
            f"You are connected to a **{self._db_type.upper()}** database.",
            f"Database: {self._database}",
        ]

        if self._schema:
            db_context_parts.append(f"Schema: {self._schema}")

        db_context_parts.extend(
            [
                "",
                f"IMPORTANT: You MUST generate SQL in the **{self._db_type.upper()}** dialect.",
                "Do NOT use syntax from other databases (SQLite, MySQL, PostgreSQL, etc.).",
            ]
        )

        if self._db_type == "snowflake":
            db_context_parts.extend(
                [
                    "",
                    "Snowflake-specific reminders:",
                    "- Use INFORMATION_SCHEMA.TABLES to list tables",
                    "- Use INFORMATION_SCHEMA.COLUMNS to list columns of a table",
                    "- Use CURRENT_SCHEMA() for the current schema",
                    "- Use LIMIT (not TOP) to limit results",
                    "- Use QUALIFY for window function filters",
                    "- Use FLATTEN for JSON / VARIANT columns",
                    "",
                    "CRITICAL Snowflake INFORMATION_SCHEMA limitations:",
                    "- NEVER use INFORMATION_SCHEMA.KEY_COLUMN_USAGE — it does NOT exist in Snowflake",
                    "- NEVER use INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE — it does NOT exist in Snowflake",
                    "- NEVER use INFORMATION_SCHEMA.TABLE_CONSTRAINTS — it does NOT exist in Snowflake",
                    "- NEVER use INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS — it does NOT exist in Snowflake",
                    "- To find primary keys, use: SHOW PRIMARY KEYS IN SCHEMA "
                    f"{self._database}.{self._schema}"
                    if self._schema
                    else f"{self._database}",
                    "- To find foreign keys, use: SHOW IMPORTED KEYS IN SCHEMA "
                    f"{self._database}.{self._schema}"
                    if self._schema
                    else f"{self._database}",
                    "",
                    "SQL execution rules:",
                    "- Do NOT use USE DATABASE or USE SCHEMA commands — the connection context is already set",
                    "- Only generate SELECT or SHOW statements — never DDL (CREATE/ALTER/DROP) or DML (INSERT/UPDATE/DELETE)",
                    "- Do NOT send multiple statements separated by semicolons in a single call",
                    "- If a query fails, try a DIFFERENT approach — do NOT retry the same query",
                ]
            )
        elif self._db_type == "postgresql":
            db_context_parts.extend(
                [
                    "",
                    "PostgreSQL-specific reminders:",
                    "- Use information_schema.tables or pg_catalog for metadata",
                    "- Use LIMIT / OFFSET for pagination",
                ]
            )
        elif self._db_type == "mysql":
            db_context_parts.extend(
                [
                    "",
                    "MySQL-specific reminders:",
                    "- Use information_schema.tables for metadata",
                    "- Use LIMIT for pagination",
                    "- Use backticks for reserved word identifiers",
                ]
            )
        elif self._db_type == "oracle":
            db_context_parts.extend(
                [
                    "",
                    "Oracle-specific reminders:",
                    "- Use USER_TABLES / ALL_TABLES for metadata",
                    "- Use ROWNUM or FETCH FIRST N ROWS for pagination",
                ]
            )

        db_context = "\n".join(db_context_parts)

        return f"{base_prompt}\n{db_context}"
