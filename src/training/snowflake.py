"""
Snowflake trainer module.

Handles schema extraction, training examples, and documentation specifically
for Snowflake databases.
"""

import logging
import os

import snowflake.connector

from src.database.registry import DatabaseRegistry
from src.training.base import BaseTrainer

logger = logging.getLogger(__name__)

TRAINING_EXAMPLES = [
    {
        "question": "Count the total number of records in each table",
        "sql": "SELECT COUNT(*) AS total_rows FROM {schema}.{table}",
    },
    {
        "question": "Show the first 10 rows of a table",
        "sql": "SELECT * FROM {schema}.{table} LIMIT 10",
    },
    {
        "question": "List all tables in the current schema",
        "sql": (
            "SELECT TABLE_NAME, TABLE_TYPE, ROW_COUNT "
            "FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = CURRENT_SCHEMA() "
            "ORDER BY TABLE_NAME"
        ),
    },
    {
        "question": "Show the structure of a table with its columns",
        "sql": (
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = CURRENT_SCHEMA() AND TABLE_NAME = '<TABLE>' "
            "ORDER BY ORDINAL_POSITION"
        ),
    },
    {
        "question": "Find the distinct values of a column",
        "sql": "SELECT DISTINCT <COLUMN> FROM {schema}.{table} ORDER BY 1",
    },
    {
        "question": "Count the NULL values in each column",
        "sql": (
            "SELECT "
            "COUNT(*) - COUNT(col1) AS col1_nulls, "
            "COUNT(*) - COUNT(col2) AS col2_nulls "
            "FROM {schema}.{table}"
        ),
    },
]

SNOWFLAKE_DOCUMENTATION = """
Snowflake SQL Documentation:
- Snowflake uses standard ANSI SQL.
- Identifiers are case-insensitive unless enclosed in double quotes.
- Use LIMIT instead of TOP to limit results.
- Window functions (OVER, PARTITION BY) are widely supported.
- Use QUALIFY to filter window function results.
- Common data types: NUMBER, VARCHAR, DATE, TIMESTAMP, BOOLEAN, VARIANT (JSON).
- Aggregate functions: COUNT, SUM, AVG, MIN, MAX, LISTAGG.
- Date functions: DATEADD, DATEDIFF, DATE_TRUNC, CURRENT_DATE().
- Use FLATTEN to parse VARIANT / JSON columns.
- CTEs (WITH ... AS) are preferred over nested subqueries.
"""


class SnowflakeTrainer(BaseTrainer):
    """Trainer implementation for Snowflake databases."""

    def extract_schema(self) -> tuple[str, int]:
        """Extract DDL from Snowflake via INFORMATION_SCHEMA."""
        conn = self.connection_factory.connect()
        logger.info("Connection established")
        cursor = conn.cursor()

        table_query = (
            "SELECT TABLE_NAME, TABLE_TYPE"
            " FROM INFORMATION_SCHEMA.TABLES"
            " WHERE TABLE_SCHEMA = CURRENT_SCHEMA()"
            " ORDER BY TABLE_NAME;"
        )
        cursor.execute(table_query)
        tables = cursor.fetchall()
        logger.info("%d tables found", len(tables))

        if self.demo:
            limit = 5
            logger.info("Demo mode active: targeting first %d tables/views", limit)
        else:
            try:
                limit = int(os.environ.get("SNOWFLAKE_TRAIN_TABLE_LIMIT", ""))
            except (ValueError, TypeError):
                limit = 0  # no limit

        if limit > 0 and len(tables) > limit:
            logger.info("Limiting tables to first %d for training (was %d)", limit, len(tables))
            tables = tables[:limit]

        ddl = f"-- Snowflake Database: {conn.database}\n"
        ddl += f"-- Schema: {conn.schema}\n\n"

        for i, (table_name, table_type) in enumerate(tables, 1):
            logger.info("[%d/%d] %s (%s)", i, len(tables), table_name, table_type)

            cursor.execute(
                """
                SELECT
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE,
                    COLUMN_DEFAULT,
                    CHARACTER_MAXIMUM_LENGTH,
                    NUMERIC_PRECISION,
                    NUMERIC_SCALE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
                AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION;
                """,
                (table_name,),
            )
            columns = cursor.fetchall()
            logger.debug("  %d columns", len(columns))

            ddl += f"CREATE TABLE {table_name} (\n"
            column_defs = []

            for col in columns:
                (
                    col_name,
                    data_type,
                    is_nullable,
                    col_default,
                    char_max_len,
                    num_precision,
                    num_scale,
                ) = col

                if data_type == "TEXT" and char_max_len:
                    col_def = f"    {col_name} VARCHAR({char_max_len})"
                elif data_type == "NUMBER" and num_precision:
                    if num_scale:
                        col_def = f"    {col_name} NUMBER({num_precision},{num_scale})"
                    else:
                        col_def = f"    {col_name} NUMBER({num_precision})"
                else:
                    col_def = f"    {col_name} {data_type}"

                if is_nullable == "NO":
                    col_def += " NOT NULL"

                if col_default:
                    col_def += f" DEFAULT {col_default}"

                column_defs.append(col_def)

            ddl += ",\n".join(column_defs)
            ddl += "\n);\n\n"

        # views for snowflake only
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.VIEWS
            WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
            ORDER BY TABLE_NAME;
        """)
        views = cursor.fetchall()
        logger.info("%d views found", len(views))

        if limit > 0 and len(views) > limit:
            logger.info("Limiting views to first %d for training (was %d)", limit, len(views))
            views = views[:limit]

        if views:
            ddl += "\n-- VIEWS --\n\n"
            for (view_name,) in views:
                try:
                    cursor.execute(
                        """
                        SELECT COLUMN_NAME, DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
                        AND TABLE_NAME = %s
                        ORDER BY ORDINAL_POSITION;
                    """,
                        (view_name,),
                    )
                    view_cols = cursor.fetchall()
                    ddl += f"-- View: {view_name}\n"
                    ddl += f"-- Columns: {', '.join(f'{c[0]} ({c[1]})' for c in view_cols)}\n\n"
                    logger.info("  View %s — %d columns", view_name, len(view_cols))
                except snowflake.connector.ProgrammingError as e:
                    logger.debug("Skipping view %s: %s", view_name, e)

        conn.close()
        logger.info("Connection closed")

        return ddl, len(tables)

    async def add_examples(self) -> None:
        """Add question->SQL examples for Snowflake."""
        schema = getattr(self.connection_factory, "schema", "PUBLIC") or "PUBLIC"

        for i, example in enumerate(TRAINING_EXAMPLES, 1):
            sql = example["sql"].replace("{schema}", schema).replace("{table}", "<TABLE>")
            training_content = f"Question: {example['question']}\nSQL: {sql}\nStatus: VALIDATED"
            await self.agent_memory.save_text_memory(content=training_content, context=self.context)
            logger.info('  [%d/%d] Q: "%s"', i, len(TRAINING_EXAMPLES), example["question"])

        logger.info("%d examples saved", len(TRAINING_EXAMPLES))

    async def add_documentation(self) -> None:
        """Add Snowflake documentation."""
        await self.agent_memory.save_text_memory(
            content=SNOWFLAKE_DOCUMENTATION, context=self.context
        )
        logger.info("Documentation saved")


DatabaseRegistry.register_trainer("snowflake", SnowflakeTrainer)
