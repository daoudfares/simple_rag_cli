"""
PostgreSQL trainer module.

Handles schema extraction specifically for PostgreSQL databases.
"""

import logging

from src.database.registry import DatabaseRegistry
from src.training.base import BaseTrainer

logger = logging.getLogger(__name__)


class PostgresTrainer(BaseTrainer):
    """Trainer implementation for PostgreSQL databases."""

    def extract_schema(self) -> tuple[str, int]:
        """Extract DDL from PostgreSQL via INFORMATION_SCHEMA."""
        conn = self.connection_factory.connect()
        logger.info("Connection established")
        try:
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

            if self.demo and len(tables) > 5:
                logger.info("Demo mode: Limiting tables to first 5 (was %d)", len(tables))
                tables = tables[:5]

            ddl = f"-- Database: {getattr(conn, 'database', '')}\n\n"

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
                            col_def = f"    {col_name} NUMERIC({num_precision},{num_scale})"
                        else:
                            col_def = f"    {col_name} NUMERIC({num_precision})"
                    else:
                        col_def = f"    {col_name} {data_type}"

                    if is_nullable == "NO":
                        col_def += " NOT NULL"

                    if col_default:
                        col_def += f" DEFAULT {col_default}"

                    column_defs.append(col_def)

                ddl += ",\n".join(column_defs)
                ddl += "\n);\n\n"

            return ddl, len(tables)
        finally:
            conn.close()
            logger.info("Connection closed")

    async def add_examples(self) -> None:
        """Add question->SQL examples for PostgreSQL."""
        # Generic examples can be added here
        logger.info("No specific examples configured for PostgreSQL yet.")

    async def add_documentation(self) -> None:
        """Add PostgreSQL documentation."""
        logger.info("No specific documentation configured for PostgreSQL yet.")


DatabaseRegistry.register_trainer("postgresql", PostgresTrainer)
DatabaseRegistry.register_trainer("postgres", PostgresTrainer)
