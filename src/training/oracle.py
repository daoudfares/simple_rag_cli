"""
Oracle trainer module.

Handles schema extraction specifically for Oracle databases.
"""

import logging

from src.database.registry import DatabaseRegistry
from src.training.base import BaseTrainer

logger = logging.getLogger(__name__)


class OracleTrainer(BaseTrainer):
    """Trainer implementation for Oracle databases."""

    def extract_schema(self) -> tuple[str, int]:
        """Extract DDL from Oracle via USER_TABLES and USER_TAB_COLUMNS."""
        conn = self.connection_factory.connect()
        logger.info("Connection established")
        try:
            cursor = conn.cursor()

            # Oracle uses USER_TABLES; TYPE is always 'TABLE'
            table_query = "SELECT table_name, 'TABLE' FROM user_tables ORDER BY table_name"
            cursor.execute(table_query)
            tables = cursor.fetchall()
            logger.info("%d tables found", len(tables))

            if self.demo and len(tables) > 5:
                logger.info("Demo mode: Limiting tables to first 5 (was %d)", len(tables))
                tables = tables[:5]

            ddl = f"-- Oracle DSN: {self.connection_factory.dsn}\n\n"

            for i, (table_name, table_type) in enumerate(tables, 1):
                logger.info("[%d/%d] %s (%s)", i, len(tables), table_name, table_type)

                col_query = (
                    "SELECT column_name, data_type, nullable, data_default, "
                    "data_length, data_precision, data_scale "
                    "FROM user_tab_columns WHERE table_name = :1 "
                    "ORDER BY column_id"
                )
                cursor.execute(col_query, (table_name,))
                columns = cursor.fetchall()

                # normalize to tuple length 7
                columns = [
                    (
                        col[0],  # name
                        col[1],  # type
                        "NO" if col[2] == "N" else "YES",
                        col[3],
                        col[4],
                        col[5],
                        col[6],
                    )
                    for col in columns
                ]

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

            return ddl, len(tables)
        finally:
            conn.close()
            logger.info("Connection closed")

    async def add_examples(self) -> None:
        """Add question->SQL examples for Oracle."""
        examples = [
            {"question": "Show all tables", "sql": "SELECT table_name FROM user_tables;"},
            {
                "question": "Show first 5 rows",
                "sql": "SELECT * FROM {table} FETCH FIRST 5 ROWS ONLY;",
            },
        ]
        for example in examples:
            sql = example["sql"].replace("{table}", "<TABLE>")
            content = f"Question: {example['question']}\nSQL: {sql}\nStatus: VALIDATED"
            await self.agent_memory.save_text_memory(content=content, context=self.context)
        logger.info("%d examples saved", len(examples))

    async def add_documentation(self) -> None:
        """Add Oracle documentation."""
        doc = "Oracle specific: Use ROWNUM or FETCH FIRST, quotes for case-sensitivity."
        await self.agent_memory.save_text_memory(content=doc, context=self.context)
        logger.info("Documentation saved")


DatabaseRegistry.register_trainer("oracle", OracleTrainer)
