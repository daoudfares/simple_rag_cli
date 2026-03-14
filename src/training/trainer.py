"""
Agent training module.

Handles schema extraction from the database, example loading, and
ChromaDB memory population with a guard against duplicate training.
"""

import logging

from vanna.integrations.chromadb import ChromaAgentMemory

from src.training.mysql import MySQLTrainer
from src.training.oracle import OracleTrainer
from src.training.postgres import PostgresTrainer
from src.training.snowflake import SnowflakeTrainer

logger = logging.getLogger(__name__)


async def train_if_needed(
    agent_memory: ChromaAgentMemory,
    connection_factory,  # any object exposing .connect()
    demo: bool = False,
) -> None:
    """Train the agent only if the memory is empty (guard against duplicates)."""
    try:
        # Vanna 0.1.0 uses a private _get_collection() method
        coll = agent_memory._get_collection()
        existing = coll.count() if coll else 0
    except Exception as e:
        logger.debug("Failed to get collection count: %s", e)
        existing = 0

    if existing > 0:
        logger.info("Memory already populated (%d documents) — skipping training", existing)
        return

    await train_agent(agent_memory, connection_factory, demo=demo)


async def train_agent(
    agent_memory: ChromaAgentMemory,
    connection_factory,
    demo: bool = False,
) -> None:
    """
    Train the agent by routing to the specific database trainer.
    """
    backend = getattr(connection_factory, "backend", "snowflake")

    if backend == "snowflake":
        trainer = SnowflakeTrainer(agent_memory, connection_factory, demo=demo)
    elif backend in ("postgres", "postgresql"):
        trainer = PostgresTrainer(agent_memory, connection_factory, demo=demo)
    elif backend == "mysql":
        trainer = MySQLTrainer(agent_memory, connection_factory, demo=demo)
    elif backend == "oracle":
        trainer = OracleTrainer(agent_memory, connection_factory, demo=demo)
    else:
        raise ValueError(f"Unsupported backend for training: {backend}")

    await trainer.train()
