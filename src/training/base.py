"""
Base trainer module for Vanna AI CLI.

Provides the abstract BaseTrainer class which defines the common workflow
for agent training across all database backends.
"""

import abc
import logging
import uuid

from vanna.core.tool.models import ToolContext
from vanna.core.user import User
from vanna.integrations.chromadb import ChromaAgentMemory

from src.config.constants import DEFAULT_USER_EMAIL

logger = logging.getLogger(__name__)


def _build_training_context(agent_memory: ChromaAgentMemory) -> ToolContext:
    """Create a ToolContext for training operations."""
    user = User(
        id=DEFAULT_USER_EMAIL,
        email=DEFAULT_USER_EMAIL,
        group_memberships=["admin"],
    )
    return ToolContext(
        user=user,
        conversation_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        agent_memory=agent_memory,
    )


class BaseTrainer(abc.ABC):
    """Abstract base class for database-specific trainers."""

    def __init__(self, agent_memory: ChromaAgentMemory, connection_factory, demo: bool = False):
        self.agent_memory = agent_memory
        self.connection_factory = connection_factory
        self.demo = demo
        self.context = _build_training_context(agent_memory)

    async def train(self) -> None:
        """
        Train the agent by extracting the schema and adding examples.

        Steps:
            1. Extract DDL via specific database implementation
            2. Add question→SQL training examples
            3. Add database documentation
        """
        logger.info("=" * 80)
        logger.info("📚 AGENT TRAINING")
        logger.info("=" * 80)

        # ── Step 1: Database schema extraction ────────────────────────────────────
        logger.info("STEP 1/3 — Extracting database schema")
        backend = getattr(self.connection_factory, "backend", "unknown")
        logger.info("Using backend: %s", backend)

        try:
            ddl, num_tables = self.extract_schema()

            logger.info("Saving DDL to memory (%d chars)", len(ddl))
            await self.agent_memory.save_text_memory(content=ddl, context=self.context)
            logger.info("Schema saved — %d tables", num_tables)

        except Exception as e:
            logger.error("Failed to extract schema: %s", e)
            raise RuntimeError(f"Database connection or schema extraction failed: {e}") from e

        # ── Step 2: Question→SQL examples ────────────────────────────────────────
        logger.info("STEP 2/3 — Adding question→SQL examples")
        await self.add_examples()

        # ── Step 3: Database documentation ───────────────────────────────────────
        logger.info("STEP 3/3 — Adding database documentation")
        await self.add_documentation()

        logger.info("=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)

    @abc.abstractmethod
    def extract_schema(self) -> tuple[str, int]:
        """
        Extract DDL from the database and return it with the number of tables.

        Returns:
            Tuple[str, int]: A tuple containing the DDL string and the number of tables.
        """
        pass

    @abc.abstractmethod
    async def add_examples(self) -> None:
        """Add training examples to the agent memory."""
        pass

    @abc.abstractmethod
    async def add_documentation(self) -> None:
        """Add documentation about the database SQL dialect to the memory."""
        pass
