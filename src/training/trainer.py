"""
Agent training module.

Handles schema extraction from the database, example loading, and
ChromaDB memory population with a guard against duplicate training.
"""

import logging
import uuid
from typing import Any

from vanna.core.tool.models import ToolContext
from vanna.core.user import User
from vanna.integrations.chromadb import ChromaAgentMemory

from src.config.constants import DEFAULT_USER_EMAIL
from src.database.registry import DatabaseRegistry

logger = logging.getLogger(__name__)


async def _is_agent_memory_empty(agent_memory: ChromaAgentMemory) -> bool:
    """
    Check if the agent memory (ChromaDB collection) is empty.

    Uses public API methods to determine if training has been completed.
    Falls back to pessimistic approach (assumes empty) if detection fails.

    Returns:
        True if memory is empty or check failed, False if memory has content.
    """
    # Create a ToolContext for real usage, but handle test mocks gracefully
    context = None
    try:
        user = User(
            id=DEFAULT_USER_EMAIL,
            email=DEFAULT_USER_EMAIL,
            group_memberships=["admin"],
        )
        context = ToolContext(
            user=user,
            conversation_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            agent_memory=agent_memory,
        )
    except (TypeError, ValueError):
        # In tests with mocks, context creation may fail due to validation
        pass

    try:
        # Try to search for any text memories - returns empty list if none exist
        if context:
            memories = await agent_memory.get_recent_text_memories(context=context, limit=1)
        else:
            # For mocks in tests
            memories = await agent_memory.get_recent_text_memories()
        return len(memories) == 0
    except Exception as e:
        logger.debug("Failed to check memory status via get_recent_text_memories: %s", e)

    try:
        # Alternative: try search_text_memories with a dummy query
        if context:
            results = await agent_memory.search_text_memories(
                query="SELECT", context=context, limit=1
            )
        else:
            # For mocks in tests
            results = await agent_memory.search_text_memories(query="SELECT")
        memory_exists = len(results) > 0
        return not memory_exists
    except Exception as e:
        logger.debug("Failed to check memory status via search_text_memories: %s", e)

    # If all checks fail, log warning and return True (assume empty to permit training)
    logger.warning(
        "Could not determine agent memory status; attempting training anyway. "
        "This may result in duplicate training if memory already exists."
    )
    return True


async def train_if_needed(
    agent_memory: ChromaAgentMemory,
    connection_factory: Any,
    demo: bool = False,
) -> None:
    """
    Train the agent only if the memory is empty (guard against duplicates).

    Args:
        agent_memory: ChromaDB-backed agent memory for vector storage.
        connection_factory: Database connection factory with .connect() method.
        demo: If True, trains with demo data; otherwise trains with real schema.
    """
    if not await _is_agent_memory_empty(agent_memory):
        logger.info("Memory already populated — skipping training")
        return

    logger.info("Agent memory is empty; starting training (demo=%s)", demo)
    await train_agent(agent_memory, connection_factory, demo=demo)


async def train_agent(
    agent_memory: ChromaAgentMemory,
    connection_factory: Any,
    demo: bool = False,
) -> None:
    """
    Train the agent by routing to the specific database trainer.

    Args:
        agent_memory: ChromaDB-backed agent memory for storing trained schema/examples.
        connection_factory: Database connection factory with backend type.
        demo: If True, trains with demo data; otherwise trains with real schema.

    Raises:
        ValueError: If the connection factory's backend is not supported.
        Exception: Any exception raised by the trainer during training.
    """
    backend = getattr(connection_factory, "backend", "snowflake")
    logger.info("Starting agent training for backend: %s (demo=%s)", backend, demo)

    try:
        trainer = DatabaseRegistry.get_trainer(backend, connection_factory, agent_memory)
        # Override demo flag (BaseTrainer accepts it in __init__)
        trainer.demo = demo
        await trainer.train()
        logger.info("Agent training completed successfully for backend: %s", backend)
    except ValueError as e:
        logger.error("Unsupported backend for training: %s — %s", backend, e)
        raise ValueError(f"Unsupported backend for training: {backend}") from e
    except Exception as e:
        logger.error("Agent training failed for backend %s: %s", backend, e, exc_info=True)
        raise
