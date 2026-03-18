"""
Feedback manager.

Tracks the last agent interaction and persists positive / negative
feedback into ChromaDB for continuous learning.
"""

import logging

from vanna.core.tool.models import ToolContext
from vanna.integrations.chromadb import ChromaAgentMemory

from src.config.constants import DEFAULT_USER_EMAIL
from src.services.audit_logger import log_interaction

logger = logging.getLogger(__name__)


def format_training_content(
    question: str,
    sql: str | None = None,
    response: str | None = None,
    *,
    status: str = "VALIDATED",
    correction: str | None = None,
) -> str:
    """Build a standardised training-content string for ChromaDB storage."""
    if status == "INCORRECT":
        content = f"Question: {question}\nIncorrect Response: {sql or response}\nStatus: INCORRECT"
        if correction:
            content += f"\nCorrect Answer: {correction}"
    else:
        if sql:
            content = f"Question: {question}\nSQL: {sql}\nStatus: VALIDATED"
        else:
            content = f"Question: {question}\nResponse: {response}\nStatus: VALIDATED"
    return content


class FeedbackManager:
    """Manages user feedback and agent continuous learning."""

    def __init__(self, agent_memory: ChromaAgentMemory) -> None:
        self.agent_memory: ChromaAgentMemory = agent_memory
        self.last_interaction: dict | None = None

    def store_interaction(
        self,
        question: str,
        sql: str | None = None,
        response: str | None = None,
        user_email: str = DEFAULT_USER_EMAIL,
    ) -> None:
        """Store the last interaction for later feedback."""
        self.last_interaction = {
            "question": question,
            "sql": sql,
            "response": response,
        }

        # Keep an operational audit trail without impacting user flow on I/O errors.
        try:
            log_interaction(
                user_email=user_email,
                question=question,
                sql=sql,
                response=response,
            )
        except Exception as e:
            logger.warning("Failed to write audit log entry: %s", e)

    async def save_positive_feedback(self, context: ToolContext, *, raw: bool = False) -> None:
        """Save the last interaction as a validated example.

        Args:
            context: Vanna ToolContext for memory operations.
            raw: When True, suppress stdout prints (caller handles JSON output).

        Returns:
            None (prints feedback to user via stdout)
        """
        if not self.last_interaction:
            logger.warning("No interaction to save as positive feedback")
            if not raw:
                print("⚠️ No interaction to save")
            return

        try:
            interaction = self.last_interaction
            training_content = format_training_content(
                question=interaction["question"],
                sql=interaction["sql"],
                response=interaction["response"],
                status="VALIDATED",
            )

            await self.agent_memory.save_text_memory(
                content=training_content,
                context=context,
            )

            logger.info("Positive feedback saved for: %s", interaction["question"])
            if not raw:
                print("✅ Positive feedback saved! The agent will learn from this example.")
        except Exception as e:
            logger.error("Failed to save positive feedback: %s", e, exc_info=True)
            if not raw:
                print(f"❌ Error saving feedback: {e}")
            raise

    async def save_negative_feedback(
        self, context: ToolContext, correction: str | None = None, *, raw: bool = False
    ) -> None:
        """Save the last interaction as an incorrect example.

        Args:
            context: Vanna ToolContext for memory operations.
            correction: Optional correction/expected answer.
            raw: When True, suppress stdout prints (caller handles JSON output).

        Returns:
            None (prints feedback to user via stdout)
        """
        if not self.last_interaction:
            logger.warning("No interaction to correct with negative feedback")
            if not raw:
                print("⚠️ No interaction to correct")
            return

        try:
            interaction = self.last_interaction
            training_content = format_training_content(
                question=interaction["question"],
                sql=interaction["sql"],
                response=interaction["response"],
                status="INCORRECT",
                correction=correction,
            )

            await self.agent_memory.save_text_memory(
                content=training_content,
                context=context,
            )

            logger.info("Negative feedback saved for: %s", interaction["question"])
            if not raw:
                print("❌ Negative feedback saved. The agent will avoid this approach.")
        except Exception as e:
            logger.error("Failed to save negative feedback: %s", e, exc_info=True)
            if not raw:
                print(f"❌ Error saving feedback: {e}")
            raise
