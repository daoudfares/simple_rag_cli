"""
Feedback manager.

Tracks the last agent interaction and persists positive / negative
feedback into ChromaDB for continuous learning.
"""

import logging

from vanna.core.tool.models import ToolContext
from vanna.integrations.chromadb import ChromaAgentMemory

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
    ) -> None:
        """Store the last interaction for later feedback."""
        self.last_interaction = {
            "question": question,
            "sql": sql,
            "response": response,
        }

    async def save_positive_feedback(self, context: ToolContext) -> None:
        """Save the last interaction as a validated example."""
        if not self.last_interaction:
            logger.warning("No interaction to save")
            print("⚠️ No interaction to save")
            return

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
        print("✅ Positive feedback saved! The agent will learn from this example.")

    async def save_negative_feedback(
        self, context: ToolContext, correction: str | None = None
    ) -> None:
        """Save the last interaction as an incorrect example."""
        if not self.last_interaction:
            logger.warning("No interaction to correct")
            print("⚠️ No interaction to correct")
            return

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
        print("❌ Negative feedback saved. The agent will avoid this approach.")
