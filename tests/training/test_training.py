"""
Unit tests for the training module.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.training.trainer import train_if_needed


class TestTrainIfNeeded:
    """Tests for the training guard logic."""

    def test_skips_when_memory_populated(self):
        """train_if_needed should skip training when memory has text memories."""
        mock_memory = MagicMock()
        mock_memory.get_recent_text_memories = AsyncMock(return_value=["memory_1", "memory_2"])
        mock_factory = MagicMock()

        with patch("src.training.trainer.train_agent", new_callable=AsyncMock) as mock_train:
            asyncio.run(train_if_needed(mock_memory, mock_factory))
            mock_train.assert_not_called()

    def test_trains_when_memory_empty(self):
        """train_if_needed should call train_agent when memory has no text memories."""
        mock_memory = MagicMock()
        mock_memory.get_recent_text_memories = AsyncMock(return_value=[])
        mock_factory = MagicMock()

        with patch("src.training.trainer.train_agent", new_callable=AsyncMock) as mock_train:
            asyncio.run(train_if_needed(mock_memory, mock_factory))
            mock_train.assert_called_once_with(mock_memory, mock_factory, demo=False)

    def test_trains_when_get_recent_fails_fallback_to_search(self):
        """train_if_needed should fallback to search_text_memories when get_recent_text_memories fails."""
        mock_memory = MagicMock()
        mock_memory.get_recent_text_memories = AsyncMock(side_effect=Exception("method unavailable"))
        mock_memory.search_text_memories = AsyncMock(return_value=[])
        mock_factory = MagicMock()

        with patch("src.training.trainer.train_agent", new_callable=AsyncMock) as mock_train:
            asyncio.run(train_if_needed(mock_memory, mock_factory))
            mock_train.assert_called_once_with(mock_memory, mock_factory, demo=False)

    def test_trains_when_all_checks_fail(self):
        """train_if_needed should train when all memory checks fail (pessimistic approach)."""
        mock_memory = MagicMock()
        mock_memory.get_recent_text_memories = AsyncMock(side_effect=Exception("no method"))
        mock_memory.search_text_memories = AsyncMock(side_effect=Exception("no collection"))
        mock_factory = MagicMock()

        with patch("src.training.trainer.train_agent", new_callable=AsyncMock) as mock_train:
            asyncio.run(train_if_needed(mock_memory, mock_factory))
            # Should train because we can't determine if memory exists
            mock_train.assert_called_once_with(mock_memory, mock_factory, demo=False)

    def test_train_agent_generic_backend(self):
        """train_if_needed should still invoke train_agent with non-snowflake factory."""
        mock_memory = MagicMock()
        mock_memory.get_recent_text_memories = AsyncMock(return_value=[])

        class DummyFactory:
            backend = "mysql"

            def connect(self):
                return MagicMock()

        with patch("src.training.trainer.train_agent", new_callable=AsyncMock) as mock_train:
            dummy = DummyFactory()
            asyncio.run(train_if_needed(mock_memory, dummy))
            # ensure it was invoked; backend-specific behaviour is validated
            # elsewhere so we don't need to compare the exact factory object
            mock_train.assert_called_once()
