"""
Integration tests for the Vanna AI CLI interactive loops and application entry point.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app


class TestCLIIntegration:
    """Integration test cases for the command line application flows."""

    @pytest.mark.asyncio
    @patch("app.get_available_llms", return_value=["test_llm"])
    @patch("app.get_available_databases", return_value=["test_db"])
    @patch("argparse.ArgumentParser.parse_args")
    @patch("app.create_app")
    @patch("app.train_if_needed", new_callable=AsyncMock)
    @patch("app.interactive_mode", new_callable=AsyncMock)
    async def test_main_startup_flow(
        self,
        mock_interactive,
        mock_train,
        mock_create_app,
        mock_parse_args,
        mock_get_dbs,
        mock_get_llms,
    ):
        """Test that main() correctly wires the CLI args and starts interactive mode."""

        args = MagicMock()
        args.llm = "test_llm"
        args.database = "test_db"
        args.refresh = False
        args.demo = False
        args.raw = False
        mock_parse_args.return_value = args

        mock_app = {
            "agent": MagicMock(),
            "agent_memory": MagicMock(),
            "feedback_manager": MagicMock(),
            "connection_factory": MagicMock(),
        }
        mock_create_app.return_value = mock_app

        await app.main()

        mock_create_app.assert_called_once_with(llm_name="test_llm", db_name="test_db")
        mock_train.assert_awaited_once_with(
            mock_app["agent_memory"], mock_app["connection_factory"], demo=args.demo
        )
        mock_interactive.assert_awaited_once_with(
            agent=mock_app["agent"],
            agent_memory=mock_app["agent_memory"],
            feedback_manager=mock_app["feedback_manager"],
            raw=args.raw,
            show_charts=args.show_charts,
        )

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["What is the total sales?", "exit"])
    @patch("app.query_agent", new_callable=AsyncMock)
    @patch("app.handle_feedback", new_callable=AsyncMock)
    async def test_interactive_mode_flow(self, mock_handle_feedback, mock_query_agent, mock_input):
        """Test the interactive loop processes questions and terminates on 'exit'."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_feedback = MagicMock()

        await app.interactive_mode(mock_agent, mock_memory, mock_feedback)

        # User asked one question before exiting
        mock_query_agent.assert_awaited_once_with(
            mock_agent,
            mock_feedback,
            "What is the total sales?",
            "admin@example.com",
            raw=False,
            show_charts=False,
        )
        mock_handle_feedback.assert_awaited_once_with(
            mock_memory, mock_feedback, "admin@example.com"
        )

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["user test@example.com", "Show me data", "q"])
    @patch("app.query_agent", new_callable=AsyncMock)
    @patch("app.handle_feedback", new_callable=AsyncMock)
    async def test_interactive_mode_user_switching(
        self, mock_handle_feedback, mock_query_agent, mock_input
    ):
        """Test the interactive loop allows switching the active user."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_feedback = MagicMock()

        await app.interactive_mode(mock_agent, mock_memory, mock_feedback)

        mock_query_agent.assert_awaited_once_with(
            mock_agent,
            mock_feedback,
            "Show me data",
            "test@example.com",
            raw=False,
            show_charts=False,
        )

    @pytest.mark.asyncio
    async def test_query_agent_interactions(self, capsys):
        """Test that query_agent yields responses from the agent and stores feedback."""
        mock_agent = MagicMock()
        mock_feedback_manager = MagicMock()

        # Mock the stream of components returned by the agent
        async def mock_send_message(*args, **kwargs):
            class Component:
                def __init__(self, text=None, sql=None):
                    if text is not None:
                        self.text = text
                    if sql is not None:
                        self.sql = sql

                def __str__(self):
                    return getattr(self, "text", "") or getattr(self, "sql", "")

            yield Component(text="This is an ")
            yield Component(text="answer based on: ")
            yield Component(sql="SELECT * FROM table")

        mock_agent.send_message = mock_send_message

        await app.query_agent(
            agent=mock_agent,
            feedback_manager=mock_feedback_manager,
            question="Show test?",
            user_email="test@user.com",
        )

        # Verify the feedback manager recorded the interaction
        mock_feedback_manager.store_interaction.assert_called_once_with(
            question="Show test?",
            sql="SELECT * FROM table",
            response="This is an answer based on: ",
        )

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["ok"])
    @patch("app.ToolContext", return_value=MagicMock())
    async def test_handle_feedback_positive(self, mock_tool_context, mock_input):
        """Test that handle_feedback correctly processes positive feedback."""
        mock_memory = MagicMock()
        mock_feedback_manager = AsyncMock()

        await app.handle_feedback(mock_memory, mock_feedback_manager, "test@example.com")

        mock_feedback_manager.save_positive_feedback.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["no", "The fix is x=1"])
    @patch("app.ToolContext", return_value=MagicMock())
    async def test_handle_feedback_negative_with_correction(self, mock_tool_context, mock_input):
        """Test that handle_feedback correctly processes negative feedback and captures correction."""
        mock_memory = MagicMock()
        mock_feedback_manager = AsyncMock()

        await app.handle_feedback(mock_memory, mock_feedback_manager, "test@example.com")

        mock_feedback_manager.save_negative_feedback.assert_awaited_once()
        # Verify the context and the correction were passed
        call_args = mock_feedback_manager.save_negative_feedback.await_args
        assert call_args[0][1] == "The fix is x=1"
