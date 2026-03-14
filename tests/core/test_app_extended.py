"""
Extended tests for app.py — covering query_agent, handle_feedback,
interactive_mode, and create_app error paths.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class TestQueryAgent:
    """Tests for the query_agent function."""

    def test_query_agent_streams_response(self):
        """query_agent should stream components and store interaction."""
        from app import query_agent

        mock_agent = MagicMock()
        mock_feedback = MagicMock()
        mock_feedback.store_interaction = MagicMock()

        sql_component = SimpleNamespace(sql="SELECT 1", text="Here is the result")
        text_component = SimpleNamespace(text=" with more info")

        async def fake_send_message(**kwargs):
            yield sql_component
            yield text_component

        mock_agent.send_message = fake_send_message

        asyncio.run(query_agent(mock_agent, mock_feedback, "test question"))

        mock_feedback.store_interaction.assert_called_once_with(
            question="test question",
            sql="SELECT 1",
            response="Here is the result with more info",
        )

    def test_query_agent_handles_error(self):
        """query_agent should catch exceptions and log them."""
        from app import query_agent

        mock_agent = MagicMock()
        mock_feedback = MagicMock()

        async def fail_send(**kwargs):
            raise RuntimeError("agent crash")
            yield  # noqa: F841

        mock_agent.send_message = fail_send

        asyncio.run(query_agent(mock_agent, mock_feedback, "fail"))
        mock_feedback.store_interaction.assert_not_called()

    def test_query_agent_no_sql_component(self):
        """query_agent should handle responses without sql attribute."""
        from app import query_agent

        mock_agent = MagicMock()
        mock_feedback = MagicMock()
        mock_feedback.store_interaction = MagicMock()

        text_only = SimpleNamespace(text="Just a response")

        async def fake_send(**kwargs):
            yield text_only

        mock_agent.send_message = fake_send

        asyncio.run(query_agent(mock_agent, mock_feedback, "no sql"))

        mock_feedback.store_interaction.assert_called_once_with(
            question="no sql",
            sql=None,
            response="Just a response",
        )


class TestHandleFeedback:
    """Tests for the handle_feedback function."""

    def _run_feedback(self, *inputs):
        """Helper to run handle_feedback with mocked input."""
        from app import handle_feedback

        mock_memory = MagicMock()
        mock_fm = MagicMock()
        mock_fm.save_positive_feedback = AsyncMock()
        mock_fm.save_negative_feedback = AsyncMock()

        input_iter = iter(inputs)

        async def fake_input(executor, fn, prompt):
            return next(input_iter)

        with patch("asyncio.get_event_loop") as mock_loop, patch("app.ToolContext"):
            loop = MagicMock()
            loop.run_in_executor = fake_input
            mock_loop.return_value = loop

            asyncio.run(handle_feedback(mock_memory, mock_fm))

        return mock_fm

    def test_positive_feedback(self):
        """handle_feedback should call save_positive_feedback for 'ok'."""
        fm = self._run_feedback("ok")
        fm.save_positive_feedback.assert_called_once()

    def test_positive_feedback_oui(self):
        """handle_feedback should accept 'oui' as positive."""
        fm = self._run_feedback("oui")
        fm.save_positive_feedback.assert_called_once()

    def test_negative_feedback_with_correction(self):
        """handle_feedback should call save_negative_feedback with correction for 'non'."""
        fm = self._run_feedback("non", "correct SQL")
        fm.save_negative_feedback.assert_called_once()
        call_args = fm.save_negative_feedback.call_args
        assert call_args[0][1] == "correct SQL"

    def test_skip_feedback(self):
        """handle_feedback should do nothing for 'skip'."""
        fm = self._run_feedback("skip")
        fm.save_positive_feedback.assert_not_called()
        fm.save_negative_feedback.assert_not_called()

    def test_empty_feedback(self):
        """handle_feedback should skip for empty input."""
        fm = self._run_feedback("")
        fm.save_positive_feedback.assert_not_called()
        fm.save_negative_feedback.assert_not_called()

    def test_unrecognised_feedback(self):
        """handle_feedback should skip unrecognised input."""
        fm = self._run_feedback("maybe")
        fm.save_positive_feedback.assert_not_called()
        fm.save_negative_feedback.assert_not_called()

    def test_feedback_error_handling(self):
        """handle_feedback should catch exceptions gracefully."""
        from app import handle_feedback

        mock_memory = MagicMock()
        mock_fm = MagicMock()

        async def raise_error(executor, fn, prompt):
            raise RuntimeError("input error")

        with patch("asyncio.get_event_loop") as mock_loop:
            loop = MagicMock()
            loop.run_in_executor = raise_error
            mock_loop.return_value = loop

            # Should not raise
            asyncio.run(handle_feedback(mock_memory, mock_fm))


class TestInteractiveMode:
    """Tests for the interactive_mode function."""

    def _run_interactive(self, *inputs):
        """Helper to run interactive_mode with mocked input sequence."""
        from app import interactive_mode

        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_fm = MagicMock()

        input_iter = iter(inputs)

        async def fake_input(executor, fn, prompt):
            return next(input_iter)

        with patch("asyncio.get_event_loop") as mock_loop:
            loop = MagicMock()
            loop.run_in_executor = fake_input
            mock_loop.return_value = loop

            asyncio.run(interactive_mode(mock_agent, mock_memory, mock_fm))

    def test_exit_command(self):
        """interactive_mode should exit on 'exit' command."""
        self._run_interactive("exit")

    def test_quit_command(self):
        """interactive_mode should exit on 'quit' command."""
        self._run_interactive("quit")

    def test_q_command(self):
        """interactive_mode should exit on 'q' command."""
        self._run_interactive("q")

    def test_help_command(self, capsys):
        """interactive_mode should show help and loop back."""
        self._run_interactive("help", "quit")
        output = capsys.readouterr().out
        assert "AVAILABLE COMMANDS" in output

    def test_user_command(self, capsys):
        """interactive_mode should change user on 'user <email>'."""
        self._run_interactive("user test@example.com", "q")
        output = capsys.readouterr().out
        assert "test@example.com" in output

    def test_empty_input_skipped(self):
        """interactive_mode should skip empty input."""
        self._run_interactive("", "q")


class TestCreateAppErrors:
    """Tests for create_app error paths with named profiles."""

    def test_create_app_mysql_backend(self):
        """create_app should use MySQL factory with mysql type profile."""
        from app import create_app

        with (
            patch("app.get_llm", return_value=MagicMock()),
            patch("app.get_db_tool", return_value=MagicMock()),
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={
                    "type": "mysql",
                    "host": "h",
                    "user": "u",
                    "password": "p",
                    "database": "db",
                },
            ),
            patch("src.database.registry.DatabaseRegistry.get_connection_factory") as mock_reg,
        ):
            result = create_app(llm_name="x", db_name="my")
            mock_reg.assert_called_once()
            assert "connection_factory" in result

    def test_create_app_oracle_backend(self):
        """create_app should use Oracle factory with oracle type profile."""
        from app import create_app

        with (
            patch("app.get_llm", return_value=MagicMock()),
            patch("app.get_db_tool", return_value=MagicMock()),
            patch("app.get_agent_memory", return_value=MagicMock()),
            patch("app.get_tool_registry", return_value=MagicMock()),
            patch("app.Agent", return_value=MagicMock()),
            patch("app.FeedbackManager", return_value=MagicMock()),
            patch(
                "app.get_database_config",
                return_value={
                    "type": "oracle",
                    "user": "u",
                    "password": "p",
                    "dsn": "host/sid",
                    "database": "orcl",
                },
            ),
            patch("src.database.registry.DatabaseRegistry.get_connection_factory") as mock_reg,
        ):
            result = create_app(llm_name="x", db_name="ora")
            mock_reg.assert_called_once()
            assert "connection_factory" in result
