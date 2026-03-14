"""
Unit tests for AI management factory functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.ai_management import get_agent_memory, get_tool_registry


class TestGetLlm:
    """Tests for the LLM factory with named profiles."""

    def test_local_llm_profile(self):
        """get_llm() should instantiate Local LLM from named profile."""
        with (
            patch("src.llm.ai_management.get_llm_config") as mock_config,
            patch("src.llm.ai_management.LocalLLM") as MockClass,
        ):
            mock_config.return_value = {
                "provider": "local-llm",
                "model": "test-model",
                "base_url": "http://test:9999/v1",
                "api_key": "test-key",
            }
            MockClass.return_value = MagicMock()
            from src.llm.ai_management import get_llm

            llm = get_llm("lmstudio")
            mock_config.assert_called_once_with("lmstudio")
            MockClass.assert_called_once_with(
                model="test-model",
                base_url="http://test:9999/v1",
                api_key="test-key",
            )
            assert llm is MockClass.return_value

    def test_anthropic_profile(self):
        """get_llm() should instantiate Anthropic LLM from named profile."""
        with (
            patch("src.llm.ai_management.get_llm_config") as mock_config,
            patch("src.llm.ai_management.AnthropicLLM") as mock_class,
        ):
            mock_config.return_value = {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "api_key": "sk-ant-test",
            }
            mock_class.return_value = MagicMock()
            from src.llm.ai_management import get_llm

            llm = get_llm("anthropic")
            mock_config.assert_called_once_with("anthropic")
            mock_class.assert_called_once()
            assert llm is mock_class.return_value

    def test_ollama_profile(self):
        """get_llm() should instantiate Ollama LLM from named profile."""
        with (
            patch("src.llm.ai_management.get_llm_config") as mock_config,
            patch("src.llm.ai_management.OllamaLLM") as mock_class,
        ):
            mock_config.return_value = {
                "provider": "ollama",
                "model": "llama2",
                "base_url": "http://localhost:11434",
            }
            mock_class.return_value = MagicMock()
            from src.llm.ai_management import get_llm

            llm = get_llm("ollama")
            mock_config.assert_called_once_with("ollama")
            mock_class.assert_called_once()
            assert llm is mock_class.return_value

    def test_gemini_profile(self):
        """get_llm() should instantiate Gemini LLM from named profile."""
        with (
            patch("src.llm.ai_management.get_llm_config") as mock_config,
            patch("src.llm.ai_management.GeminiLLM") as mock_class,
        ):
            mock_config.return_value = {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "api_key": "test-gemini-key",
            }
            mock_class.return_value = MagicMock()
            from src.llm.ai_management import get_llm

            llm = get_llm("gemini")
            mock_config.assert_called_once_with("gemini")
            mock_class.assert_called_once_with(model="gemini-2.0-flash", api_key="test-gemini-key")
            assert llm is mock_class.return_value

    def test_unknown_provider_raises(self):
        """A profile with an unknown provider should raise ValueError."""
        with patch("src.llm.ai_management.get_llm_config") as mock_config:
            mock_config.return_value = {
                "provider": "not-a-real-service",
                "model": "foo",
            }
            from src.llm.ai_management import get_llm

            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm("bad_profile")

    def test_unknown_profile_raises(self):
        """get_llm() should raise ValueError for non-existent profile name."""
        with patch("src.llm.ai_management.get_llm_config", side_effect=ValueError("not found")):
            from src.llm.ai_management import get_llm

            with pytest.raises(ValueError, match="not found"):
                get_llm("nonexistent")

    def test_missing_provider_or_model_raises(self):
        """Profile without provider or model should raise ValueError."""
        with patch("src.llm.ai_management.get_llm_config") as mock_config:
            mock_config.return_value = {"provider": "ollama"}
            from src.llm.ai_management import get_llm

            with pytest.raises(ValueError, match="must specify both"):
                get_llm("incomplete")


class TestGetAgentMemory:
    """Tests for the agent memory factory."""

    def test_returns_chroma_memory(self):
        """get_agent_memory() should return a ChromaAgentMemory instance."""
        memory = get_agent_memory()
        from vanna.integrations.chromadb import ChromaAgentMemory

        assert isinstance(memory, ChromaAgentMemory)


class TestGetToolRegistry:
    """Tests for the tool registry factory."""

    def test_returns_tool_registry(self):
        """get_tool_registry() should return a ToolRegistry."""
        mock_db = MagicMock()
        registry = get_tool_registry(mock_db)
        from vanna.core.registry import ToolRegistry

        assert isinstance(registry, ToolRegistry)
