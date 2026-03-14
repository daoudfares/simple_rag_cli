"""
Unit tests for the LLM class hierarchy defined in ``src.models``.
"""

from unittest.mock import MagicMock, patch

from src.llm.providers.anthropic import AnthropicLLM
from src.llm.providers.base import BaseLLM
from src.llm.providers.gemini import GeminiLLM
from src.llm.providers.local_llm import LocalLLM
from src.llm.providers.ollama import OllamaLLM


class TestLLMClasses:
    def test_local_llm_wrapper_initialises_service(self):
        """LocalLLM should construct underlying service with args."""
        with patch("vanna.integrations.openai.OpenAILlmService") as MockService:
            MockService.return_value = MagicMock()
            obj = LocalLLM(model="mymodel", base_url="http://foo", api_key="key")
            MockService.assert_called_once_with(
                model="mymodel", base_url="http://foo", api_key="key"
            )
            assert hasattr(obj, "service")
            assert obj.service is MockService.return_value

    def test_anthropic_wrapper_initialises_service(self):
        with patch("vanna.integrations.anthropic.AnthropicLlmService") as MockService:
            MockService.return_value = MagicMock()
            obj = AnthropicLLM(model="claude", api_key="secret")
            MockService.assert_called_once_with(model="claude", api_key="secret")
            assert obj.service is MockService.return_value

    def test_ollama_wrapper_initialises_service(self):
        with patch("vanna.integrations.ollama.OllamaLlmService") as MockService:
            MockService.return_value = MagicMock()
            obj = OllamaLLM(model="llama2", base_url="http://bar")
            MockService.assert_called_once_with(model="llama2", base_url="http://bar")
            assert obj.service is MockService.return_value

    def test_gemini_wrapper_initialises_service(self):
        with patch("vanna.integrations.google.GeminiLlmService") as MockService:
            MockService.return_value = MagicMock()
            obj = GeminiLLM(model="gemini-2.0-flash", api_key="test-key")
            MockService.assert_called_once_with(model="gemini-2.0-flash", api_key="test-key")
            assert obj.service is MockService.return_value

    def test_base_generate_delegates(self):
        # create a fake subclass to exercise BaseLLM.generate
        class Fake(BaseLLM):
            def __init__(self):
                self.service = MagicMock()

        f = Fake()
        f.service.generate.return_value = "xyz"
        assert f.generate("prompt") == "xyz"
