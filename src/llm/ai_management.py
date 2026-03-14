"""
AI component factories.

Provides factory functions for the LLM service, agent memory (ChromaDB),
and tool registry used by the Vanna AI agent.
"""

import logging
from typing import Any

from vanna.core.registry import ToolRegistry
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.tools import VisualizeDataTool
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsTool,
    SaveTextMemoryTool,
    SearchSavedCorrectToolUsesTool,
)

from src.config.config_loader import get_llm_config
from src.llm.providers.anthropic import AnthropicLLM

# local llm type classes
from src.llm.providers.base import BaseLLM
from src.llm.providers.gemini import GeminiLLM
from src.llm.providers.local_llm import LocalLLM
from src.llm.providers.ollama import OllamaLLM

logger = logging.getLogger(__name__)


class RobustVisualizeDataTool(VisualizeDataTool):
    """
    A robust version of VisualizeDataTool that handles errors gracefully.
    Instead of raising an exception when visualization fails, it returns
    a helpful message as a string.
    """

    async def execute(self, **kwargs) -> Any:
        try:
            return await super().execute(**kwargs)
        except Exception as e:
            logger.error("Visualization tool failed: %s", e)
            # Return a friendly error message instead of crashing
            return f"Note: Could not generate a chart for this data. {str(e)}"


def get_llm(llm_name: str) -> BaseLLM:
    """
    Return an instance of a subclass of :class:`BaseLLM` configured
    according to the ``[llm.<llm_name>]`` section of the application config.

    Args:
        llm_name: Name of the LLM profile to use (must exist in secrets.toml).

    Raises:
        ValueError: If the profile does not exist or has an unknown provider.
    """
    llm_config = get_llm_config(llm_name)
    provider = llm_config.get("provider")
    model = llm_config.get("model")
    base_url = llm_config.get("base_url")
    api_key = llm_config.get("api_key")

    if not provider or not model:
        raise ValueError(f"LLM profile '{llm_name}' must specify both 'provider' and 'model'")

    logger.info(
        "Initialising LLM '%s' with provider: %s, model: %s",
        llm_name,
        provider,
        model,
    )

    if provider == "local-llm":
        logger.info("Initialising Local LLM: model=%s, base_url=%s", model, base_url)
        return LocalLLM(model=model, base_url=base_url, api_key=api_key)
    elif provider == "anthropic":
        logger.info("Initialising Anthropic LLM: model=%s", model)
        return AnthropicLLM(model=model, api_key=api_key)
    elif provider == "ollama":
        logger.info("Initialising Ollama LLM: model=%s, base_url=%s", model, base_url)
        return OllamaLLM(model=model, base_url=base_url)
    elif provider == "gemini":
        logger.info("Initialising Gemini LLM: model=%s", model)
        return GeminiLLM(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM provider '{provider}' in profile '{llm_name}'")


def get_agent_memory(
    collection_name: str = "vanna_memory",
    persist_directory: str = "./chroma_db_vanna",
) -> ChromaAgentMemory:
    """Return the ChromaDB agent memory instance.

    Args:
        collection_name: Name of the ChromaDB collection (default: ``vanna_memory``).
        persist_directory: Path to persist the ChromaDB data (default: ``./chroma_db_vanna``).
    """
    return ChromaAgentMemory(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def get_tool_registry(db_tool: Any) -> ToolRegistry:
    """Register and return the tool registry with all available tools."""
    tools = ToolRegistry()
    tools.register_local_tool(db_tool, access_groups=["admin", "user"])
    tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=["admin"])
    tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=["admin", "user"])
    tools.register_local_tool(SaveTextMemoryTool(), access_groups=["admin", "user"])
    tools.register_local_tool(RobustVisualizeDataTool(), access_groups=["admin", "user"])
    logger.info("Tool registry initialised with %d tools", 5)
    return tools
