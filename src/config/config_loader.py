"""
Centralized configuration loader.

Reads ``secrets.toml`` at the project root and exposes typed helpers
to access each configuration section.

The configuration file uses *named profiles* so that several LLM and
database back-ends can coexist in the same file::

    [llm.lmstudio]
    provider = "local-llm"
    model    = "openai/gpt-oss-120b"
    ...

    [database.snowflake]
    type     = "snowflake"
    account  = "jules-it"
    ...

At runtime the caller picks which profile to use by name (e.g. via CLI
flags ``--llm local-llm --database snowflake``).
"""

import logging
import os
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------


def _find_config_path() -> Path:
    """Return the path to secrets.toml.

    Priority:
    1. ``VANNA_SECRETS_PATH`` environment variable
    2. First ``secrets.toml`` found walking up from cwd
    3. Two levels above this file (project root fallback)
    """
    env = os.getenv("VANNA_SECRETS_PATH")
    if env:
        return Path(env)

    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / "secrets.toml"
        if candidate.exists():
            return candidate

    return Path(__file__).resolve().parents[2] / "secrets.toml"


_CONFIG_PATH = _find_config_path()

# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

_REQUIRED_DB_KEYS: dict[str, set[str]] = {
    "snowflake": {"account", "user", "role", "warehouse", "database", "schema", "private_key_path"},
    "postgresql": {"host", "database", "user", "password"},
    "mysql": {"host", "database", "user", "password"},
    "oracle": {"user", "password", "dsn"},
}

_REQUIRED_LLM_KEYS_BY_PROVIDER: dict[str, set[str]] = {
    "local-llm": {"provider", "model", "api_key", "base_url"},
    "anthropic": {"provider", "model", "api_key"},
    "ollama": {"provider", "model", "base_url"},
    "gemini": {"provider", "model", "api_key"},
}

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Load and return the full configuration (cached after first call)."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {_CONFIG_PATH}\n"
            "Please create secrets.toml at the project root."
        )

    with open(_CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    # Validate all LLM profiles
    for name, block in config.get("llm", {}).items():
        if not isinstance(block, dict):
            continue
        provider = block.get("provider")
        if provider:
            required = _REQUIRED_LLM_KEYS_BY_PROVIDER.get(provider)
            if required is None:
                logger.warning(
                    "Unknown LLM provider '%s' in profile [llm.%s]; valid providers: %s",
                    provider,
                    name,
                    list(_REQUIRED_LLM_KEYS_BY_PROVIDER),
                )
            else:
                missing = required - block.keys()
                if missing:
                    raise ValueError(
                        f"Missing required keys in [llm.{name}] "
                        f"for provider '{provider}': {missing}"
                    )

    # Validate all database profiles
    for name, block in config.get("database", {}).items():
        if not isinstance(block, dict):
            continue
        db_type = block.get("type")
        if not db_type:
            raise ValueError(
                f"Missing required 'type' key in [database.{name}]. "
                f"Valid types: {list(_REQUIRED_DB_KEYS)}"
            )
        required = _REQUIRED_DB_KEYS.get(db_type)
        if required is None:
            logger.warning(
                "Unknown database type '%s' in profile [database.%s]; valid types: %s",
                db_type,
                name,
                list(_REQUIRED_DB_KEYS),
            )
        else:
            missing = required - block.keys()
            if missing:
                raise ValueError(
                    f"Missing required keys in [database.{name}] for type '{db_type}': {missing}"
                )

    return config


# ---------------------------------------------------------------------------
# Public accessors — named profiles
# ---------------------------------------------------------------------------


def get_available_llms() -> list[str]:
    """Return the names of all configured LLM profiles."""
    return list(get_config().get("llm", {}).keys())


def get_available_databases() -> list[str]:
    """Return the names of all configured database profiles."""
    return list(get_config().get("database", {}).keys())


def get_llm_config(name: str) -> dict[str, Any]:
    """Return the configuration dict for LLM profile *name*.

    Raises:
        ValueError: If the profile does not exist.
    """
    profiles = get_config().get("llm", {})
    if name not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"LLM profile '{name}' not found in configuration. Available profiles: {available}"
        )
    return profiles[name]


def get_database_config(name: str) -> dict[str, Any]:
    """Return the configuration dict for database profile *name*.

    Raises:
        ValueError: If the profile does not exist.
    """
    profiles = get_config().get("database", {})
    if name not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"Database profile '{name}' not found in configuration. Available profiles: {available}"
        )
    return profiles[name]
