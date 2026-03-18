"""
Unit tests for the centralized configuration, RSA key management,
and FeedbackManager.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Tests: config_loader
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigLoader:
    """Tests for the centralized TOML config loader with named profiles."""

    def test_config_loads_successfully(self):
        """get_config() should return a dict with 'llm' and 'database' sections."""
        from src.config.config_loader import get_config

        config = get_config()
        assert isinstance(config, dict)
        assert "llm" in config
        assert "database" in config

    def test_get_available_llms_returns_list(self):
        """get_available_llms() should return a non-empty list of profile names."""
        from src.config.config_loader import get_available_llms

        llms = get_available_llms()
        assert isinstance(llms, list)
        assert len(llms) > 0

    def test_get_available_databases_returns_list(self):
        """get_available_databases() should return a non-empty list of profile names."""
        from src.config.config_loader import get_available_databases

        dbs = get_available_databases()
        assert isinstance(dbs, list)
        assert len(dbs) > 0

    def test_get_llm_config_returns_profile(self):
        """get_llm_config(name) should return a dict with provider and model."""
        from src.config.config_loader import get_available_llms, get_llm_config

        llms = get_available_llms()
        if llms:
            config = get_llm_config(llms[0])
            assert isinstance(config, dict)
            assert "provider" in config
            assert "model" in config

    def test_get_database_config_returns_profile(self):
        """get_database_config(name) should return a dict with type."""
        from src.config.config_loader import get_available_databases, get_database_config

        dbs = get_available_databases()
        if dbs:
            config = get_database_config(dbs[0])
            assert isinstance(config, dict)
            assert "type" in config

    def test_get_llm_config_unknown_raises(self):
        """get_llm_config should raise ValueError for unknown profile."""
        from src.config.config_loader import get_llm_config

        with pytest.raises(ValueError, match="not found"):
            get_llm_config("nonexistent_llm_profile")

    def test_get_database_config_unknown_raises(self):
        """get_database_config should raise ValueError for unknown profile."""
        from src.config.config_loader import get_database_config

        with pytest.raises(ValueError, match="not found"):
            get_database_config("nonexistent_db_profile")

    def test_config_file_not_found_raises(self, tmp_path):
        """_load_config should raise FileNotFoundError for a missing TOML."""
        from src.config import config_loader as mod

        original = mod._CONFIG_PATH
        try:
            mod._CONFIG_PATH = tmp_path / "nonexistent.toml"
            with pytest.raises(FileNotFoundError):
                mod._load_config()
        finally:
            mod._CONFIG_PATH = original

    def test_validates_missing_llm_keys(self, tmp_path):
        """_load_config should raise ValueError when required LLM keys are missing."""
        from src.config import config_loader as mod

        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text(
            '[llm.test]\nprovider = "local-llm"\nmodel = "x"\n# missing api_key and base_url\n'
        )

        original = mod._CONFIG_PATH
        try:
            mod._CONFIG_PATH = bad_toml
            with pytest.raises(ValueError, match="Missing required keys in \\[llm.test\\]"):
                mod._load_config()
        finally:
            mod._CONFIG_PATH = original

    def test_validates_missing_db_type(self, tmp_path):
        """_load_config should raise ValueError when database profile has no type."""
        from src.config import config_loader as mod

        bad_toml = tmp_path / "bad2.toml"
        bad_toml.write_text('[database.test]\nhost = "localhost"\n')

        original = mod._CONFIG_PATH
        try:
            mod._CONFIG_PATH = bad_toml
            with pytest.raises(ValueError, match="Missing required 'type' key"):
                mod._load_config()
        finally:
            mod._CONFIG_PATH = original

    def test_validates_missing_db_keys(self, tmp_path):
        """_load_config should raise ValueError when required DB keys are missing."""
        from src.config import config_loader as mod

        bad_toml = tmp_path / "bad3.toml"
        bad_toml.write_text(
            "[database.test]\n"
            'type = "postgresql"\n'
            'host = "localhost"\n'
            "# missing database, user, password\n"
        )

        original = mod._CONFIG_PATH
        try:
            mod._CONFIG_PATH = bad_toml
            with pytest.raises(ValueError, match="Missing required keys in \\[database.test\\]"):
                mod._load_config()
        finally:
            mod._CONFIG_PATH = original

    def test_valid_multi_profile_config(self, tmp_path):
        """_load_config should accept a valid multi-profile configuration."""
        from src.config import config_loader as mod

        good = tmp_path / "good.toml"
        key_file = tmp_path / "k.p8"
        key_file.write_text("dummy")
        good.write_text(
            "[llm.test_ollama]\n"
            'provider = "ollama"\n'
            'model = "llama2"\n'
            'base_url = "http://localhost:11434"\n'
            "\n"
            "[database.test_snowflake]\n"
            'type = "snowflake"\n'
            'account = "a"\n'
            'user = "u"\n'
            'role = "r"\n'
            'warehouse = "w"\n'
            'database = "d"\n'
            'schema = "s"\n'
            'private_key_path = "k.p8"\n'
        )

        original = mod._CONFIG_PATH
        try:
            mod._CONFIG_PATH = good
            cfg = mod._load_config()
            assert cfg["llm"]["test_ollama"]["provider"] == "ollama"
            assert cfg["database"]["test_snowflake"]["type"] == "snowflake"
        finally:
            mod._CONFIG_PATH = original

    def test_lazy_loading_does_not_read_on_import(self, monkeypatch, tmp_path):
        """Importing the module shouldn't hit disk until a get_*_config() function is called."""
        import importlib

        secret = tmp_path / "secrets.toml"
        secret.write_text(
            '[llm.x]\nprovider="ollama"\nmodel="m"\nbase_url="http://x"\n'
            '[database.y]\ntype="snowflake"\naccount="a"\nuser="u"\nrole="r"\n'
            'warehouse="w"\ndatabase="d"\nschema="s"\nprivate_key_path="k"\n'
        )
        monkeypatch.setenv("VANNA_SECRETS_PATH", str(secret))

        called = False

        def fake_load():
            nonlocal called
            called = True
            return {
                "llm": {"x": {"provider": "ollama", "model": "m"}},
                "database": {"y": {"type": "snowflake"}},
            }

        import src.config.config_loader as mod

        mod = importlib.reload(mod)
        monkeypatch.setattr(mod, "_load_config", fake_load)

        assert not called, "_load_config should not run on import"
        _ = mod.get_available_llms()
        assert called

    def test_config_path_can_be_overridden_with_envvar(self, tmp_path, monkeypatch):
        """Setting VANNA_SECRETS_PATH should dictate the file used by the loader."""
        cfg = tmp_path / "my.toml"
        cfg.write_text(
            '[llm.x]\nprovider="ollama"\nmodel="m"\nbase_url="http://x"\n'
            '[database.y]\ntype="snowflake"\naccount="a"\nuser="u"\nrole="r"\n'
            'warehouse="w"\ndatabase="d"\nschema="s"\nprivate_key_path="k"\n'
        )
        monkeypatch.setenv("VANNA_SECRETS_PATH", str(cfg))
        from importlib import reload

        import src.config.config_loader as mod

        mod._config = None
        reload(mod)
        assert cfg == mod._CONFIG_PATH


# ──────────────────────────────────────────────────────────────────────────────
# Tests: key_management
# ──────────────────────────────────────────────────────────────────────────────


class TestKeyManagement:
    """Tests for RSA key loading."""

    def test_missing_key_file_raises(self):
        """load_private_key should raise RSAKeyLoadError if file doesn't exist."""
        from src.security.key_management import RSAKeyLoadError, load_private_key

        with pytest.raises(RSAKeyLoadError):
            load_private_key(key_path="/tmp/nonexistent_key.p8")

    def test_non_rsa_key_raises(self, tmp_path):
        """load_private_key should raise RSAKeyLoadError if key is not RSA."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        from src.security.key_management import RSAKeyLoadError, load_private_key

        ec_key = ec.generate_private_key(ec.SECP256R1())
        pem_data = ec_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        key_file = tmp_path / "ec_key.p8"
        key_file.write_bytes(pem_data)

        with pytest.raises(RSAKeyLoadError, match="not an RSA private key"):
            load_private_key(key_path=str(key_file))

    def test_invalid_pem_raises(self, tmp_path):
        """load_private_key should raise RSAKeyLoadError if PEM is invalid."""
        from src.security.key_management import RSAKeyLoadError, load_private_key

        bad_key = tmp_path / "bad_key.p8"
        bad_key.write_text("This is not a valid PEM key")

        with pytest.raises(RSAKeyLoadError, match="Failed to load private key"):
            load_private_key(key_path=str(bad_key))

    def test_valid_rsa_key_loads(self, tmp_path):
        """load_private_key should successfully load a valid RSA key."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        from src.security.key_management import get_snowflake_key_bytes, load_private_key

        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem_data = rsa_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        key_file = tmp_path / "valid_key.p8"
        key_file.write_bytes(pem_data)

        private_key = load_private_key(key_path=str(key_file))
        assert private_key is not None

        private_key_bytes = get_snowflake_key_bytes(key_path=str(key_file))
        assert isinstance(private_key_bytes, bytes)
        assert len(private_key_bytes) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: FeedbackManager
# ──────────────────────────────────────────────────────────────────────────────


class TestFeedbackManager:
    """Tests for FeedbackManager interaction tracking and feedback."""

    def _make_manager(self):
        mock_memory = MagicMock()
        mock_memory.save_text_memory = AsyncMock()
        from src.services.feedback_manager import FeedbackManager

        return FeedbackManager(agent_memory=mock_memory), mock_memory

    def test_store_interaction(self):
        """store_interaction should save question, sql, and response."""
        manager, _ = self._make_manager()
        manager.store_interaction(
            question="Top 10 artistes",
            sql="SELECT * FROM artists LIMIT 10",
            response="result",
        )
        assert manager.last_interaction is not None
        assert manager.last_interaction["question"] == "Top 10 artistes"
        assert manager.last_interaction["sql"] == "SELECT * FROM artists LIMIT 10"

    def test_positive_feedback_saves_validated(self):
        """save_positive_feedback should save content with Status: VALIDATED."""
        manager, mock_memory = self._make_manager()
        manager.store_interaction(question="Q1", sql="SELECT 1")

        context = MagicMock()
        asyncio.run(manager.save_positive_feedback(context))

        mock_memory.save_text_memory.assert_called_once()
        saved_content = mock_memory.save_text_memory.call_args.kwargs["content"]
        assert "VALIDATED" in saved_content
        assert "Q1" in saved_content

    def test_negative_feedback_saves_incorrect(self):
        """save_negative_feedback should save content with Status: INCORRECT."""
        manager, mock_memory = self._make_manager()
        manager.store_interaction(question="Q2", sql="BAD SQL")

        context = MagicMock()
        asyncio.run(manager.save_negative_feedback(context, correction="GOOD SQL"))

        mock_memory.save_text_memory.assert_called_once()
        saved_content = mock_memory.save_text_memory.call_args.kwargs["content"]
        assert "INCORRECT" in saved_content
        assert "Correct Answer: GOOD SQL" in saved_content

    def test_no_interaction_skips_feedback(self, capsys):
        """Feedback methods should print a warning if no interaction stored."""
        manager, mock_memory = self._make_manager()
        context = MagicMock()

        asyncio.run(manager.save_positive_feedback(context))
        mock_memory.save_text_memory.assert_not_called()

        captured = capsys.readouterr()
        assert "No interaction" in captured.out


# ──────────────────────────────────────────────────────────────────────────────
# Tests: format_training_content
# ──────────────────────────────────────────────────────────────────────────────


class TestFormatTrainingContent:
    """Tests for the shared training content formatter."""

    def test_validated_with_sql(self):
        from src.services.feedback_manager import format_training_content

        result = format_training_content(question="Q", sql="SELECT 1")
        assert "Question: Q" in result
        assert "SQL: SELECT 1" in result
        assert "Status: VALIDATED" in result

    def test_validated_with_response(self):
        from src.services.feedback_manager import format_training_content

        result = format_training_content(question="Q", response="Some text")
        assert "Response: Some text" in result
        assert "Status: VALIDATED" in result

    def test_incorrect_with_correction(self):
        from src.services.feedback_manager import format_training_content

        result = format_training_content(
            question="Q", sql="BAD", status="INCORRECT", correction="GOOD"
        )
        assert "INCORRECT" in result
        assert "Correct Answer: GOOD" in result

    def test_incorrect_without_correction(self):
        from src.services.feedback_manager import format_training_content

        result = format_training_content(question="Q", sql="BAD", status="INCORRECT")
        assert "INCORRECT" in result
        assert "Correct Answer" not in result
