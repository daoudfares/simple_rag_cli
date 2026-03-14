## Project Overview
- **Description:** Interactive Simple RAG CLI agent for querying databases (Snowflake, Postgres, MySQL, Oracle) using natural language.
- **Main Stack:** Python 3.10+, Vanna AI, OpenAI, ChromaDB, Snowflake Connector, Rich, Pytest, Ruff.

## Architecture
- **`src/`:** Contains core business logic organized by domain (`config`, `database`, `llm`, `security`, `services`, `training`).
- **`tests/`:** Test suite categorized by component (`core`, `database`, `integration`, `llm`, `training`).
- **Architectural Patterns:** Heavy usage of the **Factory Pattern** (e.g., `create_app` in `app.py`, connection management in `src/database/connections/`). Highly modular and decoupled architecture.

## Development Commands
- **Install:** `pip install -r requirements.txt` (or `pip install .`)
- **Run (specific profiles):** `python app.py --llm <profile> --database <profile>`
- **Docker:** `docker compose run --rm simple-rag-cli`
- **Lint:** `ruff check .`
- **Test:** `python -m pytest tests/ -v`
- **Coverage:** `python -m pytest tests/ -v --cov=src --cov=app`

## Code Conventions
- **Style:** Managed by `ruff` (max line length 100, target Python 3.10, rule E501 ignored).
- **Typing:** Strict usage of Python type hints (`-> None`, `| None`, etc.).
- **Imports:** Sorted and formatted via `isort` (`known-first-party = ["src"]`).
- **Language:** All code, variables, and names (functions/files) must be in English.

## Key Files & Entry Points
- **`app.py`:** Main entry point (asynchronous interactive loop and application factory).
- **`pyproject.toml`:** Project metadata, system dependencies, and tooling configuration (Ruff, Pytest).
- **`secrets.toml`:** Mandatory configuration file (unversioned) for LLM and DBMS access credentials (template: `secrets.toml.example`).
- **`src/training/trainer.py`:** Handles RAG (ChromaDB vector memory) and AI agent training.
- **`src/config/config_loader.py`:** TOML configuration loading and validation.

## Testing
- **Framework:** `pytest`, `pytest-cov`, and `pytest-asyncio` for the asynchronous local execution loop.
- **Running Tests:** `python -m pytest tests/`

## Important Rules
- **Never commit** `secrets.toml` files or the `.rsa/` directory containing private keys (verify against `.gitignore`).
- **Never use** `docker compose up`; the asynchronous interactive mode requires `docker compose run --rm vanna-cli`.
- **Do not** hardcode or raise specific database exceptions within generic code (use the Factory pattern to isolate Snowflake/Postgres specificities).
- **The selected backend must load** credentials and details from `secrets.toml` rather than `.env` files.
