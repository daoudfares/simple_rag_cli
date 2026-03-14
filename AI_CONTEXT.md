## Project Overview
- **Name:** Simple RAG CLI
- **Description:** Interactive RAG CLI agent for querying databases using natural language.
- **Main Business Objective:** Enable intuitive database querying via LLMs and refine RAG results over time through continuous learning via user feedback.
- **Full Tech Stack:** Python 3.10+, Vanna AI, OpenAI/Local LLMs, ChromaDB, Pytest, Ruff. Supported DBMS: Snowflake, PostgreSQL, MySQL, Oracle.

## Architecture
- **Folder Structure:**
  - `src/config/`: Loading and validation of TOML configurations.
  - `src/database/`: Database connection management (using factory instances).
  - `src/llm/`: Tool registry definition, system prompts, and memory setup (ChromaDB).
  - `src/security/`: Access management and private keys (e.g., Snowflake RSA keys).
  - `src/services/` & `src/training/`: RAG loop logic and DB feedback persistence.
  - `tests/`: Unit and integration testing suite.
- **Architectural Patterns:** Modular, highly decoupled utilizing the Factory Pattern (e.g., `ConnectionFactory` to uniformize disparate DBMS support).
- **Primary Data Flow:** CLI Input → LLM Agent → RAG Execution (Schema/ChromaDB Search) → DBMS Query → CLI Output → User Feedback → ChromaDB Learning update.

## Development Setup
- **Prerequisites:** Python 3.10+
- **Step-by-step Installation:**
  1. `python -m venv .venv` then `source .venv/bin/activate`
  2. `pip install -r requirements.txt` (or install extras via: `pip install .[postgres,mysql,oracle]`)
  3. `cp secrets.toml.example secrets.toml` (and edit it manually with real credentials).
- **Required Environment Variables:** None explicitly via `.env` or system. All configuration is strictly required inside `secrets.toml` (minimum 1 `[llm.profile]` and 1 `[database.profile]`).

## Development Commands
- **Run (Local):** `python app.py --llm <llm_profile> --database <db_profile>`
- **Run (Docker):** `docker compose run --rm simple-rag-cli`
- **Test:** `python -m pytest tests/ -v`
- **Coverage:** `python -m pytest tests/ -v --cov=src --cov=app`
- **Lint (Ruff):** `ruff check .`

## Code Conventions
- **Naming:** Variables/functions in `snake_case`, Classes in `PascalCase`. English everywhere (code, variables, files, logs).
- **Observed Style:** Max line length of 100 characters. Ruff rules target E, F, I, W, UP, B, SIM, but specifically ignore E501.
- **Imports Organization:** Managed automatically via `isort` module (`known-first-party = ["src"]`).
- **Preferred Patterns:** Strictly typed (`-> None`, `| None`). Asynchronous flow (`async`/`await`) for CLI interaction and agent queries. DBMS specificities are strictly hidden behind abstract classes/factory interfaces.

## Key Files & Entry Points
- **App Entry Point:** `app.py` (CLI flow orchestration, async `interactive_mode` loop).
- **Critical Config Files:** `pyproject.toml` (metadata, test/lint tooling), `secrets.toml` (credentials, never versioned).
- **Do Not Modify:** Never manually alter the `.rsa/` folder (locally generated proxy keys) or the generated vector db output `chroma_db_vanna/`.

## Testing Strategy
- **Framework(s):** `pytest` along with `pytest-cov` and `pytest-asyncio`.
- **Test Types:** Isolated unit tests (with DBMS mocking) and integration tests (workflow testing).
- **Commands & Expected Coverage:** Local tests are configured to enforce a minimal operational code coverage of 70% (`fail_under = 70`).

## Domain & Business Logic
- **Key Domain Entities/Concepts:** Vanna `Agent`, `ToolRegistry`, Vector `AgentMemory`, `FeedbackManager`.
- **Important Business Rules:** 
  - Every user prompt goes through a strict workflow: Input -> Action -> Review (OK/KO).
  - An "OK" feedback results in a persistent save of the Q/A pair to assist the Agent later inside `chroma_db_vanna/`.
- **Glossary:** `RAG` = Retrieval-Augmented Generation ; `Profile` = Stanza/blocks prefixed by `[xxx]` inside `secrets.toml`.

## Do's & Don'ts
- **Always Do:** Specify explicitly the CLI arguments to choose both the DBMS engine and the LLM instance. Secure tokens and credentials via the TOML config exclusively.
- **Never Do:** Hardcode specific tokens and credentials. Use `.env` file for the main DB connection string. Run the app via `docker compose up` (it will bypass STDIN support required by the interactive CLI).
- **Frequent Errors to Avoid:** Adding a custom `try/except` block for a specific DB connector inside the global app workflow (always delegate specific error handling to the suitable Factory in `src/database/connections/`).
