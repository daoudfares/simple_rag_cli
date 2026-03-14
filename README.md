# 🤖 Simple RAG CLI

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)
![Snowflake](https://img.shields.io/badge/Snowflake-Ready-blue.svg?logo=snowflake)

Conversational command-line agent to query databases (Snowflake, PostgreSQL, MySQL, Oracle) in natural language.

## 🚀 Quickstart

1. **Install dependencies:**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration:**
   Copy `secrets.toml.example` to `secrets.toml` at the root and edit it. Define your LLM and Database profiles:

   ```toml
   [llm.my_llm]
   provider = "local-llm"
   model = "model-name"
   base_url = "http://localhost:1234/v1"
   api_key = "your-api-key"

   [database.my_db]
   type = "snowflake"
   account = "your-account"            
   user = "your-email@company.com"    
   private_key_path = ".rsa/rsa_key.p8"
   role = "YOUR_ROLE"                 
   warehouse = "YOUR_WAREHOUSE"      
   database = "YOUR_DATABASE"      
   schema = "YOUR_SCHEMA"           
   ```

3. **Launch:**
   Run the app by specifying the profiles defined in your `secrets.toml`:
   ```bash
   python app.py --llm local-llm --database snowflake
   ```

## 🐳 Docker Alternative

```bash
docker compose build
docker compose run --rm simple-rag-cli --llm local-llm --database snowflake
```

## 💬 Usage

- **Ask questions**: Type queries in natural language.
- **Provide feedback**: Answer `ok`/`no`/`skip` to help the agent learn after each query.
- **Commands**: Type `help` for options or `exit` to quit.

## 🧪 Tests

Use `pytest` to run the test suite. Make sure your virtual environment is activated.

- **Run all tests (unit and integration):**
  ```bash
  python -m pytest tests/ -v
  ```

- **Run unit tests only:**
  ```bash
  python -m pytest tests/core tests/database tests/llm tests/training -v
  ```

- **Run integration tests only:**
  ```bash
  python -m pytest tests/integration -v
  ```

## 🏗️ Architecture & Design Patterns

The project follows a clean, decoupled architecture enforcing the **RAG (Retrieval-Augmented Generation)** pipeline. Key design patterns used:

- **Factory Pattern:** Dynamically instantiates the correct database connection (`BaseConnectionFactory` and subclasses like `SnowflakeConnectionFactory`).
- **Proxy / Adapter Pattern:** Uniformizes LLM access (`BaseLLM`) by silently forwarding method calls to the underlying LLM services.
- **Dependency Injection:** Key dependencies (ChromaDB memory, LLM service, tool registries) are injected into the Vanna `Agent` during bootstrap (`app.py`), ensuring modularity and testability.
- **Strategy Pattern:** The agent's backend behavior (LLM provider and Database engine) changes dynamically at runtime based on user configuration.
