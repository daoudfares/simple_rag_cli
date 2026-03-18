"""
Simple RAG CLI — Interactive RAG agent for databases.

This module is the application entry-point.  All heavy initialisation is
deferred to ``create_app()`` so that the module can be safely imported
without side-effects (useful for testing and tooling).
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import uuid

from rich.markdown import Markdown
from rich.panel import Panel
from vanna import Agent
from vanna.core.tool.models import ToolContext
from vanna.core.user import RequestContext, User

from src.config.config_loader import (
    get_available_databases,
    get_available_llms,
    get_database_config,
)
from src.config.constants import DEFAULT_USER_EMAIL
from src.config.paths import CHROMA_DB_DIR
from src.database.database_management import get_db_tool
from src.database.registry import DatabaseRegistry, register_all_builtins
from src.llm.ai_management import get_agent_memory, get_llm, get_tool_registry
from src.llm.system_prompt import VannaSystemPromptBuilder
from src.security.simple_user_resolver import SimpleUserResolver
from src.services.feedback_manager import FeedbackManager
from src.services.question_analyzer import QuestionAnalyzer
from src.training.trainer import train_if_needed
from src.ui.formatter import CLIFormatter
from src.ui.json_serializer import component_to_dict, emit

logger = logging.getLogger(__name__)

# Plugins are lazily registered once, on first create_app() call.
_builtins_registered = False


def _ensure_supported_python() -> None:
    """Fail fast with a clear message on unsupported Python versions."""
    if sys.version_info < (3, 10):  # noqa: UP036
        raise SystemExit(f"Python 3.10+ is required. Detected: {sys.version}")


# ============================================================================
# 🏗️ APPLICATION FACTORY
# ============================================================================


def create_app(llm_name: str, db_name: str, router_llm_name: str | None = None) -> dict:
    """
    Initialise and return all application components.

    Args:
        llm_name: Name of the LLM profile to use for heavy lifting.
        db_name: Name of the database profile to use.
        router_llm_name: Optional name of the LLM profile to use for routing/analysis.

    Returns a dict with keys: agent, agent_memory, feedback_manager,
    connection_factory, analyzer.

    Raises:
        ValueError: If any critical component fails to initialise.
    """
    global _builtins_registered  # noqa: PLW0603
    if not _builtins_registered:
        register_all_builtins()
        _builtins_registered = True

    logger.info(
        "Creating application with llm=%s, db=%s, router_llm=%s",
        llm_name,
        db_name,
        router_llm_name,
    )

    try:
        # 1. LLM
        logger.debug("Initialising main LLM service")
        llm = get_llm(llm_name)

        # 1.b Router LLM (optional, defaults to main LLM)
        logger.debug("Initialising router LLM service")
        router_llm = get_llm(router_llm_name) if router_llm_name else llm

        # 2. Database config
        logger.debug("Loading database configuration for profile: %s", db_name)
        db_config = get_database_config(db_name)
        db_type = db_config["type"]

        # 3. Database tool
        logger.debug("Creating database tool for type: %s", db_type)
        db_tool = get_db_tool(db_name)

        # 4. Agent memory (ChromaDB)
        logger.debug("Initialising agent memory (ChromaDB)")
        agent_memory = get_agent_memory()

        # 5. Tool registry
        logger.debug("Initialising tool registry")
        tools = get_tool_registry(db_tool)

        # 6. User resolver
        logger.debug("Initialising user resolver")
        user_resolver = SimpleUserResolver()

        # 7. System prompt builder (injects database context)
        logger.debug("Creating system prompt builder for database type: %s", db_type)
        system_prompt_builder = VannaSystemPromptBuilder(
            db_type=db_type,
            database=db_config["database"],
            schema=db_config.get("schema", ""),
        )

        # 8. Vanna Agent
        logger.debug("Creating Vanna Agent")
        agent = Agent(
            llm_service=llm,
            tool_registry=tools,
            user_resolver=user_resolver,
            agent_memory=agent_memory,
            system_prompt_builder=system_prompt_builder,
        )

        # 9. Question Analyzer
        logger.debug("Initialising question analyzer")
        analyzer = QuestionAnalyzer(router_llm)

        # 10. Feedback manager
        logger.debug("Initialising feedback manager")
        feedback_manager = FeedbackManager(agent_memory)

        # 11. Connection factory (for training) - using Registry
        logger.debug("Getting connection factory for training")
        try:
            connection_factory = DatabaseRegistry.get_connection_factory(db_type, db_config)
        except ValueError as e:
            logger.error("Failed to create connection factory for profile '%s': %s", db_name, e)
            raise ValueError(f"Error in profile '{db_name}': {e}") from e

        logger.info("Application created successfully (llm=%s, db=%s)", llm_name, db_name)
        return {
            "agent": agent,
            "agent_memory": agent_memory,
            "feedback_manager": feedback_manager,
            "connection_factory": connection_factory,
            "analyzer": analyzer,
        }
    except Exception as e:
        logger.error("Failed to create application: %s", e, exc_info=True)
        raise


# ============================================================================
# 💬 QUERY / FEEDBACK
# ============================================================================


def _unwrap_component(component):
    """Extract the actual renderable component from a Vanna wrapper."""
    if hasattr(component, "rich_component") and component.rich_component is not None:
        return component.rich_component
    if hasattr(component, "simple_component") and component.simple_component is not None:
        return component.simple_component
    return component


async def query_agent(
    agent: Agent,
    feedback_manager: FeedbackManager,
    analyzer: QuestionAnalyzer,
    question: str,
    user_email: str = DEFAULT_USER_EMAIL,
    raw: bool = False,
    show_charts: bool = False,
) -> None:
    """Send *question* to the agent and stream the response."""
    request_context = RequestContext(
        cookies={"vanna_email": user_email},
        headers={},
    )

    logger.info("Question: %s", question)

    if not raw:
        print(f"\n🔍 Question: {question}")
        print("=" * 80)

    # 1. Analyze complexity
    analysis = await analyzer.analyze(question)
    complexity = analysis.get("complexity", "SIMPLE")
    sub_questions = analysis.get("sub_questions", [])

    if raw:
        emit({"type": "analysis", "complexity": complexity, "sub_questions": sub_questions})
    elif complexity == "COMPLEX":
        print(f"🧩 Question COMPLEXE détectée. Décomposition en {len(sub_questions)} sous-questions...")
        for i, sq in enumerate(sub_questions):
            print(f"  {i+1}. {sq}")
        print("-" * 80)
    else:
        print("✨ Question SIMPLE détectée.")
        print("-" * 80)

    if complexity == "COMPLEX" and sub_questions:

        results = []
        formatter = None if raw else CLIFormatter(show_charts=show_charts)

        # Execute sub-questions sequentially to avoid interleaved console output
        for sq in sub_questions:
            sq_response = ""
            sq_sql = None

            if not raw:
                print(f"\n⏳ Processing sub-question: {sq}")

            async for component in agent.send_message(
                request_context=request_context,
                message=sq,
                conversation_id=None,
            ):
                actual_component = _unwrap_component(component)

                if raw:
                    emit(component_to_dict(actual_component))
                else:
                    renderable = formatter.format_component(actual_component)
                    if renderable:
                        formatter.console.print(renderable)

                if hasattr(actual_component, "sql"):
                    sq_sql = actual_component.sql
                if hasattr(actual_component, "text"):
                    sq_response += actual_component.text

            results.append({"question": sq, "response": sq_response, "sql": sq_sql})

        # 2. Synthesize final report
        if not raw:
            print("\n📝 Synthesizing final report...")

        final_report = await analyzer.synthesize(question, results)

        if raw:
            emit({"type": "synthesis", "report": final_report})
        else:
            formatter.console.print(
                Panel(Markdown(final_report), title="[bold]Final Report[/bold]", border_style="green")
            )

        # Store interaction (take first SQL if needed, or handle differently)
        feedback_manager.store_interaction(
            question=question,
            sql=results[0]["sql"] if results else None,
            response=final_report,
        )

    else:
        # SIMPLE path (original logic)
        sql_query: str | None = None
        full_response: str = ""
        formatter = None if raw else CLIFormatter(show_charts=show_charts)

        try:
            async for component in agent.send_message(
                request_context=request_context,
                message=question,
                conversation_id=None,
            ):
                actual_component = _unwrap_component(component)

                if raw:
                    emit(component_to_dict(actual_component))
                else:
                    renderable = formatter.format_component(actual_component)
                    if renderable:
                        formatter.console.print(renderable)

                if hasattr(actual_component, "sql"):
                    sql_query = actual_component.sql

                if hasattr(actual_component, "text"):
                    full_response += actual_component.text

            feedback_manager.store_interaction(
                question=question,
                sql=sql_query,
                response=full_response if full_response else None,
            )

        except Exception as e:
            logger.error("Agent error: %s", e)
            if raw:
                emit({"type": "error", "error": str(e)})
            else:
                print(f"❌ Error: {e}")

    if not raw:
        print("=" * 80)


async def handle_feedback(
    agent_memory,
    feedback_manager: FeedbackManager,
    user_email: str = DEFAULT_USER_EMAIL,
) -> None:
    """Prompt the user for feedback on the last interaction."""
    print("\n📊 Is this response correct?")
    print("  👍 'ok' or 'yes' : Good response (trains the agent)")
    print("  👎 'no' or 'ko' : Bad response (corrects the agent)")
    print("  ⏭️  'skip' or empty : Skip to next")

    try:
        feedback = await asyncio.get_event_loop().run_in_executor(None, input, "💬 Your feedback: ")

        feedback = feedback.strip().lower()

        user = User(id=user_email, email=user_email, group_memberships=["admin"])
        context = ToolContext(
            user=user,
            conversation_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            agent_memory=agent_memory,
        )

        if feedback in ("ok", "oui", "yes", "y", "o"):
            await feedback_manager.save_positive_feedback(context)
        elif feedback in ("non", "no", "ko", "n"):
            print("💡 Do you want to provide the correction? (Empty Enter to skip)")
            correction = await asyncio.get_event_loop().run_in_executor(
                None, input, "✏️  Correction: "
            )
            correction = correction.strip() if correction else None
            await feedback_manager.save_negative_feedback(context, correction)
        elif feedback in ("skip", "s", ""):
            print("⏭️  Feedback ignored")
        else:
            print("⚠️ Unrecognized feedback, ignored")

    except Exception as e:
        logger.error("Feedback error: %s", e)
        print(f"❌ Error during feedback: {e}")


async def handle_raw_feedback(
    cmd: dict,
    agent_memory,
    feedback_manager: FeedbackManager,
    user_email: str = DEFAULT_USER_EMAIL,
) -> None:
    """Process a JSON feedback command in raw mode and emit a JSON acknowledgement.

    Expected command shape::

        {"type": "feedback", "value": "positive"|"negative"|"neutral", "correction": "..."}

    ``correction`` is optional and only meaningful for ``"negative"`` feedback.
    Emits ``{"type": "feedback_ack", "value": "...", "status": "saved"|"skipped"|"error"}``.
    """
    value = cmd.get("value", "neutral")
    correction = cmd.get("correction") or None

    if value not in ("positive", "negative", "neutral"):
        emit({"type": "feedback_ack", "value": value, "status": "error", "error": "Unknown feedback value"})
        return

    if value == "neutral":
        logger.info("Neutral feedback received — no training performed")
        emit({"type": "feedback_ack", "value": "neutral", "status": "skipped"})
        return

    user = User(id=user_email, email=user_email, group_memberships=["admin"])
    context = ToolContext(
        user=user,
        conversation_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        agent_memory=agent_memory,
    )

    try:
        if value == "positive":
            await feedback_manager.save_positive_feedback(context, raw=True)
            emit({"type": "feedback_ack", "value": "positive", "status": "saved"})
        elif value == "negative":
            await feedback_manager.save_negative_feedback(context, correction, raw=True)
            emit({"type": "feedback_ack", "value": "negative", "status": "saved"})
    except Exception as e:
        logger.error("Raw feedback error: %s", e)
        emit({"type": "feedback_ack", "value": value, "status": "error", "error": str(e)})


# ============================================================================
# 🤖 INTERACTIVE MODE
# ============================================================================


async def interactive_mode(
    agent: Agent,
    agent_memory,
    feedback_manager: FeedbackManager,
    analyzer: QuestionAnalyzer,
    raw: bool = False,
    show_charts: bool = False,
) -> None:
    """Interactive loop for asking questions and providing feedback."""
    if not raw:
        print("\n" + "=" * 80)
        print("🤖 SIMPLE RAG CLI INTERACTIVE MODE — WITH FEEDBACK SYSTEM")
        print("=" * 80)
        print("Type your questions (or 'exit', 'quit', 'q' to quit)")
        print("Type 'help' to see available commands")
        print("After each response, you can validate or correct the agent!")
        print("=" * 80 + "\n")

    user_email = DEFAULT_USER_EMAIL
    current_show_charts = show_charts

    while True:
        try:
            if raw:
                # Write prompt to stderr so stdout stays pure JSON
                sys.stderr.write("💬 You: ")
                sys.stderr.flush()
                raw_line = await asyncio.get_event_loop().run_in_executor(None, input)
            else:
                raw_line = await asyncio.get_event_loop().run_in_executor(None, input, "💬 You: ")
            question = raw_line.strip()

            # In raw mode, detect JSON feedback commands before treating as a question.
            if raw and question.startswith("{"):
                try:
                    cmd = json.loads(question)
                    if isinstance(cmd, dict) and cmd.get("type") == "feedback":
                        await handle_raw_feedback(cmd, agent_memory, feedback_manager, user_email)
                        continue
                except json.JSONDecodeError:
                    pass  # Not valid JSON — fall through and treat as a question.

            if question.lower() in ("exit", "quit", "q"):
                if not raw:
                    print("\n👋 Goodbye!")
                break

            if not raw and question.lower() == "help":
                print("\n📖 AVAILABLE COMMANDS:")
                print("  - Ask any question about the database")
                print("  - 'user <email>' : Switch user")
                print("  - 'charts on/off' : Enable/disable automatic chart display")
                print("  - 'help' : Display this help message")
                print("  - 'exit', 'quit', 'q' : Quit")
                print("\n💡 FEEDBACK SYSTEM:")
                print("  After each response, indicate if it is correct:")
                print("  - 'ok', 'yes' : Good response → The agent learns")
                print("  - 'no', 'ko' : Bad response → The agent corrects itself")
                print("  - 'skip' or empty : Skip without feedback")
                print()
                continue

            if not raw and question.lower() == "charts on":
                current_show_charts = True
                print("✅ Automatic chart display enabled\n")
                continue

            if not raw and question.lower() == "charts off":
                current_show_charts = False
                print("❌ Automatic chart display disabled\n")
                continue

            if not raw and question.lower().startswith("user "):
                user_email = question.split(" ", 1)[1].strip()
                print(f"✅ User changed: {user_email}\n")
                continue

            if not question:
                continue

            await query_agent(
                agent,
                feedback_manager,
                analyzer,
                question,
                user_email,
                raw=raw,
                show_charts=current_show_charts,
            )

            if not raw:
                await handle_feedback(agent_memory, feedback_manager, user_email)

        except KeyboardInterrupt:
            if not raw:
                print("\n\n👋 Interruption detected. Goodbye!")
            break
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            if raw:
                emit({"type": "error", "error": str(e)})
            else:
                print(f"\n❌ Unexpected error: {e}\n")


# ============================================================================
# 🚀 MAIN
# ============================================================================


async def main() -> None:
    """Application entry-point: create components, train, run interactive."""
    _ensure_supported_python()

    available_llms = get_available_llms()
    available_dbs = get_available_databases()

    parser = argparse.ArgumentParser(
        description="Simple RAG CLI - Interactive RAG agent for databases",
        epilog=(
            f"Available LLM profiles: {', '.join(available_llms) or '(none)'}  |  "
            f"Available database profiles: {', '.join(available_dbs) or '(none)'}"
        ),
    )
    parser.add_argument(
        "--llm",
        required=True,
        choices=available_llms,
        help="Main LLM profile for data extraction (smarter/more expensive)",
    )
    parser.add_argument(
        "--router-llm",
        choices=available_llms,
        help="Optional lighter LLM profile for routing and planning (cheaper/faster)",
    )
    parser.add_argument(
        "--database",
        required=True,
        choices=available_dbs,
        help="Database profile to use (as defined in secrets.toml)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Wipe agent memory and force re-training",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Limit training to the first 5 tables",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Machine-to-machine mode: emit one JSON object per line (JSON Lines / NDJSON) instead of formatted output",
    )
    parser.add_argument(
        "--show-charts",
        action="store_true",
        help="Automatically open generated charts in the default web browser",
    )
    args = parser.parse_args()

    logger.info("LLM profile: %s  |  Database profile: %s", args.llm, args.database)

    # Force refresh if requested
    if args.refresh:
        memory_path = CHROMA_DB_DIR
        if memory_path.exists():
            logger.info("Wiping agent memory as requested by --refresh...")
            shutil.rmtree(memory_path)

    # Initialise all components
    app = create_app(
        llm_name=args.llm,
        db_name=args.database,
        router_llm_name=args.router_llm
    )

    # Train agent (skips if memory already populated)
    # If --refresh was used, the memory is empty, so this will re-train
    logger.info("Checking agent training…")
    await train_if_needed(app["agent_memory"], app["connection_factory"], demo=args.demo)
    logger.info("Agent ready")

    # Start interactive mode
    await interactive_mode(
        agent=app["agent"],
        agent_memory=app["agent_memory"],
        feedback_manager=app["feedback_manager"],
        analyzer=app["analyzer"],
        raw=args.raw,
        show_charts=args.show_charts,
    )


def main_sync() -> None:
    """Synchronous wrapper for ``main()`` (used as console_scripts entry-point)."""
    log_level = os.getenv("SIMPLE_RAG_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Program stopped.")
    except Exception as e:
        logger.critical("Fatal error: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main_sync()
