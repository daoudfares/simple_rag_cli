"""
Simple RAG CLI — Interactive RAG agent for databases.

This module is the application entry-point.  All heavy initialisation is
deferred to ``create_app()`` so that the module can be safely imported
without side-effects (useful for testing and tooling).
"""

import argparse
import asyncio
import logging
import os
import shutil
import uuid

from vanna import Agent
from vanna.core.tool.models import ToolContext
from vanna.core.user import RequestContext, User

# Note: We import trainers/connections here to trigger their registration in the registry
import src.database.connections.mysql  # noqa: F401
import src.database.connections.oracle  # noqa: F401
import src.database.connections.postgres  # noqa: F401
import src.database.connections.snowflake  # noqa: F401
import src.training.mysql  # noqa: F401
import src.training.oracle  # noqa: F401
import src.training.postgres  # noqa: F401
import src.training.snowflake  # noqa: F401
from src.config.config_loader import (
    get_available_databases,
    get_available_llms,
    get_database_config,
)
from src.database.database_management import get_db_tool
from src.database.registry import DatabaseRegistry
from src.llm.ai_management import get_agent_memory, get_llm, get_tool_registry
from src.llm.system_prompt import VannaSystemPromptBuilder
from src.security.simple_user_resolver import SimpleUserResolver
from src.services.feedback_manager import FeedbackManager
from src.training.trainer import train_if_needed
from src.ui.formatter import CLIFormatter

logger = logging.getLogger(__name__)


# ============================================================================
# 🏗️ APPLICATION FACTORY
# ============================================================================


def create_app(llm_name: str, db_name: str) -> dict:
    """
    Initialise and return all application components.

    Args:
        llm_name: Name of the LLM profile to use (from secrets.toml).
        db_name: Name of the database profile to use (from secrets.toml).

    Returns a dict with keys: agent, agent_memory, feedback_manager,
    connection_factory.
    """
    # 2. LLM
    llm = get_llm(llm_name)

    # 3. Database config
    db_config = get_database_config(db_name)
    db_type = db_config["type"]

    # 4. Database tool
    db_tool = get_db_tool(db_name)

    # 5. Agent memory (ChromaDB)
    agent_memory = get_agent_memory()

    # 6. Tool registry
    tools = get_tool_registry(db_tool)

    # 7. User resolver
    user_resolver = SimpleUserResolver()

    # 8. System prompt builder (injects database context)
    system_prompt_builder = VannaSystemPromptBuilder(
        db_type=db_type,
        database=db_config["database"],
        schema=db_config.get("schema", ""),
    )

    # 9. Agent
    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=user_resolver,
        agent_memory=agent_memory,
        system_prompt_builder=system_prompt_builder,
    )

    # 10. Feedback manager
    feedback_manager = FeedbackManager(agent_memory)

    # 11. Connection factory (for training) - using Registry
    try:
        connection_factory = DatabaseRegistry.get_connection_factory(db_type, db_config)
    except ValueError as e:
        raise ValueError(f"Error in profile '{db_name}': {e}") from e

    return {
        "agent": agent,
        "agent_memory": agent_memory,
        "feedback_manager": feedback_manager,
        "connection_factory": connection_factory,
    }


# ============================================================================
# 💬 QUERY / FEEDBACK
# ============================================================================


async def query_agent(
    agent: Agent,
    feedback_manager: FeedbackManager,
    question: str,
    user_email: str = "admin@example.com",
    raw: bool = False,
    show_charts: bool = False,
) -> None:
    """Send *question* to the agent and stream the response."""
    request_context = RequestContext(
        cookies={"vanna_email": user_email},
        headers={},
    )

    logger.info("Question: %s", question)
    print(f"\n🔍 Question: {question}")
    print("=" * 80)

    sql_query: str | None = None
    full_response: str = ""
    formatter = CLIFormatter(show_charts=show_charts)

    try:
        async for component in agent.send_message(
            request_context=request_context,
            message=question,
            conversation_id=None,
        ):
            # Unwrap component for formatting and attribute extraction
            actual_component = component
            if hasattr(component, "rich_component") and component.rich_component is not None:
                actual_component = component.rich_component
            elif hasattr(component, "simple_component") and component.simple_component is not None:
                actual_component = component.simple_component

            if raw:
                # In raw mode, we just print the object representation or dict
                print(f"\n📦 RAW COMPONENT: {actual_component}")
                if hasattr(actual_component, "__dict__"):
                    print(f"Data: {actual_component.__dict__}")
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
        print(f"❌ Error: {e}")

    print("=" * 80)


async def handle_feedback(
    agent_memory,
    feedback_manager: FeedbackManager,
    user_email: str = "admin@example.com",
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


# ============================================================================
# 🤖 INTERACTIVE MODE
# ============================================================================


async def interactive_mode(
    agent: Agent,
    agent_memory,
    feedback_manager: FeedbackManager,
    raw: bool = False,
    show_charts: bool = False,
) -> None:
    """Interactive loop for asking questions and providing feedback."""
    print("\n" + "=" * 80)
    print("🤖 SIMPLE RAG CLI INTERACTIVE MODE — WITH FEEDBACK SYSTEM")
    print("=" * 80)
    print("Type your questions (or 'exit', 'quit', 'q' to quit)")
    print("Type 'help' to see available commands")
    print("After each response, you can validate or correct the agent!")
    print("=" * 80 + "\n")

    user_email = "admin@example.com"
    current_show_charts = show_charts

    while True:
        try:
            question = await asyncio.get_event_loop().run_in_executor(None, input, "💬 You: ")
            question = question.strip()

            if question.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye!")
                break

            if question.lower() == "help":
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

            if question.lower() == "charts on":
                current_show_charts = True
                print("✅ Automatic chart display enabled\n")
                continue

            if question.lower() == "charts off":
                current_show_charts = False
                print("❌ Automatic chart display disabled\n")
                continue

            if question.lower().startswith("user "):
                user_email = question.split(" ", 1)[1].strip()
                print(f"✅ User changed: {user_email}\n")
                continue

            if not question:
                continue

            await query_agent(
                agent,
                feedback_manager,
                question,
                user_email,
                raw=raw,
                show_charts=current_show_charts,
            )
            await handle_feedback(agent_memory, feedback_manager, user_email)

        except KeyboardInterrupt:
            print("\n\n👋 Interruption detected. Goodbye!")
            break
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            print(f"\n❌ Unexpected error: {e}\n")


# ============================================================================
# 🚀 MAIN
# ============================================================================


async def main() -> None:
    """Application entry-point: create components, train, run interactive."""
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
        help="LLM profile to use (as defined in secrets.toml)",
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
        help="Display raw component data (JSON-like) instead of formatted output",
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
        memory_path = "chroma_db_vanna"
        if os.path.exists(memory_path):
            logger.info("Wiping agent memory as requested by --refresh...")
            shutil.rmtree(memory_path)

    # Initialise all components
    app = create_app(llm_name=args.llm, db_name=args.database)

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
        raw=args.raw,
        show_charts=args.show_charts,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
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
