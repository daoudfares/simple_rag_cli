"""
UI service for Vanna AI CLI.
Uses prompt_toolkit for a modern interactive experience.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from vanna import Agent

from src.services.feedback_manager import FeedbackManager

style = Style.from_dict(
    {
        "prompt": "#00aa00 bold",
        "user": "#ansigreen",
        "agent": "#ansiblue",
    }
)


class VannaUI:
    def __init__(self, agent: Agent, agent_memory, feedback_manager: FeedbackManager):
        self.agent = agent
        self.agent_memory = agent_memory
        self.feedback_manager = feedback_manager
        self.session = PromptSession(history=InMemoryHistory())

    async def run(self):
        print("\n" + "=" * 80)
        print("🤖 VANNA AI INTERACTIVE MODE — MODERN UI")
        print("=" * 80)

        while True:
            try:
                question = await self.session.prompt_async("💬 You: ")
                question = question.strip()

                if not question:
                    continue

                if question.lower() in ("exit", "quit", "q"):
                    print("\n👋 Goodbye!")
                    break

                # Delegate query logic back to app or implement here
                # (For now, keeping it simple to avoid extreme refactoring in one go)
                from app import handle_feedback, query_agent

                await query_agent(self.agent, self.feedback_manager, question)
                await handle_feedback(self.agent_memory, self.feedback_manager)

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"❌ UI Error: {e}")
