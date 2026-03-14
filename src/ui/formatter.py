from typing import Any

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class CLIFormatter:
    """Formats Vanna components for beautiful CLI output using rich."""

    def __init__(self, console: Console = None, show_charts: bool = False):
        self.console = console or Console()
        self.show_charts = show_charts
        self.active_tasks = {}
        self.live_progress = None
        # Map types to formatter methods
        self.handlers = {
            "status_bar_update": self._handle_status_bar,
            "task_tracker_update": self._handle_task_tracker,
            "status_card": self._handle_status_card,
            "dataframe": self._handle_dataframe,
            "text": self._handle_text,
            "card": self._handle_card,
            "chart": self._handle_chart,
            "notification": self._handle_notification,
        }

    def format_component(self, component: Any) -> RenderableType | None:
        """
        Main entry point to format a Vanna component.
        Returns a rich renderable or None if the component shouldn't be printed directly.
        """
        # Unwrap component if it's a wrapper object (common in some Vanna versions)
        actual_component = component

        # If outer component doesn't have 'type' but has 'rich_component', unwrap it
        if not hasattr(component, "type") or getattr(component, "type", None) is None:
            if hasattr(component, "rich_component") and component.rich_component is not None:
                actual_component = component.rich_component
            elif hasattr(component, "simple_component") and component.simple_component is not None:
                actual_component = component.simple_component

        comp_type = getattr(actual_component, "type", None)
        if not comp_type:
            # If it's a string or something else without a 'type' attribute
            if isinstance(actual_component, str):
                return Text(actual_component)
            return Text(str(actual_component), style="dim")

        # Check if type is an enum or string
        type_str = str(comp_type.value) if hasattr(comp_type, "value") else str(comp_type)

        handler = self.handlers.get(type_str)

        if handler:
            return handler(actual_component)

        return Text(f"Unknown component type: {type_str}", style="yellow italic")

    def _handle_status_bar(self, component: Any) -> None:
        # We might want to use console.status for this globally,
        # but for now we'll just log it or ignore if redundant
        message = getattr(component, "message", "")
        status = getattr(component, "status", "")
        if message:
            self.console.print(f"[{status}] {message}", style="blue dim")
        return None

    def _handle_task_tracker(self, component: Any) -> None:
        operation = getattr(component, "operation", "")
        # operation is likely an enum like TaskOperation.ADD_TASK
        op_str = operation.value if hasattr(operation, "value") else str(operation)

        task = getattr(component, "task", None)
        task_id = getattr(component, "task_id", None)

        if op_str == "add_task" and task:
            self.console.print(
                f"⏳ Starting: [bold]{task.title}[/bold] - {task.description or ''}", style="cyan"
            )
        elif op_str == "update_task":
            status = getattr(component, "status", "")
            detail = getattr(component, "detail", "")
            if status == "completed":
                self.console.print(
                    f"✅ Completed task {task_id or ''} {f': {detail}' if detail else ''}",
                    style="green",
                )
        return None

    def _handle_status_card(self, component: Any) -> RenderableType:
        title = getattr(component, "title", "Status")
        status = getattr(component, "status", "info")
        description = getattr(component, "description", "")
        icon = getattr(component, "icon", "ℹ️")

        color = "green" if status == "success" else "yellow" if status == "running" else "blue"

        return Panel(
            Text(f"{icon} {description}", style=color),
            title=f"[bold]{title}[/bold]",
            border_style=color,
        )

    def _handle_dataframe(self, component: Any) -> RenderableType:
        rows = getattr(component, "rows", [])
        columns = getattr(component, "columns", [])
        title = getattr(component, "title", "Query Results")

        table = Table(
            title=title, show_header=True, header_style="bold magenta", border_style="dim"
        )

        for col in columns:
            table.add_column(col)

        for row in rows:
            table.add_row(*[str(row.get(col, "")) for col in columns])

        return table

    def _handle_text(self, component: Any) -> RenderableType:
        content = getattr(component, "content", "")
        is_markdown = getattr(component, "markdown", False)

        if is_markdown:
            return Panel(Markdown(content), border_style="bright_blue")
        else:
            return Panel(Text(content), border_style="bright_blue")

    def _handle_card(self, component: Any) -> RenderableType:
        title = getattr(component, "title", "")
        content = getattr(component, "content", "")
        icon = getattr(component, "icon", "📇")

        return Panel(
            Markdown(content) if getattr(component, "markdown", False) else Text(content),
            title=f"{icon} {title}",
            border_style="white",
        )

    def _handle_chart(self, component: Any) -> RenderableType:
        import os
        import webbrowser
        from datetime import datetime

        import plotly.io as pio

        title = getattr(component, "title", "Chart")
        data = getattr(component, "data", {})

        if not self.show_charts:
            return Panel(
                Text.assemble(
                    ("📊 ", "yellow"),
                    (f"Chart generated: {title}\n", "bold"),
                    ("Note: Automatic display is disabled. Type ", "dim"),
                    ("'charts on'", "bold cyan"),
                    (" to enable.", "dim"),
                ),
                border_style="yellow",
            )

        try:
            # Create exports directory if it doesn't exist
            export_dir = os.path.join(os.getcwd(), "exports", "charts")
            os.makedirs(export_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join([c if c.isalnum() else "_" for c in title]).strip("_")
            filename = f"chart_{timestamp}_{safe_title}.html"
            filepath = os.path.join(export_dir, filename)

            # Save as HTML
            # pio.write_html accepts a figure or a dict representing a figure
            pio.write_html(data, file=filepath, auto_open=False, title=title)

            # Open in browser
            webbrowser.open(f"file://{filepath}")

            return Panel(
                Text.assemble(
                    ("✅ ", "green"),
                    ("Chart opened in browser: ", "bold"),
                    (filename, "blue underline"),
                    ("\nSaved to: ", "dim"),
                    (filepath, "dim italic"),
                ),
                border_style="green",
            )
        except Exception as e:
            return Panel(
                Text(f"❌ Error displaying chart: {e}", style="red"),
                title="Chart Error",
                border_style="red",
            )

    def _handle_notification(self, component: Any) -> RenderableType:
        content = getattr(component, "content", "")
        title = getattr(component, "title", "Notification")
        notif_type = getattr(component, "notification_type", "info")

        # Map notification types to colors and icons
        # Values are often strings like "success", "info", "warning", "error"
        type_lower = str(notif_type).lower()

        if "success" in type_lower:
            color = "green"
            icon = "✅"
        elif "error" in type_lower or "fail" in type_lower:
            color = "red"
            icon = "❌"
        elif "warning" in type_lower:
            color = "yellow"
            icon = "⚠️"
        else:
            color = "blue"
            icon = "ℹ️"

        return Panel(
            Text(f"{icon} {content}"),
            title=f"[bold]{title}[/bold]",
            border_style=color,
            expand=False,
        )
