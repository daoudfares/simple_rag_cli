from unittest.mock import MagicMock

from rich.panel import Panel

from src.ui.formatter import CLIFormatter


def test_format_notification():
    formatter = CLIFormatter()
    component = MagicMock()
    component.type = "notification"
    component.title = "Success Notification"
    component.content = "Operation completed successfully"
    component.notification_type = "success"
    component.rich_component = None
    component.simple_component = None

    result = formatter.format_component(component)
    assert isinstance(result, Panel)
    assert "Success Notification" in result.title
    assert "✅" in result.renderable.plain
    assert "Operation completed successfully" in result.renderable.plain
    assert result.border_style == "green"

def test_format_notification_error():
    formatter = CLIFormatter()
    component = MagicMock()
    component.type = "notification"
    component.title = "Error Notification"
    component.content = "Something went wrong"
    component.notification_type = "error"
    component.rich_component = None
    component.simple_component = None

    result = formatter.format_component(component)
    assert isinstance(result, Panel)
    assert "❌" in result.renderable.plain
    assert result.border_style == "red"
