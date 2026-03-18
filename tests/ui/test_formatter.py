from unittest.mock import MagicMock

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.ui.formatter import CLIFormatter


def test_formatter_initialization():
    formatter = CLIFormatter()
    assert formatter.console is not None
    assert "dataframe" in formatter.handlers


def test_format_unknown_component():
    formatter = CLIFormatter()
    component = MagicMock()
    component.type = "unknown_type"
    component.rich_component = None
    component.simple_component = None

    result = formatter.format_component(component)
    assert isinstance(result, Text)
    assert "Unknown component type" in result.plain


def test_format_dataframe():
    formatter = CLIFormatter()
    component = MagicMock()
    component.type = "dataframe"
    component.rich_component = None
    component.simple_component = None
    component.rows = [{"col1": "val1", "col2": "val2"}]
    component.columns = ["col1", "col2"]
    component.title = "Test Table"

    result = formatter.format_component(component)
    assert isinstance(result, Table)
    assert result.title == "Test Table"


def test_format_text_markdown():
    formatter = CLIFormatter()
    component = MagicMock()
    component.type = "text"
    component.rich_component = None
    component.simple_component = None
    component.content = "# Hello World"
    component.markdown = True

    result = formatter.format_component(component)
    assert isinstance(result, Panel)


def test_format_wrapped_component():
    formatter = CLIFormatter()

    # Mock for the inner component
    inner = MagicMock()
    inner.type = "text"
    inner.content = "Wrapped content"
    inner.markdown = False
    inner.rich_component = None
    inner.simple_component = None

    # Mock for the wrapper
    wrapper = MagicMock()
    # To simulate 'not hasattr(wrapper, "type")', we can't easily use MagicMock
    # because it creates attributes on the fly.
    # Instead, we can set it to None or use a side effect.
    # Our code checks: getattr(component, "type", None) is None
    wrapper.type = None
    wrapper.rich_component = inner
    wrapper.simple_component = None

    result = formatter.format_component(wrapper)
    assert isinstance(result, Panel)
    # The panel should contain the inner content (this depends on Rich internal rendering,
    # but the fact it returned a Panel is a good sign it unwrapped correctly)

def test_format_chart_disabled():
    formatter = CLIFormatter(show_charts=False)
    component = MagicMock()
    component.type = "chart"
    component.title = "Test Chart"
    component.data = {"plot": "data"}
    component.rich_component = None
    component.simple_component = None

    result = formatter.format_component(component)
    assert isinstance(result, Panel)
    assert "Chart generated: Test Chart" in result.renderable.plain
    assert "Type 'charts on' to enable" in result.renderable.plain


def test_format_chart_enabled(monkeypatch, tmp_path):
    # Mock pio.write_html and browser open
    mock_write_html = MagicMock()
    mock_open = MagicMock()

    monkeypatch.setattr("src.ui.formatter.CHART_EXPORT_DIR", tmp_path)
    monkeypatch.setattr("plotly.io.write_html", mock_write_html)
    monkeypatch.setattr("webbrowser.open", mock_open)

    formatter = CLIFormatter(show_charts=True)
    component = MagicMock()
    component.type = "chart"
    component.title = "Test Chart"
    component.data = {"plot": "data"}
    component.rich_component = None
    component.simple_component = None

    result = formatter.format_component(component)
    assert isinstance(result, Panel)
    assert "Chart opened in browser" in result.renderable.plain
    mock_write_html.assert_called_once()
    mock_open.assert_called_once()
