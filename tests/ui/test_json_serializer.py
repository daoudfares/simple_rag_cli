"""
Tests for the JSON Lines serializer (``--raw`` mode).
"""

import io
import json
from unittest.mock import MagicMock

from src.ui.json_serializer import component_to_dict, emit

# ---------------------------------------------------------------------------
# component_to_dict — one test per component type
# ---------------------------------------------------------------------------


class TestComponentToDict:

    def test_status_bar(self):
        comp = MagicMock()
        comp.type = "status_bar_update"
        comp.message = "Loading..."
        comp.status = "running"

        result = component_to_dict(comp)
        assert result == {
            "type": "status_bar",
            "message": "Loading...",
            "status": "running",
        }

    def test_task_tracker_add(self):
        task = MagicMock()
        task.title = "Generate SQL"
        task.description = "Running query generation"

        comp = MagicMock()
        comp.type = "task_tracker_update"
        comp.operation = "add_task"
        comp.task_id = "t1"
        comp.task = task
        comp.status = None
        comp.detail = None

        result = component_to_dict(comp)
        assert result["type"] == "task_tracker"
        assert result["operation"] == "add_task"
        assert result["title"] == "Generate SQL"
        assert result["description"] == "Running query generation"

    def test_task_tracker_with_enum_operation(self):
        """operation may be an enum with a .value attribute."""
        op_enum = MagicMock()
        op_enum.value = "update_task"

        comp = MagicMock()
        comp.type = "task_tracker_update"
        comp.operation = op_enum
        comp.task_id = "t2"
        comp.task = None
        comp.status = "completed"
        comp.detail = "Done"

        result = component_to_dict(comp)
        assert result["operation"] == "update_task"
        assert result["status"] == "completed"

    def test_status_card(self):
        comp = MagicMock()
        comp.type = "status_card"
        comp.title = "Connected"
        comp.status = "success"
        comp.description = "Database online"
        comp.icon = "✅"

        result = component_to_dict(comp)
        assert result == {
            "type": "status_card",
            "title": "Connected",
            "status": "success",
            "description": "Database online",
            "icon": "✅",
        }

    def test_dataframe(self):
        comp = MagicMock()
        comp.type = "dataframe"
        comp.title = "Users"
        comp.columns = ["id", "name"]
        comp.rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        result = component_to_dict(comp)
        assert result["type"] == "dataframe"
        assert result["columns"] == ["id", "name"]
        assert len(result["rows"]) == 2
        assert result["rows"][0]["name"] == "Alice"

    def test_text(self):
        comp = MagicMock()
        comp.type = "text"
        comp.content = "# Hello"
        comp.markdown = True

        result = component_to_dict(comp)
        assert result == {"type": "text", "content": "# Hello", "markdown": True}

    def test_text_plain(self):
        comp = MagicMock()
        comp.type = "text"
        comp.content = "plain text"
        comp.markdown = False

        result = component_to_dict(comp)
        assert result["markdown"] is False

    def test_card(self):
        comp = MagicMock()
        comp.type = "card"
        comp.title = "Summary"
        comp.content = "All good"
        comp.icon = "📇"
        comp.markdown = False

        result = component_to_dict(comp)
        assert result == {
            "type": "card",
            "title": "Summary",
            "content": "All good",
            "icon": "📇",
            "markdown": False,
        }

    def test_chart(self):
        comp = MagicMock()
        comp.type = "chart"
        comp.title = "Revenue"
        comp.data = {"x": [1, 2], "y": [10, 20]}

        result = component_to_dict(comp)
        assert result["type"] == "chart"
        assert result["title"] == "Revenue"
        assert result["data"] == {"x": [1, 2], "y": [10, 20]}

    def test_chart_with_plotly_figure(self):
        """If data has .to_dict() (e.g. Plotly Figure), it should be called."""
        fig = MagicMock()
        fig.to_dict.return_value = {"data": [], "layout": {}}

        comp = MagicMock()
        comp.type = "chart"
        comp.title = "Fig"
        comp.data = fig

        result = component_to_dict(comp)
        fig.to_dict.assert_called_once()
        assert result["data"] == {"data": [], "layout": {}}

    def test_notification(self):
        comp = MagicMock()
        comp.type = "notification"
        comp.title = "Done"
        comp.content = "Query finished"
        comp.notification_type = "success"

        result = component_to_dict(comp)
        assert result == {
            "type": "notification",
            "title": "Done",
            "content": "Query finished",
            "notification_type": "success",
        }

    def test_sql_fallback(self):
        """Components without a known type but with .sql should emit type=sql."""
        comp = MagicMock(spec=[])  # no attributes by default
        comp.type = None
        comp.sql = "SELECT 1"

        result = component_to_dict(comp)
        assert result == {"type": "sql", "sql": "SELECT 1"}

    def test_string_fallback(self):
        result = component_to_dict("hello world")
        assert result == {"type": "text", "content": "hello world", "markdown": False}

    def test_unknown_fallback(self):
        result = component_to_dict(42)
        assert result["type"] == "unknown"
        assert "42" in result["repr"]

    def test_enum_type(self):
        """type may be an enum (with .value)."""
        type_enum = MagicMock()
        type_enum.value = "text"

        comp = MagicMock()
        comp.type = type_enum
        comp.content = "enum text"
        comp.markdown = False

        result = component_to_dict(comp)
        assert result["type"] == "text"


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------


class TestEmit:

    def test_emit_writes_json_line(self):
        buf = io.StringIO()
        emit({"type": "text", "content": "hi"}, file=buf)
        line = buf.getvalue()

        # Must end with newline
        assert line.endswith("\n")
        # Must be valid JSON
        parsed = json.loads(line)
        assert parsed["type"] == "text"
        assert parsed["content"] == "hi"

    def test_emit_unicode_preserved(self):
        buf = io.StringIO()
        emit({"type": "text", "content": "café ☕"}, file=buf)
        parsed = json.loads(buf.getvalue())
        assert parsed["content"] == "café ☕"

    def test_emit_non_serialisable_falls_back_to_str(self):
        """Non-serialisable values should be coerced to str via default=str."""
        buf = io.StringIO()
        emit({"type": "test", "obj": object()}, file=buf)
        parsed = json.loads(buf.getvalue())
        assert isinstance(parsed["obj"], str)
