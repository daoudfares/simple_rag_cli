"""
JSON Lines serializer for machine-to-machine output.

When ``--raw`` is active every component emitted by the agent is
serialised as a single-line JSON object written to **stdout**.
One JSON object per line (JSON Lines / NDJSON).
"""

import json
import sys
from typing import Any


def component_to_dict(component: Any) -> dict[str, Any]:
    """Convert a Vanna component into a JSON-serialisable dict.

    The returned dict always contains a ``"type"`` key.  All values are
    primitives, lists, or dicts — never Rich renderables or custom objects.
    """
    comp_type = getattr(component, "type", None)

    # Resolve enum → string
    type_str: str = ""
    if comp_type is not None:
        type_str = str(comp_type.value) if hasattr(comp_type, "value") else str(comp_type)

    # --- per-type extractors ------------------------------------------------

    if type_str == "status_bar_update":
        return {
            "type": "status_bar",
            "message": getattr(component, "message", ""),
            "status": getattr(component, "status", ""),
        }

    if type_str == "task_tracker_update":
        operation = getattr(component, "operation", "")
        op_str = operation.value if hasattr(operation, "value") else str(operation)
        task = getattr(component, "task", None)
        return {
            "type": "task_tracker",
            "operation": op_str,
            "task_id": getattr(component, "task_id", None),
            "title": getattr(task, "title", None) if task else None,
            "description": getattr(task, "description", None) if task else None,
            "status": getattr(component, "status", None),
            "detail": getattr(component, "detail", None),
        }

    if type_str == "status_card":
        return {
            "type": "status_card",
            "title": getattr(component, "title", ""),
            "status": getattr(component, "status", ""),
            "description": getattr(component, "description", ""),
            "icon": getattr(component, "icon", ""),
        }

    if type_str == "dataframe":
        rows = getattr(component, "rows", [])
        columns = getattr(component, "columns", [])
        # Ensure rows are plain dicts (not custom objects)
        safe_rows = []
        for row in rows:
            if isinstance(row, dict):
                safe_rows.append({k: _safe_value(v) for k, v in row.items()})
            else:
                safe_rows.append(str(row))
        return {
            "type": "dataframe",
            "title": getattr(component, "title", ""),
            "columns": list(columns),
            "rows": safe_rows,
        }

    if type_str == "text":
        return {
            "type": "text",
            "content": getattr(component, "content", ""),
            "markdown": bool(getattr(component, "markdown", False)),
        }

    if type_str == "card":
        return {
            "type": "card",
            "title": getattr(component, "title", ""),
            "content": getattr(component, "content", ""),
            "icon": getattr(component, "icon", ""),
            "markdown": bool(getattr(component, "markdown", False)),
        }

    if type_str == "chart":
        data = getattr(component, "data", {})
        # Plotly figures may be dicts or Figure objects – try to_dict()
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        return {
            "type": "chart",
            "title": getattr(component, "title", ""),
            "data": data,
        }

    if type_str == "notification":
        notif_type = getattr(component, "notification_type", "info")
        return {
            "type": "notification",
            "title": getattr(component, "title", ""),
            "content": getattr(component, "content", ""),
            "notification_type": str(notif_type),
        }

    # --- SQL helper (some components carry an `sql` attribute) ---------------
    sql = getattr(component, "sql", None)
    if sql:
        return {"type": "sql", "sql": sql}

    # --- Fallback -----------------------------------------------------------
    if isinstance(component, str):
        return {"type": "text", "content": component, "markdown": False}

    return {
        "type": "unknown",
        "repr": str(component),
    }


def emit(obj: dict[str, Any], *, file=None) -> None:
    """Write *obj* as a single JSON line to *file* (default: ``sys.stdout``).

    Uses ``ensure_ascii=False`` so that Unicode data is preserved,
    and ``default=str`` as a last-resort serialiser for unexpected types.
    """
    out = file or sys.stdout
    out.write(json.dumps(obj, ensure_ascii=False, default=str))
    out.write("\n")
    out.flush()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_value(v: Any) -> Any:
    """Coerce *v* to a JSON-friendly primitive."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)
