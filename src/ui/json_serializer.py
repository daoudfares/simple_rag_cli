"""
JSON Lines serializer for machine-to-machine output.

When ``--raw`` is active every component emitted by the agent is
serialised as a single-line JSON object written to **stdout**.
One JSON object per line (JSON Lines / NDJSON).

If ``[output].log_file`` is configured in ``secrets.toml``, every
emitted object is *also* appended to that file so it can be consumed
by external tools or test harnesses.

Every emitted line includes:
- ``timestamp`` – ISO 8601 UTC instant.
- ``session_id`` – unique identifier for the application session
  (stable while the process is running).
- ``turn_id`` – unique identifier for the current question/answer
  turn (reset on each new user question).
- ``role`` – semantic origin of the event: ``user``, ``assistant``,
  ``tool``, or ``system``.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import IO, Any

from src.config.config_loader import get_output_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session & turn tracking
# ---------------------------------------------------------------------------

_session_id: str = str(uuid.uuid4())
"""Application-wide session id, created once at import time."""

_current_turn_id: str | None = None
"""Turn id set by the caller at the beginning of each user question."""


def get_session_id() -> str:
    """Return the current session id."""
    return _session_id


def new_turn() -> str:
    """Start a new turn and return its id.

    A *turn* corresponds to one user question and all the events
    produced while answering it.  Call this once per question
    **before** emitting any events.
    """
    global _current_turn_id  # noqa: PLW0603
    _current_turn_id = str(uuid.uuid4())
    return _current_turn_id


def get_turn_id() -> str | None:
    """Return the current turn id (``None`` if no turn has started)."""
    return _current_turn_id


# ---------------------------------------------------------------------------
# Log-file handle (lazily opened once)
# ---------------------------------------------------------------------------

_log_file_handle: IO[str] | None = None
_log_file_initialised: bool = False


def _get_log_file() -> IO[str] | None:
    """Return the log-file handle, opening it on first call.

    Returns ``None`` when no ``log_file`` is configured.
    """
    global _log_file_handle, _log_file_initialised  # noqa: PLW0603
    if _log_file_initialised:
        return _log_file_handle

    _log_file_initialised = True
    output_cfg = get_output_config()
    path = output_cfg.get("log_file")
    if not path:
        return None

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _log_file_handle = open(path, "a", encoding="utf-8")  # noqa: SIM115
        logger.info("JSON log file opened: %s", path)
    except OSError as exc:
        logger.error("Failed to open log file %s: %s", path, exc)
        _log_file_handle = None

    return _log_file_handle


# ---------------------------------------------------------------------------
# Role assignment helpers
# ---------------------------------------------------------------------------

_ROLE_MAP: dict[str, str] = {
    # User-facing content
    "text": "assistant",
    "synthesis": "assistant",
    "card": "assistant",
    # Data / tool outputs
    "dataframe": "tool",
    "chart": "tool",
    "sql": "tool",
    "status_card": "tool",
    # System / orchestration
    "status_bar": "system",
    "task_tracker": "system",
    "notification": "system",
    "analysis": "system",
    "chat_input_update": "system",
    # Errors
    "error": "system",
    "sub_question_error": "system",
    # Feedback
    "feedback_ack": "system",
    # User input (set explicitly via emit_user_input)
    "user_input": "user",
}


def _role_for(event_type: str) -> str:
    """Return the semantic role for a given event type."""
    return _ROLE_MAP.get(event_type, "system")


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

    # --- chat_input_update (was falling into unknown) -----------------------
    if type_str == "chat_input_update":
        return {
            "type": "chat_input_update",
            "placeholder": getattr(component, "placeholder", ""),
            "disabled": bool(getattr(component, "disabled", False)),
            "visible": bool(getattr(component, "visible", True)),
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


def _enrich(obj: dict[str, Any]) -> dict[str, Any]:
    """Add standard envelope fields (timestamp, session, turn, role)."""
    enriched: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": _session_id,
    }
    if _current_turn_id is not None:
        enriched["turn_id"] = _current_turn_id
    enriched["role"] = obj.get("role", _role_for(obj.get("type", "")))
    enriched.update(obj)
    # Remove role duplicate if it was already in obj
    # (the one from _role_for is authoritative when not overridden)
    return enriched


def _write_json_line(obj: dict[str, Any], *, file: IO[str]) -> None:
    """Write *obj* as a single JSON line to *file*."""
    file.write(json.dumps(obj, ensure_ascii=False, default=str))
    file.write("\n")
    file.flush()


def emit(obj: dict[str, Any], *, file=None) -> None:
    """Write *obj* as a single JSON line to *file* (default: ``sys.stdout``).

    Uses ``ensure_ascii=False`` so that Unicode data is preserved,
    and ``default=str`` as a last-resort serialiser for unexpected types.

    If a log file is configured in ``[output].log_file``, the object is
    also appended there with envelope fields.
    """
    out = file or sys.stdout
    _write_json_line(obj, file=out)

    # Mirror to log file
    log_fh = _get_log_file()
    if log_fh is not None and log_fh is not out:
        _write_json_line(_enrich(obj), file=log_fh)


def emit_to_log(obj: dict[str, Any]) -> None:
    """Write *obj* **only** to the log file (not to stdout).

    This is intended for the non-raw (interactive CLI) code path:
    the human-friendly output goes to the console via Rich, while the
    structured JSON record is silently appended to the log file for
    downstream consumers.

    No-op when no log file is configured.
    """
    log_fh = _get_log_file()
    if log_fh is None:
        return
    _write_json_line(_enrich(obj), file=log_fh)


def emit_user_input(question: str, *, user_email: str = "") -> None:
    """Log the user's question as a ``user_input`` event.

    This is always written to the log file (never to stdout in raw
    mode, since stdout carries the agent's response stream).
    """
    obj: dict[str, Any] = {
        "type": "user_input",
        "role": "user",
        "content": question,
    }
    if user_email:
        obj["user_email"] = user_email
    emit_to_log(obj)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_value(v: Any) -> Any:
    """Coerce *v* to a JSON-friendly primitive."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)
