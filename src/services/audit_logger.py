"""
Audit logging utilities.

Writes one JSON object per line for each user interaction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config.paths import AUDIT_LOG_FILE, ensure_runtime_dirs


def log_interaction(
    *,
    user_email: str,
    question: str,
    sql: str | None,
    response: str | None,
) -> None:
    """Append an interaction entry to the local JSONL audit log."""
    ensure_runtime_dirs()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_email": user_email,
        "question": question,
        "sql": sql,
        "response": response,
    }
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")
