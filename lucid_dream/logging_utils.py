"""JSONL call logging helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any


class CallLogger:
    """Thread-safe JSONL logger for all model calls."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_call(self, record: dict[str, Any]) -> None:
        """Append one record to calls.jsonl."""

        enriched = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        line = json.dumps(enriched, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
