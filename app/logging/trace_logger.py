"""JSONL structured trace logging for full research run auditability."""

from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Patterns for masking sensitive credentials in trace logs
SECRET_PATTERNS = [
    r"AIza[0-9A-Za-z\-_]{20,}",  # Google API key pattern
    r"tvly-[0-9A-Za-z\-_]{20,}",  # Tavily API key pattern
]


def sanitize_secrets(value: Any) -> Any:
    """Recursively scrub secrets from log summaries."""
    if isinstance(value, str):
        cleaned = value
        for pattern in SECRET_PATTERNS:
            cleaned = re.sub(pattern, "[MASKED_API_KEY]", cleaned)
        return cleaned
    elif isinstance(value, dict):
        return {k: sanitize_secrets(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_secrets(item) for item in value]
    return value


class TraceLogger:
    """Writes detailed, chronological execution events to JSONL files per research question."""

    def __init__(
        self,
        question_id: str,
        run_id: Optional[str] = None,
        logs_dir: Optional[str] = None,
    ) -> None:
        """Initialize TraceLogger for a specific question run."""
        self.question_id = question_id
        self.run_id = run_id or f"run_{question_id}"
        self.logs_dir = Path(logs_dir or settings.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / f"{self.question_id}.jsonl"

    def log_event(
        self,
        stage: str,
        agent: str,
        action: str,
        input_summary: Any = None,
        output_summary: Any = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an execution event as a single line of JSON."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Sanitize summaries
        clean_input = sanitize_secrets(input_summary)
        clean_output = sanitize_secrets(output_summary)
        clean_error = sanitize_secrets(error)

        event = {
            "timestamp": now,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "agent": agent,
            "stage": stage,
            "action": action,
            "input_summary": clean_input,
            "output_summary": clean_output,
            "latency_ms": round(latency_ms, 2),
            "error": clean_error,
        }

        if extra:
            event["extra"] = sanitize_secrets(extra)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as exc:
            logger.error(f"Failed to write trace log event: {exc}")

        return event

    def read_events(self) -> list[Dict[str, Any]]:
        """Read all logged events for this question."""
        if not self.log_file.exists():
            return []
        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        events.append(json.loads(line_str))
                    except Exception:
                        pass
        return events
