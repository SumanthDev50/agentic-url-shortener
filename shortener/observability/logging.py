"""Structured JSON logging with a request-id context variable and secret redaction.

Rules enforced here (NFR-06, section 6 of the spec):

* one JSON object per line; ``ensure_ascii`` is on so control characters can never forge
  additional log lines;
* values stored under sensitive keys (``X-API-Key``, ``Authorization``, cookies, ...) are
  replaced by ``[REDACTED]`` wherever they appear in structured fields, and header-like
  fragments inside free text (messages, tracebacks) are scrubbed as a second line of defence;
* callers never pass raw IPs, full user agents or referrer paths (nothing to redact there).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

REDACTED = "[REDACTED]"
LOGGER_NAME = "shortener"
_HANDLER_MARKER = "_shortener_json_handler"

request_id_var: ContextVar[str | None] = ContextVar("shortener_request_id", default=None)

_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "apikey",
        "cookie",
        "set-cookie",
        "password",
        "secret",
        "token",
    }
)
_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(x-api-key|authorization|proxy-authorization|cookie|set-cookie)(\s*[:=]\s*)"
    r"(?:(?:bearer|basic)\s+)?[^\s,;\"']+"
)
_RESERVED_FIELDS = frozenset({"ts", "level", "logger", "event", "request_id", "exception"})


def current_request_id() -> str | None:
    return request_id_var.get()


def is_sensitive_key(key: object) -> bool:
    return str(key).strip().lower().replace("_", "-") in _SENSITIVE_KEYS


def scrub_text(text: str) -> str:
    """Redact ``X-API-Key: value`` style fragments inside free text."""
    return _INLINE_SECRET_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


def redact(value: Any) -> Any:
    """Recursively replace values stored under sensitive keys; scrub embedded strings."""
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


class JsonFormatter(logging.Formatter):
    """Render a record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": scrub_text(record.getMessage()),
            "request_id": current_request_id(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            for key, value in redact(fields).items():
                if key not in _RESERVED_FIELDS:
                    payload[key] = value
        if record.exc_info:
            payload["exception"] = scrub_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=True, separators=(",", ":"))


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit ``event`` with structured ``fields`` (redacted by the formatter)."""
    logger.log(level, event, extra={"fields": fields})


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> logging.Handler:
    """Install (or replace) the JSON handler on the service loggers; safe to call repeatedly."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _HANDLER_MARKER, True)
    for name in (LOGGER_NAME, "uvicorn.error"):
        logger = logging.getLogger(name)
        for existing in [h for h in logger.handlers if getattr(h, _HANDLER_MARKER, False)]:
            logger.removeHandler(existing)
        logger.addHandler(handler)
        logger.setLevel(level)
    logging.getLogger("uvicorn.error").propagate = False
    return handler
