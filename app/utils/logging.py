"""Structured logging.

Emits log records with timestamp, job_id, stage, status, message, and error
fields. Secrets must never be passed through these fields.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("job_id", "stage", "status", "error"):
            if hasattr(record, key):
                val = getattr(record, key)
                if val is not None:
                    payload[key] = val
        if record.exc_info and not payload.get("error"):
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "yt_shorts", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_stage(
    logger: logging.Logger,
    stage: str,
    status: str,
    message: str,
    job_id: Optional[str] = None,
    error: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    extra = {"stage": stage, "status": status}
    if job_id is not None:
        extra["job_id"] = job_id
    if error is not None:
        extra["error"] = error
    logger.log(level, message, extra=extra)
