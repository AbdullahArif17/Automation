"""Safe subprocess execution helpers.

All external command execution must go through here to avoid shell injection
and to enforce timeouts. Commands are passed as argument lists, never strings.
"""
from __future__ import annotations

import subprocess
from typing import List, Optional

from .logging import get_logger

logger = get_logger(__name__)


def run_command(
    args: List[str],
    timeout: int = 300,
    check: bool = True,
    capture: bool = True,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a command safely (no shell, list args only)."""
    logger.debug("running command", extra={"stage": "subprocess", "status": "start"})
    try:
        result = subprocess.run(
            args,
            timeout=timeout,
            check=check,
            capture_output=capture,
            text=True,
            cwd=cwd,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        logger.error(
            "command timed out",
            extra={"stage": "subprocess", "status": "timeout", "error": str(exc)},
        )
        raise
    except subprocess.CalledProcessError as exc:
        logger.error(
            "command failed",
            extra={
                "stage": "subprocess",
                "status": "error",
                "error": (exc.stderr or "")[:500],
            },
        )
        raise
