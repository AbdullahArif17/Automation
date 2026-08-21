"""File management: safe creation, size limits, validation."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB safety cap


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bytes(data: bytes, dest: Path, max_bytes: int = MAX_FILE_BYTES) -> Path:
    if len(data) > max_bytes:
        raise ValueError(f"file too large: {len(data)} > {max_bytes}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    logger.debug(f"saved {len(data)} bytes to {dest}")
    return dest


def copy_file(src: Path, dest: Path, max_bytes: int = MAX_FILE_BYTES) -> Path:
    size = src.stat().st_size
    if size > max_bytes:
        raise ValueError(f"source too large: {size} > {max_bytes}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def file_size_ok(path: Path, max_bytes: int = MAX_FILE_BYTES) -> bool:
    try:
        return path.stat().st_size <= max_bytes
    except OSError:
        return False


def safe_remove(path: Path, missing_ok: bool = True) -> None:
    try:
        path.unlink(missing_ok=missing_ok)
    except OSError as exc:
        logger.warning(f"could not remove {path}: {exc}")
