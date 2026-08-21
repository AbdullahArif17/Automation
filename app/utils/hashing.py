"""Hashing helpers for deduplication and asset integrity."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_hash(text: str) -> str:
    """Hash of whitespace/case-normalized text for similarity dedup."""
    norm = " ".join(text.lower().split())
    return sha256_text(norm)
