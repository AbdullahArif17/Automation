"""Prompt template loader.

Loads prompt files from the `prompts/` directory and formats them with
variables. Keeps prompts out of code so they can be edited without redeploying.

Uses a safe regex substitution so literal JSON braces `{...}` inside templates
are left untouched — only `{word}` placeholders are replaced.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_CACHE: dict[str, str] = {}
_TOKEN = re.compile(r"\{(\w+)\}")


def load(name: str) -> str:
    if name not in _CACHE:
        path = PROMPTS_DIR / f"{name}.txt"
        _CACHE[name] = path.read_text(encoding="utf-8")
    return _CACHE[name]


def render(name: str, **kwargs: Any) -> str:
    tmpl = load(name)

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in kwargs:
            return m.group(0)  # leave unknown tokens untouched
        val = kwargs[key]
        if isinstance(val, list):
            return ", ".join(str(x) for x in val)
        return str(val)

    return _TOKEN.sub(_sub, tmpl)
