"""LLM provider abstraction.

Every LLM integration implements `LLMProvider`. This keeps the system
provider-independent so paid providers can be added later without rewrites.
A `MockProvider` is included so downstream phases are testable without API keys
or quota.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerant of code fences/leading text."""
    if text is None:
        raise ValueError("empty response")
    cleaned = text.strip()
    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to first balanced brace block.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("could not parse JSON from LLM response")


class LLMProvider:
    """Interface every LLM backend must implement."""

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, temperature: float = 0.7) -> Any:
        return extract_json(self.generate(prompt, temperature=temperature))


class MockProvider(LLMProvider):
    """Deterministic provider for tests and dry runs (no network, no cost)."""

    def __init__(self, responses: Optional[list[str]] = None):
        self.responses = list(responses or [])
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        # Default deterministic passthrough response.
        return json.dumps({"script": "Mock script about the topic.", "hook": "Hook",
                           "duration_estimate_seconds": 30})
