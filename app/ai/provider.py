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
        # Detect prompt type and return appropriate mock response
        prompt_lower = prompt.lower()
        if "quality_check" in prompt_lower or "score" in prompt_lower:
            # Evaluation prompt - return numeric scores
            return json.dumps({
                "hook": 8.5, "accuracy": 9.0, "clarity": 8.0,
                "retention": 8.5, "novelty": 7.5, "pacing": 8.0,
                "visual_potential": 8.0, "naturalness": 8.5,
                "policy_risk": 1.0, "total": 8.5, "verdict": "pass",
                "notes": "mock evaluation"
            })
        elif "visual_plan" in prompt_lower:
            # Visual plan prompt
            return json.dumps({
                "duration": 30,
                "scenes": [
                    {"start": 0, "end": 10, "visual_query": "AI coding",
                     "visual_type": "image", "motion": "zoom_in"},
                    {"start": 10, "end": 20, "visual_query": "code on screen",
                     "visual_type": "video", "motion": "pan"},
                    {"start": 20, "end": 30, "visual_query": "results",
                     "visual_type": "image", "motion": "static"},
                ]
            })
        elif "metadata" in prompt_lower:
            # Metadata prompt
            return json.dumps({
                "titles": ["AI Coding Tools", "Best AI for Developers", "Code with AI"],
                "description": "Discover the best AI coding tools. #ai #coding",
                "hashtags": ["#ai", "#coding", "#programming", "#shorts"]
            })
        elif "topic" in prompt_lower and "score" in prompt_lower:
            # Topic scoring prompt
            return json.dumps({
                "trend": 0.7, "interest": 0.8, "novelty": 0.6,
                "visual": 0.7, "shorts": 0.8, "source_quality": 0.7
            })
        elif "research" in prompt_lower:
            # Research prompt
            return json.dumps({
                "summary": "Mock research summary about the topic.",
                "facts": ["Fact 1 about AI", "Fact 2 about coding"],
                "sources": ["source1.com", "source2.com"],
                "confidence": 0.8
            })
        elif "verify" in prompt_lower or "fact" in prompt_lower:
            # Verification prompt
            return json.dumps({
                "verified_facts": [
                    {"fact": "Fact 1 about AI", "verified": True, "confidence": 0.9},
                    {"fact": "Fact 2 about coding", "verified": True, "confidence": 0.8}
                ]
            })
        # Default: script generation
        return json.dumps({
            "script": "Mock script about the topic. It reads naturally and stays accurate to the facts. Local, free tooling makes this possible.",
            "hook": "Hook",
            "duration_estimate_seconds": 30
        })
