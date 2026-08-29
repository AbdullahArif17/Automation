"""Metadata generation: titles, descriptions, hashtags.

Uses LLM to generate 3 title candidates, description, and relevant hashtags.
All output is validated for length and compliance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from app.ai.prompts import render
from app.ai.provider import LLMProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VideoMetadata:
    titles: list[str]       # 3 candidates, max 60 chars each
    description: str        # 1-2 sentences + hashtags
    hashtags: list[str]     # 3-8 tags, lowercase, no spaces

    def primary_title(self) -> str:
        return self.titles[0] if self.titles else "Untitled Short"

    def to_dict(self) -> dict:
        return {
            "titles": self.titles,
            "description": self.description,
            "hashtags": self.hashtags,
        }


class MetadataGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, topic: str, script: str, duration: float,
                 job_id: Optional[str] = None) -> VideoMetadata:
        prompt = render("metadata", topic=topic, script=script)
        result = self.provider.generate_json(prompt, temperature=0.7)

        titles = [str(t)[:60] for t in result.get("titles", [])[:3]]
        # Ensure exactly 3 titles
        while len(titles) < 3:
            titles.append(f"{topic} - Short")

        description = str(result.get("description", ""))[:5000]
        hashtags = [str(h).lower().replace(" ", "") for h in result.get("hashtags", [])[:8]]
        # Ensure # prefix
        hashtags = [h if h.startswith("#") else f"#{h}" for h in hashtags]
        # Ensure #shorts is always present (required for Shorts shelf)
        if "#shorts" not in hashtags:
            hashtags.insert(0, "#shorts")
        # Default tags only if LLM returned none at all
        if not hashtags or hashtags == ["#shorts"]:
            hashtags = ["#shorts", "#ai", "#tech", "#programming"]

        metadata = VideoMetadata(titles=titles, description=description, hashtags=hashtags)
        logger.info(f"generated metadata: {metadata.primary_title()}",
                    extra={"job_id": job_id, "stage": "metadata", "status": "done"})
        return metadata