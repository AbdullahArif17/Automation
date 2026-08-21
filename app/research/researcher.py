"""Research engine: given a topic + sources, produce structured research JSON.

Uses an LLMProvider (Gemini or Mock) to extract facts, summary, confidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

from app.ai.prompts import render
from app.ai.provider import LLMProvider
from app.research.sources import TopicCandidate
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResearchResult:
    topic: str
    summary: str
    sources: list[str]
    facts: list[str]
    publication_dates: list[str]
    confidence: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class Researcher:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def research(self, topic: str, candidates: list[TopicCandidate],
                 job_id: Optional[str] = None) -> ResearchResult:
        # Build context from top candidates
        source_ctx = "\n\n".join(
            f"Source: {c.url}\nTitle: {c.title}\nSummary: {c.summary[:500]}"
            for c in candidates[:5]
        )
        prompt = render("research", topic=topic, sources=source_ctx)
        result = self.provider.generate_json(prompt, temperature=0.3)

        # Validate and coerce
        summary = str(result.get("summary", ""))
        sources = [str(s) for s in result.get("sources", [])]
        facts = [str(f) for f in result.get("facts", [])]
        pub_dates = [str(d) for d in result.get("publication_dates", [])]
        confidence = float(result.get("confidence", 0.5))

        if not facts:
            logger.warning("research returned no facts; confidence lowered",
                           extra={"job_id": job_id, "stage": "research", "status": "low_facts"})
            confidence = min(confidence, 0.4)

        res = ResearchResult(
            topic=topic, summary=summary, sources=sources,
            facts=facts, publication_dates=pub_dates, confidence=confidence
        )
        logger.info(f"research done: {len(facts)} facts, conf={confidence:.2f}",
                    extra={"job_id": job_id, "stage": "research", "status": "done"})
        return res