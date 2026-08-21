"""Fact verification.

Lightweight check: uses the LLM to verify each fact against its cited source.
Does not do full web search (that's a later phase option). Flags contradictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.ai.prompts import render
from app.ai.provider import LLMProvider
from app.research.researcher import ResearchResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VerifiedFact:
    fact: str
    verified: bool
    confidence: float
    notes: str


class FactVerifier:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def verify(self, research: ResearchResult,
               job_id: Optional[str] = None) -> list[VerifiedFact]:
        # Build a compact verification prompt with all facts + sources
        facts_block = "\n".join(f"- {f}" for f in research.facts)
        sources_block = "\n".join(f"- {s}" for s in research.sources)

        prompt = f"""Verify each fact against the provided sources. Return ONLY JSON object:
{{
  "verified_facts": [
    {{"fact": "...", "verified": true, "confidence": 0.9, "notes": "supported by source X"}},
    ...
  ]
}}

Facts:
{facts_block}

Sources:
{sources_block}

Be strict: if a fact is not clearly supported by any source, mark verified=false."""
        result = self.provider.generate_json(prompt, temperature=0.1)

        verified = []
        items = result.get("verified_facts", []) if isinstance(result, dict) else []
        for item in items:
            verified.append(VerifiedFact(
                fact=str(item.get("fact", "")),
                verified=bool(item.get("verified", False)),
                confidence=float(item.get("confidence", 0.0)),
                notes=str(item.get("notes", "")),
            ))

        passed = sum(1 for v in verified if v.verified)
        logger.info(f"verification: {passed}/{len(verified)} facts verified",
                    extra={"job_id": job_id, "stage": "verify", "status": "done"})
        return verified