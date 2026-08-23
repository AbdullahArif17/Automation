"""Topic scoring and selection.

Implements the weighted scoring from the spec:
final = trend*0.25 + interest*0.20 + novelty*0.15 + shorts*0.15
      + visual*0.10 + source_quality*0.15 - duplicate_penalty

All sub-scores are 0-1. Duplicate penalty is 0-1 (deducts from final).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from app.ai.prompts import render
from app.ai.provider import LLMProvider
from app.config.settings import get_settings
from app.research.sources import TopicCandidate
from app.research.researcher import ResearchResult
from app.research.verifier import VerifiedFact
from app.utils.hashing import normalized_hash
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TopicScore:
    topic: str
    trend: float
    interest: float
    novelty: float
    visual: float
    shorts: float
    source_quality: float
    duplicate_penalty: float
    final: float
    research: Optional[ResearchResult] = None
    verified_facts: Optional[list] = None

    @property
    def passed(self) -> bool:
        return self.final >= 0.5 and self.duplicate_penalty < 0.7


class TopicSelector:
    def __init__(self, provider: LLMProvider, db=None):
        self.provider = provider
        self.settings = get_settings()
        self.weights = self.settings.topic_weights
        self.db = db

    def _compute_duplicate_penalty(self, topic: str) -> float:
        """Compare against past topics/scripts in DB. 0 = no dup, 1 = identical."""
        if not self.db:
            return 0.0
        # Simple hash-based check on normalized topic
        topic_hash = normalized_hash(topic)
        rows = self.db.fetchall(
            "SELECT topic FROM topics WHERE final_score IS NOT NULL"
        )
        if not rows:
            return 0.0
        past_hashes = [normalized_hash(r["topic"]) for r in rows]
        # Exact normalized match
        if topic_hash in past_hashes:
            return 1.0
        # Simple token overlap heuristic
        tokens = set(topic.lower().split())
        max_overlap = 0.0
        for r in rows:
            other = set(r["topic"].lower().split())
            if tokens and other:
                overlap = len(tokens & other) / len(tokens | other)
                max_overlap = max(max_overlap, overlap)
        return min(max_overlap, 0.5)  # cap at 0.5 for near-dup

    def _llm_score(self, candidate: TopicCandidate,
                   research: ResearchResult,
                   verified_facts: list) -> dict[str, float]:
        """Ask LLM to score the topic on each axis 0-1."""
        facts_str = "\n".join(f"- {vf.fact}" for vf in verified_facts)
        prompt = f"""Score this topic for a YouTube Short (AI/tech channel) on each axis 0-1.
Topic: {candidate.title}
Research summary: {research.summary}
Verified facts: {facts_str}

Return ONLY JSON:
{{
  "trend": 0.0,
  "interest": 0.0,
  "novelty": 0.0,
  "visual": 0.0,
  "shorts": 0.0,
  "source_quality": 0.0
}}"""
        result = self.provider.generate_json(prompt, temperature=0.2)
        # Clamp to 0-1
        return {k: max(0.0, min(1.0, float(result.get(k, 0.5)))) for k in
                ["trend", "interest", "novelty", "visual", "shorts", "source_quality"]}

    def score(self, candidate: TopicCandidate,
              research: ResearchResult,
              verified_facts: list,
              job_id: Optional[str] = None) -> TopicScore:
        scores = self._llm_score(candidate, research, verified_facts)
        dup_penalty = self._compute_duplicate_penalty(candidate.title)

        final = (
            scores["trend"] * self.weights.get("trend", 0.25)
            + scores["interest"] * self.weights.get("interest", 0.20)
            + scores["novelty"] * self.weights.get("novelty", 0.15)
            + scores["shorts"] * self.weights.get("shorts", 0.15)
            + scores["visual"] * self.weights.get("visual", 0.10)
            + scores["source_quality"] * self.weights.get("source_quality", 0.15)
            - dup_penalty
        )
        final = max(0.0, min(1.0, final))

        ts = TopicScore(
            topic=candidate.title,
            trend=scores["trend"],
            interest=scores["interest"],
            novelty=scores["novelty"],
            visual=scores["visual"],
            shorts=scores["shorts"],
            source_quality=scores["source_quality"],
            duplicate_penalty=dup_penalty,
            final=round(final, 3),
            research=research,
            verified_facts=verified_facts,
        )
        logger.info(f"scored '{candidate.title}': final={ts.final} (dup={dup_penalty})",
                    extra={"job_id": job_id, "stage": "score", "status": "done"})
        return ts

    def select_best(self, candidates: list[TopicCandidate],
                    job_id: Optional[str] = None,
                    max_candidates: int = 2) -> TopicScore | None:
        """Run full pipeline (research -> verify -> score) for top N candidates,
        return highest-scoring one that passes threshold."""
        from app.research.researcher import Researcher
        from app.research.verifier import FactVerifier

        researcher = Researcher(self.provider)
        verifier = FactVerifier(self.provider)

        best: TopicScore | None = None
        # Limit to max_candidates to reduce API calls (free tier: 5 RPM, 20 RPD)
        for c in candidates[:max_candidates]:
            research = researcher.research(c.title, [c], job_id=job_id)
            verified = verifier.verify(research, job_id=job_id)
            scored = self.score(c, research, verified, job_id=job_id)
            if best is None or scored.final > best.final:
                best = scored
        return best