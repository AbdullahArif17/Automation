"""Script generation pipeline.

Generates candidate scripts for a topic, evaluates each, selects the best, and
regenerates rejected ones up to `max_regeneration_attempts` (default 3).
Provider-independent: any `LLMProvider` works (MockProvider for tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.ai.evaluator import ScriptEvaluation, ScriptEvaluator
from app.ai.prompts import render
from app.ai.provider import LLMProvider
from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GeneratedScript:
    text: str
    hook: str
    duration_estimate: float
    evaluation: Optional[ScriptEvaluation] = None

    @property
    def score(self) -> float:
        return self.evaluation.total if self.evaluation else 0.0


class ScriptGenerator:
    def __init__(self, provider: LLMProvider, num_candidates: int = 1,
                 max_attempts: Optional[int] = None):
        self.provider = provider
        self.settings = get_settings()
        self.num_candidates = num_candidates
        self.max_attempts = max_attempts or self.settings.max_regeneration_attempts
        self.evaluator = ScriptEvaluator(provider)

    def _generate_one(self, topic: str, summary: str, facts: list[str]) -> GeneratedScript:
        prompt = render("script", topic=topic, summary=summary, facts=facts)
        result = self.provider.generate_json(prompt, temperature=0.8)
        return GeneratedScript(
            text=str(result.get("script", "")),
            hook=str(result.get("hook", "")),
            duration_estimate=float(result.get("duration_estimate_seconds", 30)),
        )

    def generate(self, topic: str, summary: str = "", facts: Optional[list[str]] = None,
                 job_id: Optional[str] = None) -> GeneratedScript:
        facts = facts or []
        best: Optional[GeneratedScript] = None
        attempts = 0

        while attempts < self.max_attempts:
            attempts += 1
            candidates = [self._generate_one(topic, summary, facts)
                          for _ in range(self.num_candidates)]
            for c in candidates:
                c.evaluation = self.evaluator.evaluate(c.text, facts)
                logger.info(
                    f"script score {c.score} ({c.evaluation.verdict})",
                    extra={"job_id": job_id, "stage": "script_eval", "status": c.evaluation.verdict},
                )
            candidates.sort(key=lambda c: c.score, reverse=True)
            current_best = candidates[0]
            if current_best.evaluation and current_best.evaluation.passed:
                best = current_best
                logger.info(f"accepted script at attempt {attempts} (score {best.score})",
                            extra={"job_id": job_id, "stage": "script", "status": "approved"})
                break
            best = current_best  # keep best-effort if we exhaust attempts
            logger.warning(
                f"no script passed threshold at attempt {attempts}; regenerating",
                extra={"job_id": job_id, "stage": "script", "status": "regenerate"},
            )

        return best
