"""Script evaluation against configurable quality thresholds.

Uses an LLM to score each axis 0-10, computes a total, and decides pass/fail.
Works with any `LLMProvider` (MockProvider for tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ai.prompts import render
from app.ai.provider import LLMProvider
from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

SCORE_KEYS = [
    "hook", "accuracy", "clarity", "retention", "novelty",
    "pacing", "visual_potential", "naturalness", "policy_risk",
]


@dataclass
class ScriptEvaluation:
    scores: dict[str, float]
    total: float
    verdict: str
    notes: str

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_json(self) -> str:
        import json
        return json.dumps({
            "scores": self.scores,
            "total": self.total,
            "verdict": self.verdict,
            "notes": self.notes,
        })


class ScriptEvaluator:
    def __init__(self, provider: LLMProvider, min_quality: float | None = None):
        self.provider = provider
        self.settings = get_settings()
        self.min_quality = min_quality if min_quality is not None else self.settings.min_script_quality

    def evaluate(self, script: str, facts: list[str]) -> ScriptEvaluation:
        prompt = render("quality_check", script=script, facts=facts)
        result = self.provider.generate_json(prompt, temperature=0.2)
        scores = {k: float(result.get(k, 0)) for k in SCORE_KEYS}
        # policy_risk is inverted: high risk should lower the score.
        risk = scores.get("policy_risk", 0)
        total = result.get("total")
        if total is None:
            total = sum(scores.values()) / len(SCORE_KEYS)
        total = float(total) - (risk * 0.3)  # penalize policy risk
        verdict = result.get("verdict") or ("pass" if total >= self.min_quality else "fail")
        if total < self.min_quality:
            verdict = "fail"
        return ScriptEvaluation(scores=scores, total=round(total, 2),
                                verdict=verdict, notes=str(result.get("notes", "")))
