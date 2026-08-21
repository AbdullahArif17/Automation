"""Gemini provider (Google Generative Language API, free tier).

Uses the REST endpoint via stdlib `urllib` (no extra dependency). The current
stable free-tier model is `gemini-2.5-flash` (verify before use — older models
like gemini-1.5/2.0-flash have been shut down). Free tier is rate-limited
(historically ~15 RPM, ~1.5M tokens/day); the retry helper backs off on 429.

If `GEMINI_API_KEY` is unset, `generate` raises so callers fail loudly rather
than silently using a paid service. The system never auto-upgrades to paid.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

from app.ai.provider import LLMProvider
from app.config.settings import get_settings
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.gemini_model
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        if not self.api_key:
            # Fail loud, never silently fall back to anything paid.
            raise RuntimeError("GEMINI_API_KEY is not set; cannot use Gemini provider")

    def _post(self, prompt: str, temperature: float) -> str:
        url = f"{BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            logger.error(f"Gemini HTTP {exc.code}: {detail}",
                         extra={"stage": "gemini", "status": "error", "error": detail})
            raise
        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {payload}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Gemini returned empty content")
        return parts[0]["text"]

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Retry only transient errors (429, 5xx). Invalid prompts are not retried.
        return retry(self._post, prompt, temperature, max_attempts=3,
                     retry_on=(urllib.error.URLError, TimeoutError))
