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
import os
import time
import urllib.request
import urllib.error
from typing import Any

from app.ai.provider import LLMProvider
from app.config.settings import get_settings
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# CI environment: use longer delays to respect 15 RPM free tier
CI_BASE_DELAY = float(os.getenv("GEMINI_CI_BASE_DELAY", "20.0"))  # seconds
CI_MAX_DELAY = float(os.getenv("GEMINI_CI_MAX_DELAY", "60.0"))
# Per-request timeout; shortened in CI (GEMINI_TIMEOUT) so a systematic outage
# fails fast instead of pushing a run past the 15-min step timeout (audit R6).
REQUEST_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "60"))


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.gemini_model
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        if not self.api_key:
            # Fail loud, never silently fall back to anything paid.
            raise RuntimeError("GEMINI_API_KEY is not set; cannot use Gemini provider")
        self._in_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"

    def _post(self, prompt: str, temperature: float, model: str | None = None) -> str:
        active_model = model or self.model
        url = f"{BASE_URL}/{active_model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            if exc.code == 429:
                logger.warning(f"Gemini {active_model} quota exceeded (429), will retry",
                             extra={"stage": "gemini", "status": "retry", "model": active_model, "error": detail})
            else:
                logger.error(f"Gemini {active_model} HTTP {exc.code}: {detail}",
                             extra={"stage": "gemini", "status": "error", "model": active_model, "error": detail})
            raise
        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {payload}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Gemini returned empty content")
        return parts[0]["text"]

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Retry with exponential backoff. In CI, use much longer base delay.
        if self._in_ci:
            return self._generate_with_ci_backoff(prompt, temperature)

        candidate_models = [self.model]
        for m in ("gemini-flash-latest", "gemini-3.1-flash-lite"):
            if m not in candidate_models:
                candidate_models.append(m)

        delays = [2.0, 5.0, 10.0]
        last_exc: Exception | None = None

        for model_idx, model_name in enumerate(candidate_models):
            for attempt, delay in enumerate(delays, 1):
                try:
                    return self._post(prompt, temperature, model=model_name)
                except urllib.error.HTTPError as exc:
                    last_exc = exc
                    if exc.code in (429, 500, 502, 503, 504):
                        logger.warning(
                            f"Gemini {model_name} HTTP {exc.code} (attempt {attempt}/{len(delays)}), retrying in {delay}s",
                            extra={"stage": "gemini", "status": "retry", "model": model_name, "code": exc.code}
                        )
                        time.sleep(delay)
                    else:
                        raise
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    logger.warning(
                        f"Gemini network error on {model_name} (attempt {attempt}/{len(delays)}), retrying in {delay}s: {exc}",
                        extra={"stage": "gemini", "status": "retry", "model": model_name}
                    )
                    time.sleep(delay)

            if model_idx < len(candidate_models) - 1:
                next_model = candidate_models[model_idx + 1]
                logger.warning(
                    f"Gemini model {model_name} unavailable after retries; falling over to {next_model}",
                    extra={"stage": "gemini", "status": "failover", "from": model_name, "to": next_model}
                )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini generate failed on all candidate models")

    def _generate_with_ci_backoff(self, prompt: str, temperature: float) -> str:
        """Generate with CI-friendly exponential backoff (longer delays)."""
        max_attempts = 2
        base_delay = CI_BASE_DELAY
        max_delay = CI_MAX_DELAY
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._post(prompt, temperature)
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(f"Gemini HTTP {exc.code}, retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})",
                                 extra={"stage": "gemini", "status": "retry", "attempt": attempt})
                    time.sleep(delay)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(f"Gemini network error: {exc}, retrying in {delay:.1f}s",
                                 extra={"stage": "gemini", "status": "retry", "attempt": attempt})
                    time.sleep(delay)
                    continue
                raise
