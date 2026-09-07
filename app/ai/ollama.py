"""Ollama local LLM provider (OpenAI-compatible /v1 endpoint, zero extra deps).

Uses standard library urllib to query a locally running Ollama instance.
Provides 100% free, unlimited local inference with zero rate limits or quotas.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from app.ai.provider import LLMProvider
from app.config.settings import get_settings
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:3b"


class OllamaProvider(LLMProvider):
    """Local LLM provider querying Ollama via /v1/chat/completions."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        if not url.endswith("/v1/chat/completions"):
            if url.endswith("/v1"):
                url = f"{url}/chat/completions"
            elif url.endswith("/"):
                url = f"{url}v1/chat/completions"
            else:
                url = f"{url}/v1/chat/completions"
        self.endpoint_url = url
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", str(timeout)))

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate response from local Ollama model."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI video curator and viral editor. Return ONLY valid JSON when requested."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        def _call() -> str:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]

        try:
            return retry(_call, max_attempts=2, base_delay=3.0)
        except Exception as exc:
            logger.error(
                f"Ollama local inference failed: {exc}",
                extra={"stage": "ollama_generate", "status": "error", "model": self.model}
            )
            raise
