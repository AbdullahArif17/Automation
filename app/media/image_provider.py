"""Free image provider: Unsplash Source (no key required) and Pexels (free API).

Unsplash Source: https://source.unsplash.com/ - no API key, rate-limited.
Pexels: https://www.pexels.com/api/ - free tier 200 req/hr, 20,000/month.
Both allow commercial use. Licenses tracked per asset.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass
from typing import Optional

from app.utils.logging import get_logger
from app.utils.retry import retry
from app.utils.hashing import sha256_bytes
from app.storage.files import save_bytes

logger = get_logger(__name__)


@dataclass
class ImageAsset:
    url: str
    local_path: str
    width: int
    height: int
    license: str
    source: str
    hash: str


class UnsplashSourceProvider:
    """Unsplash Source - no key, direct image URLs. Not searchable, but free."""

    BASE = "https://source.unsplash.com"

    def fetch(self, query: str, width: int = 1080, height: int = 1920,
              job_id: Optional[str] = None) -> ImageAsset:
        # Unsplash Source format: /WIDTHxHEIGHT/?QUERY
        q = urllib.parse.quote(query)
        url = f"{self.BASE}/{width}x{height}/?{q}"
        return self._download(url, "unsplash_source", job_id)

    def _download(self, url: str, source: str, job_id: Optional[str]) -> ImageAsset:
        data = retry(lambda: _fetch_bytes(url), max_attempts=2,
                     retry_on=(urllib.error.URLError, TimeoutError))
        h = sha256_bytes(data)
        # Save to output/assets/images/
        from pathlib import Path
        out_dir = Path("output/assets/images")
        out_dir.mkdir(parents=True, exist_ok=True)
        local = out_dir / f"{h[:16]}.jpg"
        save_bytes(data, local)
        logger.info(f"downloaded image {local} ({len(data)} bytes)",
                    extra={"job_id": job_id, "stage": "image", "status": "downloaded"})
        return ImageAsset(
            url=url, local_path=str(local), width=1080, height=1920,
            license="Unsplash License (free commercial)", source=source, hash=h
        )


class PexelsProvider:
    """Pexels API - free tier 200 req/hr. Requires PEXELS_API_KEY env var."""

    BASE = "https://api.pexels.com/v1"
    PER_PAGE = 10

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        if not api_key:
            logger.warning("PEXELS_API_KEY not set; Pexels provider unavailable",
                           extra={"stage": "image", "status": "no_key"})

    def search(self, query: str, job_id: Optional[str] = None) -> list[ImageAsset]:
        if not self.api_key:
            return []
        url = f"{self.BASE}/search?query={urllib.parse.quote(query)}&per_page={self.PER_PAGE}&orientation=portrait"
        # Realistic headers to avoid Cloudflare 403 (error 1010) - bare Python UA is blocked
        req = urllib.request.Request(url, headers={
            "Authorization": self.api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.error(f"Pexels HTTP {exc.code}: {exc.read()[:200]}",
                         extra={"job_id": job_id, "stage": "image", "status": "error", "error": str(exc)})
            return []

        assets = []
        for photo in data.get("photos", []):
            img_url = photo["src"]["portrait"]
            asset = self._download(img_url, "pexels", job_id)
            if asset:
                assets.append(asset)
        return assets

    def _download(self, url: str, source: str, job_id: Optional[str]) -> Optional[ImageAsset]:
        try:
            data = retry(lambda: _fetch_bytes(url), max_attempts=2,
                         retry_on=(urllib.error.URLError, TimeoutError))
        except Exception as exc:
            logger.warning(f"download failed: {exc}", extra={"job_id": job_id, "stage": "image", "status": "error"})
            return None
        h = sha256_bytes(data)
        from pathlib import Path
        out_dir = Path("output/assets/images")
        out_dir.mkdir(parents=True, exist_ok=True)
        local = out_dir / f"{h[:16]}.jpg"
        save_bytes(data, local)
        return ImageAsset(
            url=url, local_path=str(local), width=1080, height=1920,
            license="Pexels License (free commercial)", source=source, hash=h
        )


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "YTShortsBot/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()