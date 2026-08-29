"""Free stock video provider: Pexels Videos and Pixabay (free tiers).

Pexels Videos: free tier 200 req/hr (same API key as images).
Pixabay: free tier, requires API key. Both allow commercial use.
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
class VideoAsset:
    url: str
    local_path: str
    width: int
    height: int
    duration: float
    license: str
    source: str
    hash: str


class PexelsVideoProvider:
    """Pexels Videos API - same key as images, free tier."""

    BASE = "https://api.pexels.com/videos"
    PER_PAGE = 10

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        if not api_key:
            logger.warning("PEXELS_API_KEY not set; Pexels video provider unavailable",
                           extra={"stage": "stock_video", "status": "no_key"})

    def search(self, query: str, job_id: Optional[str] = None) -> list[VideoAsset]:
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
            logger.error(f"Pexels video HTTP {exc.code}: {exc.read()[:200]}",
                         extra={"job_id": job_id, "stage": "stock_video", "status": "error", "error": str(exc)})
            return []

        assets = []
        for vid in data.get("videos", []):
            # Pick the best portrait file
            files = vid.get("video_files", [])
            portrait = [f for f in files if f.get("width", 0) >= 1080 and f.get("height", 0) >= 1920]
            if not portrait:
                portrait = sorted(files, key=lambda f: f.get("width", 0), reverse=True)[:1]
            for f in portrait:
                asset = self._download(f["link"], vid.get("duration", 10), "pexels", job_id)
                if asset:
                    assets.append(asset)
                break  # one per video
        return assets

    def _download(self, url: str, duration: float, source: str,
                  job_id: Optional[str]) -> Optional[VideoAsset]:
        try:
            data = retry(lambda: _fetch_bytes(url), max_attempts=2,
                         retry_on=(urllib.error.URLError, TimeoutError))
        except Exception as exc:
            logger.warning(f"video download failed: {exc}",
                           extra={"job_id": job_id, "stage": "stock_video", "status": "error"})
            return None
        h = sha256_bytes(data)
        from pathlib import Path
        out_dir = Path("output/assets/videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        local = out_dir / f"{h[:16]}.mp4"
        save_bytes(data, local)
        return VideoAsset(
            url=url, local_path=str(local), width=1080, height=1920,
            duration=duration, license="Pexels License (free commercial)",
            source=source, hash=h
        )


class PixabayVideoProvider:
    """Pixabay Videos - free tier, requires PIXABAY_API_KEY."""

    BASE = "https://pixabay.com/api/videos"
    PER_PAGE = 10

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        if not api_key:
            logger.warning("PIXABAY_API_KEY not set; Pixabay video provider unavailable",
                           extra={"stage": "stock_video", "status": "no_key"})

    def search(self, query: str, job_id: Optional[str] = None) -> list[VideoAsset]:
        if not self.api_key:
            return []
        url = (f"{self.BASE}/?key={self.api_key}"
               f"&q={urllib.parse.quote(query)}&per_page={self.PER_PAGE}"
               f"&min_width=1080&min_height=1920")
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.error(f"Pixabay HTTP {exc.code}: {exc.read()[:200]}",
                         extra={"job_id": job_id, "stage": "stock_video", "status": "error", "error": str(exc)})
            return []

        assets = []
        for hit in data.get("hits", []):
            files = hit.get("videos", {})
            # Prefer largest
            for quality in ["large", "medium", "small", "tiny"]:
                if quality in files:
                    f = files[quality]
                    asset = self._download(f["url"], hit.get("duration", 10), "pixabay", job_id)
                    if asset:
                        assets.append(asset)
                    break
        return assets

    def _download(self, url: str, duration: float, source: str,
                  job_id: Optional[str]) -> Optional[VideoAsset]:
        try:
            data = retry(lambda: _fetch_bytes(url), max_attempts=2,
                         retry_on=(urllib.error.URLError, TimeoutError))
        except Exception as exc:
            logger.warning(f"video download failed: {exc}",
                           extra={"job_id": job_id, "stage": "stock_video", "status": "error"})
            return None
        h = sha256_bytes(data)
        from pathlib import Path
        out_dir = Path("output/assets/videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        local = out_dir / f"{h[:16]}.mp4"
        save_bytes(data, local)
        return VideoAsset(
            url=url, local_path=str(local), width=1080, height=1920,
            duration=duration, license="Pixabay License (free commercial)",
            source=source, hash=h
        )


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "YTShortsBot/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()