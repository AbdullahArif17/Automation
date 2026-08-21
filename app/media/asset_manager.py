"""Asset manager: stores and retrieves media assets with license metadata.

Uses the `assets` table from Phase 2. Caches downloads to avoid re-fetching.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.storage.database import Database
from app.media.image_provider import ImageAsset, UnsplashSourceProvider, PexelsProvider
from app.media.stock_provider import VideoAsset, PexelsVideoProvider, PixabayVideoProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AssetRecord:
    id: int
    source: str
    source_url: str
    license: str
    local_path: str
    type: str  # "image" | "video"
    duration: float
    width: int
    height: int
    hash: str


class AssetManager:
    def __init__(self, db: Database,
                 unsplash: UnsplashSourceProvider | None = None,
                 pexels_img: PexelsProvider | None = None,
                 pexels_vid: PexelsVideoProvider | None = None,
                 pixabay_vid: PixabayVideoProvider | None = None):
        self.db = db
        self.unsplash = unsplash or UnsplashSourceProvider()
        self.pexels_img = pexels_img
        self.pexels_vid = pexels_vid
        self.pixabay_vid = pixabay_vid

    def get_or_fetch_image(self, query: str, job_id: Optional[str] = None) -> Optional[AssetRecord]:
        # Check cache by hash of query (approximate dedup)
        from app.utils.hashing import sha256_text
        q_hash = sha256_text(query)[:16]
        row = self.db.fetchone(
            "SELECT * FROM assets WHERE type='image' AND hash LIKE ?", (f"{q_hash}%",))
        if row:
            logger.info(f"cache hit for image query: {query}",
                        extra={"job_id": job_id, "stage": "asset", "status": "cache_hit"})
            return self._row_to_record(row)

        # Try providers in order
        asset: Optional[ImageAsset] = None
        for provider_name, provider in [
            ("unsplash", self.unsplash),
            ("pexels", self.pexels_img),
        ]:
            if provider is None:
                continue
            try:
                if provider_name == "unsplash":
                    asset = provider.fetch(query, job_id=job_id)
                else:
                    assets = provider.search(query, job_id=job_id)
                    if assets:
                        asset = assets[0]
                if asset:
                    logger.info(f"fetched image from {provider_name}: {asset.local_path}",
                                extra={"job_id": job_id, "stage": "asset", "status": "fetched"})
                    break
            except Exception as exc:
                logger.warning(f"{provider_name} failed: {exc}",
                               extra={"job_id": job_id, "stage": "asset", "status": "error"})

        if not asset:
            return None

        return self._store_asset(asset, job_id)

    def get_or_fetch_video(self, query: str, job_id: Optional[str] = None) -> Optional[AssetRecord]:
        from app.utils.hashing import sha256_text
        q_hash = sha256_text(query)[:16]
        row = self.db.fetchone(
            "SELECT * FROM assets WHERE type='video' AND hash LIKE ?", (f"{q_hash}%",))
        if row:
            logger.info(f"cache hit for video query: {query}",
                        extra={"job_id": job_id, "stage": "asset", "status": "cache_hit"})
            return self._row_to_record(row)

        asset: Optional[VideoAsset] = None
        for provider_name, provider in [
            ("pexels_video", self.pexels_vid),
            ("pixabay_video", self.pixabay_vid),
        ]:
            if provider is None:
                continue
            try:
                assets = provider.search(query, job_id=job_id)
                if assets:
                    asset = assets[0]
                    logger.info(f"fetched video from {provider_name}: {asset.local_path}",
                                extra={"job_id": job_id, "stage": "asset", "status": "fetched"})
                    break
            except Exception as exc:
                logger.warning(f"{provider_name} failed: {exc}",
                               extra={"job_id": job_id, "stage": "asset", "status": "error"})

        if not asset:
            return None

        return self._store_asset(asset, job_id)

    def _store_asset(self, asset, job_id: Optional[str]) -> AssetRecord:
        cur = self.db.execute(
            "INSERT INTO assets (source, source_url, license, local_path, type, duration, width, height, hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (asset.source, asset.url, asset.license, asset.local_path, asset.__class__.__name__.replace("Asset", "").lower(),
             getattr(asset, "duration", 0), asset.width, asset.height, asset.hash),
        )
        asset_id = cur.lastrowid
        logger.info(f"stored asset #{asset_id} ({asset.local_path})",
                    extra={"job_id": job_id, "stage": "asset", "status": "stored"})
        return AssetRecord(
            id=asset_id, source=asset.source, source_url=asset.url,
            license=asset.license, local_path=asset.local_path,
            type=asset.__class__.__name__.replace("Asset", "").lower(),
            duration=getattr(asset, "duration", 0),
            width=asset.width, height=asset.height, hash=asset.hash
        )

    @staticmethod
    def _row_to_record(row) -> AssetRecord:
        return AssetRecord(
            id=row["id"], source=row["source"], source_url=row["source_url"],
            license=row["license"], local_path=row["local_path"], type=row["type"],
            duration=row["duration"], width=row["width"], height=row["height"], hash=row["hash"]
        )