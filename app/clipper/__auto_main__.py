"""Automated clipper entry point: polls source (S3 or YouTube channel),
clips new videos, uploads.

Run via: python -m app.clipper.__auto_main__

Environment variables:
  COMMON:
    YOUTUBE_API_KEY              -- YouTube Data API v3 key (required for channel mode)

  MODE 1 - S3 BUCKET:
    CLIP_SOURCE_S3_BUCKET        -- bucket name
    CLIP_SOURCE_S3_PREFIX        -- optional prefix/folder (default: "")
    CLIP_SOURCE_S3_ENDPOINT      -- optional (for R2/Spaces/MinIO)
    CLIP_SOURCE_S3_ACCESS_KEY    -- access key
    CLIP_SOURCE_S3_SECRET_KEY    -- secret key
    CLIP_SOURCE_S3_REGION        -- region (default: us-east-1)

  MODE 2 - YOUR YOUTUBE CHANNEL:
    CLIP_SOURCE_YT_CHANNEL_ID    -- channel ID (UCxxxx) or handle (@name)
    CLIP_SOURCE_YT_PLAYLIST_ID   -- optional: specific playlist ID

  AUTO-DETECT:
    CLIP_SOURCE_MODE             -- "s3" or "youtube" (leave empty to auto-detect)

  LIMITS:
    CLIP_MAX_VIDEOS_PER_RUN      -- max videos per run (default: 2)
    CLIP_MAX_CLIPS_PER_VIDEO     -- max Shorts per source (default: 1)
    CLIP_UPLOAD                  -- "true"/"false" (default: "true")
    WHISPER_MODEL_SIZE           -- passed through
    CLIP_CROP_MODE               -- passed through
"""
from __future__ import annotations

import os
import sys

from app.clipper.pipeline import ClipperPipeline
from app.clipper.storage_poller import poll_and_clip
from app.config.settings import get_settings
from app.storage.database import Database
from app.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    settings = get_settings()

    # Limits (work for both modes)
    max_videos = int(os.getenv("CLIP_MAX_VIDEOS_PER_RUN", "2"))
    max_clips = int(os.getenv("CLIP_MAX_CLIPS_PER_VIDEO", "1"))
    upload = os.getenv("CLIP_UPLOAD", "true").lower() == "true"

    logger.info(f"auto-clipper start: max_videos={max_videos}, max_clips={max_clips}, upload={upload}")

    db = Database(settings.db_path)
    try:
        pipeline = ClipperPipeline(db=db, settings=settings)
        # poll_and_clip auto-detects mode from CLIP_SOURCE_MODE or secrets
        results = poll_and_clip(
            db=db,
            pipeline=pipeline,
            max_clips_per_video=max_clips,
            upload=upload,
            max_videos_per_run=max_videos,
        )

        ok = sum(1 for _, success, _ in results if success)
        fail = len(results) - ok
        logger.info(f"auto-clipper done: processed={len(results)}, ok={ok}, failed={fail}")

        for src, success, err in results:
            status = "OK" if success else f"FAILED: {err}"
            print(f"  {src.video_id}: {status}")

        return 0 if fail == 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())