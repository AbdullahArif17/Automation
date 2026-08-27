"""Poll sources for new videos to clip.

Supports two modes (pick one via env/secrets):
  1. S3 bucket: drop videos in an S3-compatible bucket (AWS S3, R2, Spaces, MinIO, etc.)
  2. YouTube channel: polls YOUR channel for new long-form uploads

Tracks processed files in DB (by ETag+size for S3, by YouTube video ID for channel)
to avoid re-clipping.
"""
from __future__ import annotations

import os
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.storage.database import Database
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SourceVideo:
    """A source video found in any source."""
    # Common fields
    source_type: str            # "s3" or "youtube"
    video_id: str               # S3 key or YouTube video ID
    title: str                  # For logging
    size_bytes: int
    etag: str                   # S3 ETag or YouTube video ID (used for de-dup)
    # S3-specific
    bucket: str = ""
    # YouTube-specific
    yt_video_id: str = ""
    yt_channel_id: str = ""


def _get_s3_client():
    """Lazily import boto3 and create S3 client from env."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed; add to requirements.txt")

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("CLIP_SOURCE_S3_ENDPOINT") or None,
        aws_access_key_id=os.getenv("CLIP_SOURCE_S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("CLIP_SOURCE_S3_SECRET_KEY"),
        region_name=os.getenv("CLIP_SOURCE_S3_REGION", "us-east-1"),
    )


def _get_youtube_api_key() -> Optional[str]:
    return os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _youtube_api_request(path: str, params: dict) -> dict:
    """Make a YouTube Data API v3 request."""
    key = _get_youtube_api_key()
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY not set")

    base = "https://www.googleapis.com/youtube/v3/"
    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{base}{path}?key={key}&{query}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _is_video_file(key: str) -> bool:
    return key.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"))


def _resolve_channel_id(channel_input: str) -> str:
    """Resolve @handle or UCxxxx ID to actual channel ID (UCxxxx)."""
    if channel_input.startswith("UC") and len(channel_input) == 24:
        return channel_input  # Already a channel ID
    if channel_input.startswith("@"):
        # Resolve handle via search
        data = _youtube_api_request("search", {
            "part": "snippet",
            "q": channel_input,
            "type": "channel",
            "maxResults": 1,
        })
        items = data.get("items", [])
        if items:
            return items[0]["snippet"]["channelId"]
        raise RuntimeError(f"Could not resolve channel handle: {channel_input}")
    raise RuntimeError(f"Invalid channel input: {channel_input} (must be UCxxxx or @handle)")


def _get_uploads_playlist_id(channel_id: str) -> str:
    """Get the uploads playlist ID for a channel."""
    data = _youtube_api_request("channels", {
        "part": "contentDetails",
        "id": channel_id,
    })
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"Channel not found: {channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_new_videos_s3(
    db: Database,
    bucket: str,
    prefix: str = "",
    max_files: int = 50,
) -> list[SourceVideo]:
    """List unprocessed video files in S3 bucket/prefix."""
    processed = set()
    for row in db.fetchall("SELECT source_etag, source_size FROM videos WHERE source_type='clipped' AND source_etag IS NOT NULL"):
        processed.add((row["source_etag"], row["source_size"]))

    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    new_videos: list[SourceVideo] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not _is_video_file(key):
                continue
            size = obj["Size"]
            etag = obj["ETag"].strip('"')
            if (etag, size) in processed:
                continue
            new_videos.append(SourceVideo(
                source_type="s3",
                video_id=key,
                title=Path(key).name,
                bucket=bucket,
                size_bytes=size,
                etag=etag,
                last_modified=obj["LastModified"].isoformat(),
            ))
            if len(new_videos) >= max_files:
                break
        if len(new_videos) >= max_files:
            break

    return new_videos


def list_new_videos_youtube(
    db: Database,
    channel_input: str,
    playlist_id: Optional[str] = None,
    max_videos: int = 50,
) -> list[SourceVideo]:
    """List new long-form videos from your YouTube channel."""
    # Load already-processed YouTube video IDs
    processed = set()
    for row in db.fetchall("SELECT source_etag FROM videos WHERE source_type='clipped' AND source_etag IS NOT NULL"):
        processed.add(row["source_etag"])

    channel_id = _resolve_channel_id(channel_input)

    if not playlist_id:
        playlist_id = _get_uploads_playlist_id(channel_id)

    # Fetch playlist items (newest first)
    data = _youtube_api_request("playlistItems", {
        "part": "snippet,contentDetails",
        "playlistId": playlist_id,
        "maxResults": min(max_videos * 3, 50),  # fetch extra to filter shorts
    })

    new_videos: list[SourceVideo] = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        content = item.get("contentDetails", {})
        yt_video_id = content.get("videoId", "")
        if not yt_video_id or yt_video_id in processed:
            continue

        # Get video details to check duration (filter out Shorts < 60s)
        vdata = _youtube_api_request("videos", {
            "part": "contentDetails,snippet",
            "id": yt_video_id,
        })
        vitems = vdata.get("items", [])
        if not vitems:
            continue
        v = vitems[0]
        duration_iso = v["contentDetails"].get("duration", "PT0S")
        # Parse ISO 8601 duration (e.g., PT15M33S -> 933s)
        import re
        dur_match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_iso)
        hours = int(dur_match.group(1) or 0)
        minutes = int(dur_match.group(2) or 0)
        seconds = int(dur_match.group(3) or 0)
        duration_secs = hours * 3600 + minutes * 60 + seconds

        if duration_secs < 60:  # Skip Shorts (< 60s)
            continue

        title = v["snippet"].get("title", "Untitled")
        new_videos.append(SourceVideo(
            source_type="youtube",
            video_id=yt_video_id,
            title=title,
            size_bytes=0,  # unknown until download
            etag=yt_video_id,  # use video ID as de-dup key
            yt_video_id=yt_video_id,
            yt_channel_id=channel_id,
        ))
        if len(new_videos) >= max_videos:
            break

    return new_videos


def download_video_s3(source: SourceVideo, dest_dir: Path) -> Path:
    """Download S3 video to local path."""
    s3 = _get_s3_client()
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / Path(source.video_id).name
    s3.download_file(source.bucket, source.video_id, str(local_path))
    logger.info(f"downloaded s3://{source.bucket}/{source.video_id} -> {local_path} ({source.size_bytes} bytes)")
    return local_path


def download_video_youtube(source: SourceVideo, dest_dir: Path) -> Path:
    """Download YouTube video using yt-dlp (best quality).

    Requires YT_COOKIES_PATH env var (path to cookies.txt decoded from
    YT_COOKIES_B64 repo secret by the CI workflow) to bypass bot-detection
    on CI runners. Fails with a clear error if the path is missing.
    """
    import subprocess

    cookies_path = os.getenv("YT_COOKIES_PATH")
    if not cookies_path or not Path(cookies_path).exists():
        raise RuntimeError(
            "YT_COOKIES_PATH not set or file missing. The CI workflow must decode "
            "YT_COOKIES_B64 secret to cookies.txt before running. Export cookies.txt "
            "from a logged-in YouTube session (e.g. via 'Get cookies.txt LOCALLY' "
            "browser extension), base64-encode it, store as YT_COOKIES_B64 repo secret."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / f"{source.yt_video_id}.mp4"

    cmd = [
        "yt-dlp",
        "--cookies", cookies_path,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(local_path),
        f"https://www.youtube.com/watch?v={source.yt_video_id}",
    ]

    # Retry on bot-detection (transient), capped at 2 attempts
    last_err = ""
    for attempt in range(2):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                break
            last_err = result.stderr[-1000:]
            # Bot-detection is often transient; retry once
            if "confirm you're not a bot" in last_err.lower() and attempt == 0:
                logger.warning(f"yt-dlp bot-detection hit for {source.yt_video_id}, retrying")
                continue
            raise RuntimeError(f"yt-dlp failed: {last_err}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"yt-dlp timed out downloading {source.yt_video_id}")
    else:
        raise RuntimeError(f"yt-dlp failed after retry: {last_err}")

    size = local_path.stat().st_size
    logger.info(f"downloaded youtube:{source.yt_video_id} -> {local_path} ({size} bytes)")
    return local_path


def mark_video_processed(
    db: Database,
    source: SourceVideo,
    local_path: Path,
    video_id: int,
) -> None:
    """Record that this source video has been clipped."""
    if source.source_type == "s3":
        db.execute(
            "UPDATE videos SET source_etag=?, source_size=? WHERE id=?",
            (source.etag, source.size_bytes, video_id),
        )
    else:  # youtube
        db.execute(
            "UPDATE videos SET source_etag=?, source_size=? WHERE id=?",
            (source.yt_video_id, local_path.stat().st_size, video_id),
        )
    logger.info(f"marked video {video_id} as processed: etag={source.etag}, size={local_path.stat().st_size}")


def poll_and_clip(
    db: Database,
    pipeline,
    max_clips_per_video: int = 1,
    upload: bool = True,
    max_videos_per_run: int = 2,
) -> list[tuple[SourceVideo, bool, Optional[str]]]:
    """Main entry: auto-detects source mode, polls, clips, returns results."""
    source_mode = os.getenv("CLIP_SOURCE_MODE", "").lower()

    # Auto-detect from secrets
    if not source_mode:
        if os.getenv("CLIP_SOURCE_S3_BUCKET"):
            source_mode = "s3"
        elif os.getenv("CLIP_SOURCE_YT_CHANNEL_ID"):
            source_mode = "youtube"
        else:
            raise RuntimeError("No clip source configured. Set CLIP_SOURCE_S3_BUCKET or CLIP_SOURCE_YT_CHANNEL_ID")

    if source_mode == "s3":
        bucket = os.getenv("CLIP_SOURCE_S3_BUCKET")
        prefix = os.getenv("CLIP_SOURCE_S3_PREFIX", "")
        new_videos = list_new_videos_s3(db, bucket, prefix, max_files=max_videos_per_run)
        download_fn = download_video_s3
    elif source_mode == "youtube":
        channel_id = os.getenv("CLIP_SOURCE_YT_CHANNEL_ID")
        playlist_id = os.getenv("CLIP_SOURCE_YT_PLAYLIST_ID")
        new_videos = list_new_videos_youtube(db, channel_id, playlist_id, max_videos=max_videos_per_run)
        download_fn = download_video_youtube
    else:
        raise RuntimeError(f"Unknown CLIP_SOURCE_MODE: {source_mode} (must be 's3' or 'youtube')")

    results = []
    for src in new_videos:
        try:
            local_path = download_fn(src, Path(pipeline.settings.clip_input_dir))
            pipeline.run(
                str(local_path),
                upload=upload,
                max_clips=max_clips_per_video,
                force_retranscribe=False,
            )
            video_row = db.fetchone(
                "SELECT id FROM videos WHERE topic=? ORDER BY created_at DESC LIMIT 1",
                (f"clip:{local_path.stem}",),
            )
            if video_row:
                mark_video_processed(db, src, local_path, video_row["id"])
            results.append((src, True, None))
        except Exception as exc:
            logger.error(f"clipper failed for {src.video_id}: {exc}")
            results.append((src, False, str(exc)))

    return results