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
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.storage.database import Database
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Common yt-dlp arguments for YouTube extraction:
# - --js-runtimes deno: enables JS challenge solving (required for bot-detection)
# - --extractor-args youtube:player_client=ios;formats=missing_pot:
#   merged into single flag using yt-dlp's semicolon separator for
#   multiple extractor args. player_client tries less-restricted clients
#   (ios avoids heavy PO Token enforcement); formats=missing_pot
#   fallback to token-free formats (360p) if PO Token errors persist.
# - --remote-components ejs:github: fetches EJS challenge-solver from GitHub at runtime
#   if not bundled locally (needed when yt-dlp installed via pip, not standalone binary).
# Note: Chrome UA removed — tv/android_vr clients use different UA strings; yt-dlp handles
# client-appropriate UA automatically when player_client is specified.
# Note: Using android and web clients because android_vr and tv often reject cookies.
def _has_ipv6() -> bool:
    """Check if this machine has an active IPv6 route to the internet."""
    import socket
    if os.getenv("FORCE_IPV6", "").lower() in ("1", "true", "yes"):
        return True
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        sock.connect(("2001:4860:4860::8888", 53))
        sock.close()
        return True
    except Exception:
        return False


def _get_yt_dlp_common_args() -> list[str]:
    """Return common yt-dlp arguments, adding --force-ipv6 only if IPv6 is routable."""
    args = [
        "--js-runtimes", "deno",
        "--remote-components", "ejs:github",
    ]
    if _has_ipv6():
        args.append("--force-ipv6")
    return args


def _subprocess_env() -> dict[str, str]:
    """Return env dict with ~/.deno/bin and python directory injected into PATH so yt-dlp and Deno are always found."""
    import sys
    env = os.environ.copy()
    paths = []
    deno_bin = os.path.join(os.path.expanduser("~"), ".deno", "bin")
    if os.path.isdir(deno_bin):
        paths.append(deno_bin)
    py_dir = os.path.dirname(sys.executable)
    if py_dir and os.path.isdir(py_dir):
        paths.append(py_dir)
    current_path = env.get("PATH", "")
    sep = ";" if sys.platform == "win32" else ":"
    for p in paths:
        if p not in current_path.split(sep):
            current_path = f"{p}{sep}{current_path}" if current_path else p
    env["PATH"] = current_path
    return env


def validate_cookies_file(cookies_path: Path, test_url: str = "https://www.youtube.com/feed/subscriptions") -> tuple[int, bool]:
    """
    Validate a Netscape-format cookies.txt file.

    Checks:
    1. Line endings normalized to LF
    2. Every non-comment, non-blank line has exactly 7 tab-separated fields
    3. Cookies actually authenticate with yt-dlp against a test URL that
       requires authentication (default: /feed/subscriptions, which returns
       nothing without a valid login session).

    Returns: (valid_cookie_count, auth_ok)
    Raises: ValueError with clear message if format validation fails
    """
    import shutil

    # Read and normalize line endings (CRLF -> LF)
    raw = cookies_path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized != raw:
        try:
            cookies_path.write_bytes(normalized)
        except OSError:
            pass

    text = normalized.decode("utf-8", errors="replace")
    lines = text.split("\n")

    valid_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) != 7:
            raise ValueError(
                f"Invalid cookie line {i}: expected 7 tab-separated fields, "
                f"got {len(parts)}. Line: {line[:100]}"
            )
        valid_lines.append(line)

    cookie_count = len(valid_lines)
    if cookie_count == 0:
        raise ValueError("No valid cookie data lines found (only comments/blanks)")

    # Check if yt-dlp is available before attempting auth check
    env = _subprocess_env()
    yt_dlp_bin = shutil.which("yt-dlp", path=env.get("PATH"))
    if not yt_dlp_bin:
        logger.warning("yt-dlp not found in PATH; skipping cookie auth check (format validation passed)")
        logger.info(f"Cookies validated: {cookie_count} data lines, auth_ok=False (yt-dlp unavailable)")
        return cookie_count, False

    # Lightweight auth verification: try to fetch a page that requires login.
    # Uses a writable temporary copy so yt-dlp doesn't fail on read-only master cookies.
    auth_ok = False
    temp_val_cookies = cookies_path.parent / f".val_{cookies_path.name}"
    try:
        shutil.copyfile(cookies_path, temp_val_cookies)
        temp_val_cookies.chmod(0o600)
        result = subprocess.run(
            [yt_dlp_bin, *_get_yt_dlp_common_args(), "--cookies", str(temp_val_cookies), "--skip-download",
             "--playlist-items", "1",
             "--print", "id", test_url],
            capture_output=True, text=True, timeout=60,
            env=env
        )
        if result.returncode == 0 and result.stdout.strip():
            auth_ok = True
            logger.info(f"Cookie auth verified: extracted id={result.stdout.strip()}")
        else:
            logger.warning(f"Cookie auth check failed: yt-dlp returned {result.returncode}, stderr={result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        logger.warning("Cookie auth check timed out (60s)")
    except Exception as exc:
        logger.warning(f"Cookie auth check error: {exc}")
    finally:
        if temp_val_cookies.exists():
            try:
                temp_val_cookies.unlink()
            except Exception:
                pass

    logger.info(f"Cookies validated: {cookie_count} data lines, auth_ok={auth_ok}")
    return cookie_count, auth_ok


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
    channel_input: Optional[str] = None,
    playlist_id: Optional[str] = None,
    search_query: Optional[str] = None,
    max_videos: int = 50,
) -> list[SourceVideo]:
    """List new long-form videos from a YouTube channel or search query."""
    # Load already-processed YouTube video IDs
    processed = set()
    for row in db.fetchall("SELECT source_etag FROM videos WHERE source_type='clipped' AND source_etag IS NOT NULL"):
        processed.add(row["source_etag"])

    items_to_process = []
    
    if search_query:
        # Fetch search results
        data = _youtube_api_request("search", {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": 50,
        })
        items_to_process = data.get("items", [])
    elif channel_input:
        channel_id = _resolve_channel_id(channel_input)
        if not playlist_id:
            playlist_id = _get_uploads_playlist_id(channel_id)

        # Fetch playlist items (newest first)
        data = _youtube_api_request("playlistItems", {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,  
        })
        items_to_process = data.get("items", [])
    else:
        raise RuntimeError("Must provide either channel_input or search_query")

    new_videos: list[SourceVideo] = []
    for item in items_to_process:
        snippet = item.get("snippet", {})
        
        # Depending on if it's from search or playlistItems, videoId is in different places
        if search_query:
            yt_video_id = item.get("id", {}).get("videoId", "")
        else:
            yt_video_id = item.get("contentDetails", {}).get("videoId", "")

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

        title = v["snippet"].get("title", "Untitled")
        title_lower = title.lower()

        # Skip pre-existing Shorts / clips: must be at least 120s long and not tagged as a Short
        if duration_secs < 120 or "#shorts" in title_lower or "#short" in title_lower or "#tiktok" in title_lower:
            logger.info(f"Skipping pre-existing Short/clip ({duration_secs}s): {title}")
            continue
        new_videos.append(SourceVideo(
            source_type="youtube",
            video_id=yt_video_id,
            title=title,
            size_bytes=0,  # unknown until download
            etag=yt_video_id,  # use video ID as de-dup key
            yt_video_id=yt_video_id,
            yt_channel_id=v["snippet"].get("channelId", ""),
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

    cookies_path_str = os.getenv("YT_COOKIES_PATH")
    has_cookies = cookies_path_str and Path(cookies_path_str).exists()

    video_url = f"https://www.youtube.com/watch?v={source.yt_video_id}"

    auth_ok = False
    if has_cookies:
        cookies_path = Path(cookies_path_str)
        try:
            cookie_count, auth_ok = validate_cookies_file(cookies_path, test_url=video_url)
            if not auth_ok:
                logger.warning(
                    f"Cookies failed authentication check for {video_url} "
                    f"({cookie_count} cookies loaded). yt-dlp might fail with bot detection."
                )
        except ValueError as exc:
            logger.warning(f"Cookie validation failed, proceeding without them: {exc}")
            has_cookies = False

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / f"{source.yt_video_id}.mp4"

    common_args = _get_yt_dlp_common_args()
    cmd = [
        "yt-dlp",
        *common_args,
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", str(local_path),
        video_url,
    ]

    working_cookies = None
    if has_cookies:
        import shutil
        # Copy to a temporary working cookie file so yt-dlp never overwrites or corrupts master cookies.txt
        working_cookies = dest_dir / ".working_cookies.txt"
        shutil.copyfile(cookies_path_str, working_cookies)
        working_cookies.chmod(0o600)
        insert_idx = 1 + len(common_args)
        cmd.insert(insert_idx, "--cookies")
        cmd.insert(insert_idx + 1, str(working_cookies))

    try:
        # Retry on bot-detection (transient), capped at 2 attempts
        last_err = ""
        env = _subprocess_env()
        yt_dlp_bin = shutil.which("yt-dlp", path=env.get("PATH")) or "yt-dlp"
        cmd[0] = yt_dlp_bin
        for attempt in range(2):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
                if result.returncode == 0:
                    break
                last_err = result.stderr[-2000:]
                logger.debug(f"yt-dlp stderr for {source.yt_video_id}: {last_err}")
                # Bot-detection is often transient; retry once
                if "confirm you're not a bot" in last_err.lower() and attempt == 0:
                    logger.warning(f"yt-dlp bot-detection hit for {source.yt_video_id}, retrying")
                    continue
                raise RuntimeError(f"yt-dlp failed: {last_err}")
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"yt-dlp timed out downloading {source.yt_video_id}")
        else:
            raise RuntimeError(f"yt-dlp failed after retry: {last_err}")
    finally:
        if working_cookies and working_cookies.exists():
            try:
                working_cookies.unlink()
            except Exception:
                pass

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
        elif os.getenv("CLIP_SOURCE_YT_CHANNEL_ID") or os.getenv("CLIP_SOURCE_YT_SEARCH_QUERY"):
            source_mode = "youtube"
        else:
            raise RuntimeError("No clip source configured. Set CLIP_SOURCE_S3_BUCKET, CLIP_SOURCE_YT_CHANNEL_ID, or CLIP_SOURCE_YT_SEARCH_QUERY")

    if source_mode == "s3":
        bucket = os.getenv("CLIP_SOURCE_S3_BUCKET")
        prefix = os.getenv("CLIP_SOURCE_S3_PREFIX", "")
        new_videos = list_new_videos_s3(db, bucket, prefix, max_files=max_videos_per_run)
        download_fn = download_video_s3
    elif source_mode == "youtube":
        channel_id = os.getenv("CLIP_SOURCE_YT_CHANNEL_ID")
        playlist_id = os.getenv("CLIP_SOURCE_YT_PLAYLIST_ID")
        search_query = os.getenv("CLIP_SOURCE_YT_SEARCH_QUERY")

        if search_query:
            clean_sq = search_query.strip(' "\'')
            if "|" in clean_sq:
                import random
                queries = [q.strip(' "\'') for q in clean_sq.split("|") if q.strip(' "\'')]
                if queries:
                    search_query = random.choice(queries)
                    logger.info(f"Multiple topics found in env. Randomly selected query: '{search_query}'")
            else:
                search_query = clean_sq

        new_videos = list_new_videos_youtube(db, channel_input=channel_id, playlist_id=playlist_id, search_query=search_query, max_videos=max_videos_per_run)
        download_fn = download_video_youtube
    else:
        raise RuntimeError(f"Unknown CLIP_SOURCE_MODE: {source_mode} (must be 's3' or 'youtube')")

    results = []
    for src in new_videos:
        local_path = None
        try:
            local_path = download_fn(src, Path(pipeline.settings.clip_input_dir))
            pipeline_result = pipeline.run(
                str(local_path),
                upload=upload,
                max_clips=max_clips_per_video,
                force_retranscribe=False,
            )
            any_uploaded = any(o.published for o in pipeline_result.outcomes) if upload else True
            video_row = db.fetchone(
                "SELECT id FROM videos WHERE topic=? ORDER BY created_at DESC LIMIT 1",
                (f"clip:{local_path.stem}",),
            )
            if video_row:
                mark_video_processed(db, src, local_path, video_row["id"])
            if upload and not any_uploaded:
                clip_errors = [o.error for o in pipeline_result.outcomes if o.error]
                err_msg = "; ".join(clip_errors) if clip_errors else "no clips were published"
                results.append((src, False, err_msg))
            else:
                results.append((src, True, None))
        except Exception as exc:
            logger.error(f"clipper failed for {src.video_id}: {exc}")
            results.append((src, False, str(exc)))
        finally:
            if local_path and local_path.exists():
                try:
                    local_path.unlink()
                    logger.info(f"cleaned up source video: {local_path}")
                except Exception as e:
                    logger.warning(f"failed to clean up source video {local_path}: {e}")
                
                # Also clean up the Whisper JSON cache file
                from app.clipper.transcribe import get_transcript_cache_path
                try:
                    cache_path = Path(get_transcript_cache_path(str(local_path), pipeline.settings.whisper_model_size))
                    if cache_path.exists():
                        cache_path.unlink()
                        logger.info(f"cleaned up transcript cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"failed to clean up transcript cache: {e}")

            # If upload was enabled, clean up the rendered clips & subtitles in output/ to avoid disk leaks
            if upload:
                try:
                    stem = local_path.stem if local_path else (src.yt_video_id or src.video_id)
                    output_dir = Path(pipeline.settings.clip_output_dir)
                    for clip_file in output_dir.glob(f"{stem}_clip_*"):
                        clip_file.unlink(missing_ok=True)
                        logger.info(f"cleaned up published clip file: {clip_file.name}")
                except Exception as e:
                    logger.warning(f"failed to clean up output clips for {src.video_id}: {e}")

    return results