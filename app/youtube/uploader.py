"""YouTube uploader (Data API v3, resumable upload, stdlib urllib).

Uploads a finished MP4 with title/description/tags/privacy status. Stores the
returned video ID. Publishing is controlled by settings (AUTO_UPLOAD=false).

Security: never logs tokens; safe subprocess not needed (pure urllib).
"""
from __future__ import annotations

import json
import mimetypes
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from app.youtube.auth import YouTubeAuth, YouTubeCredentials
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
API_URL = "https://www.googleapis.com/youtube/v3/videos"


@dataclass
class UploadResult:
    video_id: str
    title: str
    privacy_status: str
    url: str


class YouTubeUploader:
    def __init__(self, auth: YouTubeAuth | None = None):
        self.auth = auth or YouTubeAuth()

    def _ensure_auth(self) -> YouTubeCredentials:
        if not self.auth.is_configured():
            raise RuntimeError("YouTube auth not configured; cannot upload")
        return self.auth.credentials()

    def upload(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        privacy_status: str = "private",
        category_id: str = "28",  # Science & Technology
        job_id: Optional[str] = None,
    ) -> UploadResult:
        creds = self._ensure_auth()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"video not found: {video_path}")

        # 1. Initiate resumable session
        metadata = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags[:500],
                "categoryId": category_id,
                "defaultLanguage": "en-US",
                "defaultAudioLanguage": "en-US",
            },
            "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False},
        }
        meta_json = json.dumps(metadata).encode("utf-8")

        init_req = urllib.request.Request(
            f"{UPLOAD_URL}?uploadType=resumable&part=snippet,status,statistics",
            data=meta_json,
            headers={
                **creds.auth_header(),
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(os.path.getsize(video_path)),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(init_req, timeout=30) as resp:
                upload_url = resp.headers.get("Location")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            logger.error(f"upload init failed: {detail}",
                         extra={"job_id": job_id, "stage": "youtube_upload", "status": "error", "error": detail})
            raise RuntimeError(f"YouTube upload init failed: {detail}")

        if not upload_url:
            raise RuntimeError("No upload URL returned from YouTube")

        # 2. Upload the file bytes
        content_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
        file_size = os.path.getsize(video_path)

        def _put():
            with open(video_path, "rb") as f:
                data = f.read()
            req = urllib.request.Request(
                upload_url,
                data=data,
                headers={
                    **creds.auth_header(),
                    "Content-Type": content_type,
                    "Content-Length": str(file_size),
                },
                method="PUT",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            result = retry(_put, max_attempts=3, retry_on=(urllib.error.URLError, TimeoutError))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            logger.error(f"upload failed: {detail}",
                         extra={"job_id": job_id, "stage": "youtube_upload", "status": "error", "error": detail})
            raise RuntimeError(f"YouTube upload failed: {detail}")

        video_id = result.get("id")
        if not video_id:
            raise RuntimeError("No video ID in YouTube response")

        # Never log the URL with token; build public URL
        url = f"https://youtu.be/{video_id}"
        logger.info(f"uploaded video {video_id} ({privacy_status})",
                    extra={"job_id": job_id, "stage": "youtube_upload", "status": "done"})
        return UploadResult(video_id=video_id, title=title,
                            privacy_status=privacy_status, url=url)

    def get_video(self, video_id: str) -> dict:
        """Fetch video metadata (statistics) by ID."""
        creds = self._ensure_auth()
        req = urllib.request.Request(
            f"{API_URL}?part=snippet,statistics&id={video_id}",
            headers=creds.auth_header(),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
