"""YouTube Analytics collection and topic optimization.

Collects video performance from YouTube Data API, stores in the analytics
table, and computes derived metrics (performance_score, engagement_rate,
views_per_hour). Uses historical data to adjust topic-scoring weights.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.storage.database import Database
from app.youtube.auth import YouTubeAuth
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

API_URL = "https://www.googleapis.com/youtube/v3/videos"
CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"


@dataclass
class VideoMetrics:
    video_id: str
    views: int
    likes: int
    comments: int
    watch_time: int = 0          # minutes, if available
    avg_view_duration: float = 0.0

    @property
    def engagement_rate(self) -> float:
        if self.views == 0:
            return 0.0
        return (self.likes + self.comments) / self.views

    def performance_score(self, duration_hours: float = 24.0) -> float:
        """0-100 normalized score: views/day + engagement."""
        views_per_day = self.views / max(duration_hours, 0.1)
        # Normalize: 1000 views/day = ~50 pts, engagement 0.05 = ~50 pts
        score = min(50.0, views_per_day / 20.0) + min(50.0, self.engagement_rate * 1000.0)
        return round(score, 2)


class AnalyticsCollector:
    def __init__(self, auth: YouTubeAuth | None = None, db: Database | None = None):
        self.auth = auth or YouTubeAuth()
        self.db = db or Database(self._db_path())

    def _db_path(self):
        from app.config.settings import get_settings
        return get_settings().db_path

    def collect_video(self, video_id: str, job_id: Optional[str] = None) -> VideoMetrics:
        """Fetch a single video's statistics from YouTube API."""
        if not self.auth.is_configured():
            raise RuntimeError("YouTube auth not configured; cannot collect analytics")

        creds = self.auth.credentials()
        req = urllib.request.Request(
            f"{API_URL}?part=statistics&id={video_id}",
            headers=creds.auth_header(),
        )

        def _get():
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:300]
                if exc.code == 403:
                    # Non-retryable permission error
                    raise PermissionError(f"HTTP 403: {detail}")
                raise RuntimeError(f"analytics fetch failed: {detail}")

        try:
            data = retry(_get, max_attempts=3, retry_on=(urllib.error.URLError, TimeoutError))
        except PermissionError:
            raise
        except Exception as exc:
            raise RuntimeError(f"analytics fetch failed: {exc}")

        items = data.get("items", [])
        if not items:
            raise RuntimeError(f"no video found for id {video_id}")

        stats = items[0].get("statistics", {})
        metrics = VideoMetrics(
            video_id=video_id,
            views=int(stats.get("viewCount", 0)),
            likes=int(stats.get("likeCount", 0)),
            comments=int(stats.get("commentCount", 0)),
        )

        # Store in DB
        self._store_metrics(metrics, job_id)
        logger.info(
            f"analytics: {video_id} views={metrics.views} likes={metrics.likes} "
            f"comments={metrics.comments} eng={metrics.engagement_rate:.3f}",
            extra={"job_id": job_id, "stage": "analytics", "status": "done"},
        )
        return metrics

    def _store_metrics(self, m: VideoMetrics, job_id: Optional[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT INTO analytics (video_id, youtube_video_id, views, likes, comments, "
            "watch_time, average_view_duration, collected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, m.video_id, m.views, m.likes, m.comments,
             m.watch_time, m.avg_view_duration, now),
        )
        # Update videos table too
        if job_id:
            self.db.execute(
                "UPDATE videos SET views=?, likes=?, comments=?, watch_time=?, "
                "average_view_duration=? WHERE id=?",
                (m.views, m.likes, m.comments, m.watch_time, m.avg_view_duration, job_id),
            )

    def collect_all_published(self) -> list[VideoMetrics]:
        """Collect analytics for all published videos lacking recent data."""
        rows = self.db.fetchall(
            "SELECT id, youtube_video_id FROM videos WHERE youtube_video_id IS NOT NULL"
        )
        results = []
        for r in rows:
            try:
                m = self.collect_video(r["youtube_video_id"], job_id=r["id"])
                results.append(m)
            except PermissionError as exc:
                logger.info("YouTube token lacks analytics/read scope; skipping pre-poll analytics sync",
                            extra={"stage": "analytics", "status": "skipped_scope"})
                break
            except Exception as exc:
                logger.warning(f"collect failed for {r['youtube_video_id']}: {exc}",
                               extra={"job_id": str(r["id"]), "stage": "analytics", "status": "error"})
        return results


class TopicOptimizer:
    """Adjusts topic-scoring weights based on historical performance."""

    def __init__(self, db: Database):
        self.db = db

    def compute_topic_performance(self) -> dict[str, float]:
        """Average performance score per topic (from published videos)."""
        rows = self.db.fetchall(
            "SELECT v.topic, a.views, a.likes, a.comments FROM videos v "
            "JOIN analytics a ON a.video_id = v.id WHERE v.youtube_video_id IS NOT NULL"
        )
        perf: dict[str, list[float]] = {}
        for r in rows:
            topic = r["topic"]
            m = VideoMetrics("", r["views"], r["likes"], r["comments"])
            score = m.performance_score()
            perf.setdefault(topic, []).append(score)

        return {t: sum(scores) / len(scores) for t, scores in perf.items()}

    def recommend_weight_adjustments(self, learning_rate: float = 0.1) -> dict[str, float]:
        """Increase weight for high-retention topics, decrease for low.

        Returns suggested delta per topic-category keyword.
        """
        perf = self.compute_topic_performance()
        if not perf:
            return {}

        avg = sum(perf.values()) / len(perf)
        adjustments: dict[str, float] = {}
        for topic, score in perf.items():
            delta = (score - avg) * learning_rate
            adjustments[topic] = round(delta, 3)
        return adjustments

    def get_best_topics(self, limit: int = 5) -> list[tuple[str, float]]:
        """Return top-performing topics by average performance score."""
        perf = self.compute_topic_performance()
        return sorted(perf.items(), key=lambda x: x[1], reverse=True)[:limit]
