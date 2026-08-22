"""Phase 11 tests: analytics collection, metrics, topic optimization."""
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from app.youtube.analytics import AnalyticsCollector, TopicOptimizer, VideoMetrics
from app.storage.database import Database


def _make_db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    return db


def test_video_metrics_engagement_rate():
    m = VideoMetrics("vid1", views=1000, likes=50, comments=50)
    assert m.engagement_rate == pytest.approx(0.1)
    zero = VideoMetrics("vid2", views=0, likes=0, comments=0)
    assert zero.engagement_rate == 0.0


def test_video_metrics_performance_score():
    m = VideoMetrics("vid1", views=20000, likes=1000, comments=0)
    score = m.performance_score()
    assert 0.0 <= score <= 100.0
    assert score > 50.0  # high views/day + decent engagement


def test_collect_video_not_configured(tmp_path):
    auth = MagicMock()
    auth.is_configured.return_value = False
    collector = AnalyticsCollector(auth=auth, db=_make_db(tmp_path))
    with pytest.raises(RuntimeError):
        collector.collect_video("abc123")


def test_collect_video_success(tmp_path):
    auth = MagicMock()
    auth.is_configured.return_value = True
    creds = MagicMock()
    creds.auth_header.return_value = {"Authorization": "Bearer x"}
    auth.credentials.return_value = creds

    db = _make_db(tmp_path)
    db.execute(
        "INSERT INTO videos (topic, status, created_at) VALUES (?, ?, ?)",
        ("AI tools", "PUBLISHED", "2026-01-01T00:00:00"),
    )
    job_id = db.fetchone("SELECT id FROM videos WHERE topic='AI tools'")["id"]
    db.execute(
        "UPDATE videos SET youtube_video_id=?, published_at=? WHERE id=?",
        ("yt_abc", "2026-01-01T00:00:00", job_id),
    )

    fake_resp = nullcontext(MagicMock(
        read=lambda: b'{"items":[{"statistics":{"viewCount":"5000","likeCount":"300","commentCount":"50"}}]}'
    ))

    collector = AnalyticsCollector(auth=auth, db=db)
    with patch("urllib.request.urlopen", return_value=fake_resp):
        m = collector.collect_video("yt_abc", job_id=job_id)

    assert m.views == 5000
    assert m.likes == 300
    assert m.comments == 50

    rows = db.fetchall("SELECT * FROM analytics WHERE youtube_video_id='yt_abc'")
    assert len(rows) == 1
    assert rows[0]["views"] == 5000

    v = db.fetchone("SELECT views, likes, comments FROM videos WHERE id=?", (job_id,))
    assert v["views"] == 5000 and v["likes"] == 300


def test_collect_video_empty_items(tmp_path):
    auth = MagicMock()
    auth.is_configured.return_value = True
    creds = MagicMock()
    creds.auth_header.return_value = {"Authorization": "Bearer x"}
    auth.credentials.return_value = creds

    db = _make_db(tmp_path)
    fake_resp = nullcontext(MagicMock(read=lambda: b'{"items":[]}'))
    collector = AnalyticsCollector(auth=auth, db=db)
    with patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(RuntimeError):
            collector.collect_video("missing")


def test_topic_optimizer_performance_and_adjustments(tmp_path):
    db = _make_db(tmp_path)
    # Two published videos on two topics with analytics rows.
    db.execute(
        "INSERT INTO videos (topic, status, youtube_video_id, created_at, published_at) "
        "VALUES (?,?,?,?,?)",
        ("AI tools", "PUBLISHED", "yt1", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    db.execute(
        "INSERT INTO videos (topic, status, youtube_video_id, created_at, published_at) "
        "VALUES (?,?,?,?,?)",
        ("Web dev", "PUBLISHED", "yt2", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    v1 = db.fetchone("SELECT id FROM videos WHERE youtube_video_id='yt1'")["id"]
    v2 = db.fetchone("SELECT id FROM videos WHERE youtube_video_id='yt2'")["id"]
    # AI tools: high views -> high score. Web dev: low views -> low score.
    db.execute(
        "INSERT INTO analytics (video_id, youtube_video_id, views, likes, comments, collected_at) "
        "VALUES (?,?,?,?,?,?)",
        (v1, "yt1", 20000, 1000, 200, "2026-01-02T00:00:00"),
    )
    db.execute(
        "INSERT INTO analytics (video_id, youtube_video_id, views, likes, comments, collected_at) "
        "VALUES (?,?,?,?,?,?)",
        (v2, "yt2", 100, 2, 1, "2026-01-02T00:00:00"),
    )

    opt = TopicOptimizer(db)
    perf = opt.compute_topic_performance()
    assert perf["AI tools"] > perf["Web dev"]

    adj = opt.recommend_weight_adjustments()
    # AI tools above average -> positive; Web dev below -> negative.
    assert adj["AI tools"] > 0
    assert adj["Web dev"] < 0

    best = opt.get_best_topics(limit=2)
    assert best[0][0] == "AI tools"


def test_topic_optimizer_empty(tmp_path):
    db = _make_db(tmp_path)
    opt = TopicOptimizer(db)
    assert opt.compute_topic_performance() == {}
    assert opt.recommend_weight_adjustments() == {}
    assert opt.get_best_topics() == []
