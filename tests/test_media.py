"""Phase 5 tests: visual planner, image/video providers, asset manager."""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.content.visual_plan import VisualPlanner, Scene, VisualPlan
from app.media.image_provider import ImageAsset, UnsplashSourceProvider, PexelsProvider
from app.media.stock_provider import VideoAsset, PexelsVideoProvider, PixabayVideoProvider
from app.media.asset_manager import AssetManager
from app.ai.provider import MockProvider


def test_scene_dataclass():
    s = Scene(0, 5, "query", "image", "zoom_in")
    d = s.to_dict()
    assert d["start"] == 0
    assert d["visual_type"] == "image"


def test_visual_plan_mock():
    p = MockProvider([json.dumps({
        "duration": 35,
        "scenes": [
            {"start": 0, "end": 5, "visual_query": "AI coding", "visual_type": "image", "motion": "zoom_in"},
            {"start": 5, "end": 12, "visual_query": "laptop code", "visual_type": "video", "motion": "pan"},
        ]
    })])
    planner = VisualPlanner(p)
    plan = planner.plan("script text", "topic", 35)
    assert isinstance(plan, VisualPlan)
    assert len(plan.scenes) == 2
    assert plan.duration == 35
    assert plan.scenes[0].visual_type == "image"
    assert plan.scenes[1].visual_type == "video"


def test_unsplash_provider_construct():
    u = UnsplashSourceProvider()
    assert u.BASE == "https://source.unsplash.com"


def test_pexels_provider_no_key():
    p = PexelsProvider(api_key=None)
    # Should not crash, just return empty
    assert p.search("test") == []


def test_pexels_video_no_key():
    p = PexelsVideoProvider(api_key=None)
    assert p.search("test") == []


def test_pixabay_video_no_key():
    p = PixabayVideoProvider(api_key=None)
    assert p.search("test") == []


def test_asset_manager_construct():
    mock_db = MagicMock()
    mock_db.fetchone.return_value = None
    mock_db.execute.return_value.lastrowid = 1
    mgr = AssetManager(mock_db)
    assert mgr.unsplash is not None


def test_asset_manager_cache_hit():
    mock_db = MagicMock()
    mock_row = MagicMock()
    mock_row.__getitem__ = lambda self, k: {
        "id": 1, "source": "unsplash", "source_url": "http://x",
        "license": "MIT", "local_path": "/tmp/x.jpg", "type": "image",
        "duration": 0, "width": 1080, "height": 1920, "hash": "abc"
    }[k]
    mock_db.fetchone.return_value = mock_row
    mgr = AssetManager(mock_db)
    rec = mgr.get_or_fetch_image("test")
    assert rec is not None
    assert rec.source == "unsplash"


def test_visual_plan_validates_timing():
    p = MockProvider([json.dumps({
        "duration": 30,
        "scenes": [
            {"start": 2, "end": 8, "visual_query": "q", "visual_type": "image", "motion": "static"},
            {"start": 8, "end": 20, "visual_query": "q2", "visual_type": "video", "motion": "pan"},
        ]
    })])
    planner = VisualPlanner(p)
    plan = planner.plan("script", "topic", 30)
    # First scene should be adjusted to start at 0
    assert plan.scenes[0].start == 0
    # Last scene should end at duration
    assert abs(plan.scenes[-1].end - 30) < 1.0


def test_asset_manager_fallback_unsplash_to_pexels():
    """When Unsplash fails, Pexels should be tried before color-fill fallback."""
    from app.media.image_provider import ImageAsset, UnsplashSourceProvider, PexelsProvider
    from app.media.asset_manager import AssetManager, AssetRecord

    mock_db = MagicMock()
    mock_db.fetchone.return_value = None  # no cache hit

    # Track which provider gets called
    calls = []

    # Mock Unsplash to fail
    class FailingUnsplash(UnsplashSourceProvider):
        def fetch(self, query, width=1080, height=1920, job_id=None):
            calls.append("unsplash")
            raise RuntimeError("Unsplash 503")

    # Mock Pexels to succeed
    class WorkingPexels(PexelsProvider):
        def search(self, query, job_id=None):
            calls.append("pexels")
            return [ImageAsset(
                url="https://pexels.test/img.jpg",
                local_path="/tmp/pexels_test.jpg",
                width=1080, height=1920,
                license="Pexels License", source="pexels", hash="abc123"
            )]

    failing_unsplash = FailingUnsplash()
    working_pexels = WorkingPexels(api_key="test_key")

    mgr = AssetManager(mock_db, unsplash=failing_unsplash, pexels_img=working_pexels)

    # Mock _store_asset to return a record without DB
    stored = {}
    def mock_store(asset, job_id=None):
        stored["asset"] = asset
        return AssetRecord(
            id=1, source=asset.source, source_url=asset.url, license=asset.license,
            local_path=asset.local_path, type="image", duration=0,
            width=asset.width, height=asset.height, hash=asset.hash
        )
    mgr._store_asset = mock_store

    result = mgr.get_or_fetch_image("test query", job_id="test")

    # Should have tried unsplash first, then pexels
    assert calls == ["unsplash", "pexels"]
    # Should have returned the Pexels asset
    assert result is not None
    assert result.source == "pexels"
    assert stored["asset"].source == "pexels"


def test_asset_manager_all_providers_fail_returns_none():
    """When ALL providers fail, get_or_fetch_image returns None (color-fill fallback in caller)."""
    from app.media.image_provider import UnsplashSourceProvider, PexelsProvider
    from app.media.asset_manager import AssetManager

    mock_db = MagicMock()
    mock_db.fetchone.return_value = None

    class FailingUnsplash(UnsplashSourceProvider):
        def fetch(self, query, width=1080, height=1920, job_id=None):
            raise RuntimeError("Unsplash 503")

    class FailingPexels(PexelsProvider):
        def search(self, query, job_id=None):
            raise RuntimeError("Pexels 429")

    mgr = AssetManager(mock_db, unsplash=FailingUnsplash(), pexels_img=FailingPexels(api_key="key"))
    result = mgr.get_or_fetch_image("test query", job_id="test")
    assert result is None