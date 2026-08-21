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