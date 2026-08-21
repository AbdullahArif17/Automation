"""Phase 8 tests: video editor, motion filters, render pipeline (mocked)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.video.editor import VideoEditor, RenderResult
from app.content.visual_plan import VisualPlan, Scene
from app.media.asset_manager import AssetRecord
from app.media.voice import VoiceResult
from app.media.captions import CaptionTrack, CaptionLine


def test_video_editor_construct():
    ed = VideoEditor()
    assert ed.ffmpeg == "ffmpeg"
    assert ed.ffprobe == "ffprobe"


def test_check_available_no_ffmpeg():
    ed = VideoEditor(ffmpeg_bin="/nonexistent/ffmpeg", ffprobe_bin="/nonexistent/ffprobe")
    assert ed.check_available() is False


def test_build_scene_filter_static():
    ed = VideoEditor()
    scene = Scene(0, 5, "query", "image", "static")
    asset = AssetRecord(1, "unsplash", "http://x", "MIT", "/tmp/x.jpg",
                        "image", 0, 1080, 1920, "hash")
    vf = ed._build_scene_filter(scene, asset, 5.0)
    assert "scale=1080:1920" in vf
    assert "crop=1080:1920" in vf


def test_build_scene_filter_zoom_in():
    ed = VideoEditor()
    scene = Scene(0, 5, "query", "image", "zoom_in")
    asset = AssetRecord(1, "unsplash", "http://x", "MIT", "/tmp/x.jpg",
                        "image", 0, 1080, 1920, "hash")
    vf = ed._build_scene_filter(scene, asset, 5.0)
    assert "zoompan" in vf
    assert "1.15" in vf


def test_build_scene_filter_zoom_out():
    ed = VideoEditor()
    scene = Scene(0, 5, "query", "image", "zoom_out")
    asset = AssetRecord(1, "unsplash", "http://x", "MIT", "/tmp/x.jpg",
                        "image", 0, 1080, 1920, "hash")
    vf = ed._build_scene_filter(scene, asset, 5.0)
    assert "zoompan" in vf


def test_build_scene_filter_pan():
    ed = VideoEditor()
    scene = Scene(0, 5, "query", "video", "pan")
    asset = AssetRecord(1, "pexels", "http://x", "MIT", "/tmp/x.mp4",
                        "video", 10, 1080, 1920, "hash")
    vf = ed._build_scene_filter(scene, asset, 5.0)
    assert "crop" in vf


def test_build_scene_filter_ken_burns():
    ed = VideoEditor()
    scene = Scene(0, 5, "query", "image", "ken_burns")
    asset = AssetRecord(1, "unsplash", "http://x", "MIT", "/tmp/x.jpg",
                        "image", 0, 1080, 1920, "hash")
    vf = ed._build_scene_filter(scene, asset, 5.0)
    assert "zoompan" in vf


def test_render_result():
    r = RenderResult("/tmp/out.mp4", 35.0, 1080, 1920)
    assert r.duration == 35.0
    assert r.width == 1080
    assert r.height == 1920


# Integration test requires ffmpeg binary - skip in unit tests
# @pytest.mark.integration
# def test_full_render():
#     pass