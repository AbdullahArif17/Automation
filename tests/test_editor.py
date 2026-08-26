"""Tests for app.video.editor (FFmpeg composition).

Covers the scene-input-index regression (each scene must map to ffmpeg
input index i, not a length-derived miscount) and an end-to-end render
that actually invokes ffmpeg and validates the output file.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.video.editor import VideoEditor
from app.content.visual_plan import VisualPlan, Scene
from app.media.asset_manager import AssetRecord
from app.media.voice import VoiceResult
from app.media.captions import CaptionTrack, CaptionLine


def _get_ffmpeg_bins():
    """Return (ffmpeg, ffprobe) paths, or None if neither is available.

    Prefers a system-installed ffmpeg (present in CI via apt), then falls
    back to static-ffmpeg if it happens to be installed locally.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    try:
        import static_ffmpeg.run as sfrun
        return sfrun.get_or_fetch_platform_executables_else_raise()
    except Exception:
        return None


def _make_fallback_assets(n: int) -> list[AssetRecord]:
    """AssetRecords with no local_path -> editor uses color-fallback scenes."""
    return [
        AssetRecord(
            id=i + 1, source="x", source_url="", license="",
            local_path="", type="image", duration=2.0, width=0, height=0, hash="",
        )
        for i in range(n)
    ]


def _make_plan(n: int = 3, scene_dur: float = 2.0):
    scenes = []
    t = 0.0
    for i in range(n):
        scenes.append(Scene(
            start=t, end=t + scene_dur, visual_query=f"q{i}",
            visual_type="image", motion=["zoom_in", "ken_burns", "static"][i % 3],
        ))
        t += scene_dur
    return VisualPlan(duration=t, scenes=scenes)


# ---------------------------------------------------------------------------
# Unit test: the index miscount regression (no ffmpeg required).
# ---------------------------------------------------------------------------
def test_scene_input_indices_is_direct_range():
    """scene i must map to ffmpeg input index i, not a length-derived value."""
    editor = VideoEditor()
    for n in (0, 1, 3, 5, 10):
        assert editor._scene_input_indices(n) == list(range(n))


def test_scene_input_indices_matches_plan_length():
    """For any multi-scene plan, the computed input indices equal range(N)."""
    editor = VideoEditor()
    plan = _make_plan(n=4)
    assert editor._scene_input_indices(len(plan.scenes)) == list(range(len(plan.scenes)))


# ---------------------------------------------------------------------------
# Integration test: actually render multi-scene video through ffmpeg.
# Skipped where ffmpeg/ffprobe are unavailable (keeps CI deterministic).
# ---------------------------------------------------------------------------
@pytest.fixture
def ffmpeg_bins():
    bins = _get_ffmpeg_bins()
    if bins is None:
        pytest.skip("ffmpeg/ffprobe not available")
    return bins


def test_render_multi_scene_end_to_end(ffmpeg_bins, tmp_path):
    """Render 3 color-fallback scenes + voice + music; verify ffmpeg exits 0
    and the output is a valid 1080x1920 MP4 of the expected duration.

    This is the real proof the scene-input miscount is gone: previously the
    wrong indices caused 'Stream specifier matches no streams' and a failure.
    Including music also exercises voice_idx / music_idx resolution.
    """
    ffmpeg, ffprobe = ffmpeg_bins

    voice_path = str(tmp_path / "voice.mp3")
    music_path = str(tmp_path / "music.mp3")
    out_path = str(tmp_path / "out.mp4")

    # Synthetic audio tracks via ffmpeg.
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=6.5",
         "-c:a", "libmp3lame", voice_path],
        check=True, capture_output=True,
    )
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=6.5",
         "-c:a", "libmp3lame", music_path],
        check=True, capture_output=True,
    )

    plan = _make_plan(n=3, scene_dur=2.0)  # 6s total
    assets = _make_fallback_assets(3)
    voice = VoiceResult(audio_path=voice_path, duration=6.5, sample_rate=44100, channels=2)
    captions = CaptionTrack(lines=[
        CaptionLine(index=1, start=0.0, end=2.0, text="Hello world"),
        CaptionLine(index=2, start=2.0, end=4.0, text="This is a test"),
        CaptionLine(index=3, start=4.0, end=6.0, text="End of clip"),
    ])

    editor = VideoEditor(ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    result = editor.render(
        plan, assets, voice, captions,
        music_path=music_path, music_volume=0.1,
        output_path=out_path, job_id="test",
    )

    # Output sanity.
    assert os.path.getsize(out_path) > 1000
    assert result.width == 1080 and result.height == 1920
    # -shortest ends at the 6s video concat.
    assert 5.5 <= result.duration <= 6.5

    # ffprobe confirms a real, playable file with video + audio streams.
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type", "-of", "json", out_path],
        check=True, capture_output=True, text=True,
    )
    import json
    info = json.loads(probe.stdout)
    types = {s["codec_type"] for s in info["streams"]}
    assert "video" in types and "audio" in types
    assert 5.5 <= float(info["format"]["duration"]) <= 6.5
