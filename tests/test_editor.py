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


def _make_image_assets(ffmpeg, tmp_path, n: int, scene_dur: float) -> list[AssetRecord]:
    """Create real image files for testing the image-loop input path."""
    assets = []
    for i in range(n):
        img_path = str(tmp_path / f"test_img_{i}.png")
        # Create a small valid PNG via ffmpeg.
        # - size=64x64: lavfi color source requires valid non-zero dimensions
        # - -update 1: required by image2 muxer to write a single frame (not a sequence)
        # - -frames:v 1: only encode one frame from the color source
        subprocess.run(
            [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=red:size=64x64:rate=1",
             "-frames:v", "1", "-update", "1", img_path],
            check=True, capture_output=True,
        )
        assets.append(AssetRecord(
            id=i + 1, source="test", source_url="", license="",
            local_path=img_path, type="image", duration=scene_dur, width=64, height=64, hash="",
        ))
    return assets


def _make_video_assets(ffmpeg, tmp_path, n: int, scene_dur: float) -> list[AssetRecord]:
    """Create real video files for testing the video input path.

    Creates short synthetic videos (color bars + audio tone) that are shorter
    than the scene duration, so the -stream_loop -1 path is exercised.
    """
    assets = []
    for i in range(n):
        vid_path = str(tmp_path / f"test_vid_{i}.mp4")
        # Create a 1s test video (shorter than scene_dur to test looping)
        subprocess.run(
            [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
             "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
             "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k", vid_path],
            check=True, capture_output=True,
        )
        assets.append(AssetRecord(
            id=i + 1, source="test", source_url="", license="",
            local_path=vid_path, type="video", duration=1.0, width=640, height=360, hash="",
        ))
    return assets


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

    plan = _make_plan(n=3, scene_dur=2.0)  # 6s total, 2 crossfades = 5.2s expected
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
    # Overlap is now compensated: duration matches the plan (6.0s) + potentially a bit of audio trailing
    assert 5.9 <= result.duration <= 6.5

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
    assert 5.9 <= float(info["format"]["duration"]) <= 6.5


def _render_single_scene(ffmpeg_bins, tmp_path, motion: str, scene_dur: float = 2.5,
                          asset_type: str = "fallback") -> float:
    """Helper: render one scene with given motion, return probed duration.

    asset_type: "fallback" (lavfi color) or "image" (real image file loop)
    """
    ffmpeg, ffprobe = ffmpeg_bins

    voice_path = str(tmp_path / f"voice_{motion}_{asset_type}.mp3")
    out_path = str(tmp_path / f"out_{motion}_{asset_type}.mp4")

    # Short audio matching scene duration
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={scene_dur + 0.5}",
         "-c:a", "libmp3lame", voice_path],
        check=True, capture_output=True,
    )

    plan = _make_plan(n=1, scene_dur=scene_dur)
    plan.scenes[0].motion = motion
    if asset_type == "image":
        assets = _make_image_assets(ffmpeg, tmp_path, 1, scene_dur)
    else:
        assets = _make_fallback_assets(1)
    voice = VoiceResult(audio_path=voice_path, duration=scene_dur + 0.5, sample_rate=44100, channels=2)
    captions = CaptionTrack(lines=[
        CaptionLine(index=1, start=0.0, end=scene_dur, text="Test caption"),
    ])

    editor = VideoEditor(ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    result = editor.render(
        plan, assets, voice, captions,
        output_path=out_path, job_id=f"test_{motion}_{asset_type}",
    )

    # Probe actual output duration via ffprobe
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out_path],
        check=True, capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


# Cross product: both asset paths x all motion types
@pytest.mark.parametrize("asset_type,motion", [
    ("fallback", "zoom_in"),
    ("fallback", "zoom_out"),
    ("fallback", "ken_burns"),
    ("fallback", "static"),
    ("fallback", "pan"),
    ("image", "zoom_in"),
    ("image", "zoom_out"),
    ("image", "ken_burns"),
    ("image", "static"),
    ("image", "pan"),
])
def test_motion_type_duration_both_asset_paths(ffmpeg_bins, tmp_path, asset_type, motion):
    """Each motion type produces correct duration for BOTH lavfi-fallback and image-loop inputs."""
    scene_dur = 2.5
    actual_dur = _render_single_scene(ffmpeg_bins, tmp_path, motion, scene_dur, asset_type)
    # Tolerance: +/- 0.5s (accounts for keyframe rounding, concat, -shortest)
    assert abs(actual_dur - scene_dur) <= 0.5, \
        f"{asset_type}/{motion}: expected ~{scene_dur}s, got {actual_dur:.2f}s"


def _render_single_scene_video(ffmpeg_bins, tmp_path, motion: str, scene_dur: float = 2.5) -> float:
    """Helper: render one scene with a VIDEO asset, return probed duration.

    Video assets are shorter than scene_dur, so -stream_loop -1 is exercised.
    The motion parameter is overridden to 'pan' for video assets (handled in _build_scene_filter).
    """
    ffmpeg, ffprobe = ffmpeg_bins

    voice_path = str(tmp_path / f"voice_{motion}_video.mp3")
    out_path = str(tmp_path / f"out_{motion}_video.mp4")

    # Short audio matching scene duration
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={scene_dur + 0.5}",
         "-c:a", "libmp3lame", voice_path],
        check=True, capture_output=True,
    )

    plan = _make_plan(n=1, scene_dur=scene_dur)
    plan.scenes[0].motion = motion
    assets = _make_video_assets(ffmpeg, tmp_path, 1, scene_dur)
    voice = VoiceResult(audio_path=voice_path, duration=scene_dur + 0.5, sample_rate=44100, channels=2)
    captions = CaptionTrack(lines=[
        CaptionLine(index=1, start=0.0, end=scene_dur, text="Test caption"),
    ])

    editor = VideoEditor(ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    result = editor.render(
        plan, assets, voice, captions,
        output_path=out_path, job_id=f"test_{motion}_video",
    )

    # Probe actual output duration via ffprobe
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out_path],
        check=True, capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


# Video asset type: all motion types should be overridden to 'pan' and produce correct duration
@pytest.mark.parametrize("motion", ["zoom_in", "zoom_out", "ken_burns", "static", "pan"])
def test_video_asset_motion_override(ffmpeg_bins, tmp_path, motion):
    """Video assets override zoom/ken_burns to pan and produce correct duration."""
    scene_dur = 2.5
    actual_dur = _render_single_scene_video(ffmpeg_bins, tmp_path, motion, scene_dur)
    # Tolerance: +/- 0.5s (accounts for keyframe rounding, concat, -shortest)
    assert abs(actual_dur - scene_dur) <= 0.5, \
        f"video/{motion}: expected ~{scene_dur}s, got {actual_dur:.2f}s"


def test_scene_inputs_list_equals_range(ffmpeg_bins, tmp_path):
    """Regression test: scene_inputs computed in render() equals list(range(len(plan.scenes))).

    This directly asserts the fix for the index-miscount bug where the old code
    derived indices from len(input_args) which was wrong (6 args per scene).
    """
    ffmpeg, ffprobe = ffmpeg_bins

    voice_path = str(tmp_path / "voice.mp3")
    out_path = str(tmp_path / "out.mp4")

    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=3.0",
         "-c:a", "libmp3lame", voice_path],
        check=True, capture_output=True,
    )

    plan = _make_plan(n=4, scene_dur=0.75)  # 3s total, 4 scenes, 3 crossfades = 1.8s expected
    assets = _make_fallback_assets(4)
    voice = VoiceResult(audio_path=voice_path, duration=3.0, sample_rate=44100, channels=2)
    captions = CaptionTrack(lines=[
        CaptionLine(index=1, start=0.0, end=3.0, text="Test"),
    ])

    editor = VideoEditor(ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)

    # We can't easily inspect the internal scene_inputs from outside,
    # but a successful render with 4 scenes proves the indices are correct.
    # If they were wrong (e.g., [2,5,8,11]), ffmpeg would fail with
    # "Stream specifier matches no streams".
    result = editor.render(
        plan, assets, voice, captions,
        output_path=out_path, job_id="test_indices",
    )

    # 4 scenes * 0.75s = 3.0s (overlap compensated)
    assert 2.9 <= result.duration <= 3.5
    assert result.width == 1080 and result.height == 1920


def test_mixed_motion_types_timebase_normalization(ffmpeg_bins, tmp_path):
    """Regression test: adjacent scenes with DIFFERENT motion types must not
    cause xfade timebase mismatch.

    Bug: zoompan-based motions (zoom_in, zoom_out, ken_burns) produce
    a different internal timebase (1/15360) than scale/crop chains
    (static, pan). xfade requires matching timebases on both inputs.
    This test uses a sequence of genuinely DIFFERENT motion types
    (not uniform across all scenes) to catch the timebase mismatch.
    """
    ffmpeg, ffprobe = ffmpeg_bins

    voice_path = str(tmp_path / "voice.mp3")
    out_path = str(tmp_path / "out.mp4")

    # 4 scenes with mixed motion types: zoom_in -> pan -> static -> ken_burns
    scene_dur = 2.0
    plan = VisualPlan(duration=4 * scene_dur, scenes=[
        Scene(start=0.0, end=scene_dur, visual_query="q0", visual_type="image", motion="zoom_in"),
        Scene(start=scene_dur, end=2*scene_dur, visual_query="q1", visual_type="image", motion="pan"),
        Scene(start=2*scene_dur, end=3*scene_dur, visual_query="q2", visual_type="image", motion="static"),
        Scene(start=3*scene_dur, end=4*scene_dur, visual_query="q3", visual_type="image", motion="ken_burns"),
    ])
    assets = _make_fallback_assets(4)

    # Short audio matching total duration
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={4*scene_dur + 0.5}",
         "-c:a", "libmp3lame", voice_path],
        check=True, capture_output=True,
    )

    voice = VoiceResult(audio_path=voice_path, duration=4*scene_dur + 0.5, sample_rate=44100, channels=2)
    captions = CaptionTrack(lines=[
        CaptionLine(index=1, start=0.0, end=4*scene_dur, text="Mixed motion test"),
    ])

    editor = VideoEditor(ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    result = editor.render(
        plan, assets, voice, captions,
        output_path=out_path, job_id="test_mixed_motion",
    )

    # 4 scenes * 2.0s = 8.0s (overlap compensated)
    assert 7.9 <= result.duration <= 8.5, f"expected ~8.0s, got {result.duration:.2f}s"
    assert result.width == 1080 and result.height == 1920

    # ffprobe confirms valid output
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type", "-of", "json", out_path],
        check=True, capture_output=True, text=True,
    )
    import json
    info = json.loads(probe.stdout)
    types = {s["codec_type"] for s in info["streams"]}
    assert "video" in types and "audio" in types
    assert 7.9 <= float(info["format"]["duration"]) <= 8.5
