"""Unit tests for the clipper pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.clipper.captions import build_caption_track_from_whisper
from app.clipper.cut import build_crop_filter
from app.clipper.highlight import ClipCandidate, parse_highlight_response
from app.clipper.storage_poller import validate_cookies_file
from app.clipper.transcribe import (
    SegmentTimestamp, TranscriptResult, WordTimestamp, get_transcript_cache_path
)
from app.media.captions import CaptionLine, CaptionTrack


# --- Transcript cache path ---
def test_transcript_cache_path():
    from pathlib import Path
    path = get_transcript_cache_path("/path/to/video.mp4", "base.en")
    assert Path(path).name == "video_transcript_base.en.json"
    assert path.endswith("video_transcript_base.en.json")

    path2 = get_transcript_cache_path("/path/to/video.mp4", "small")
    assert Path(path2).name == "video_transcript_small.json"


# --- Highlight response parsing ---
def test_parse_highlight_response_valid():
    response = json.dumps({
        "candidates": [
            {
                "start_seconds": 10.0,
                "end_seconds": 35.0,
                "reason": "Clear hook and complete thought",
                "suggested_title": "Amazing AI Discovery",
                "suggested_description": "This segment explains the breakthrough.",
                "confidence": 0.85
            },
            {
                "start_seconds": 60.0,
                "end_seconds": 100.0,
                "reason": "Visual demo of the concept",
                "suggested_title": "See It In Action",
                "suggested_description": "Watch the demo.",
                "confidence": 0.75
            }
        ]
    })
    candidates = parse_highlight_response(response, 20.0, 60.0, 300.0)
    assert len(candidates) == 2
    assert candidates[0].start_seconds == 10.0
    assert candidates[0].end_seconds == 35.0
    assert candidates[0].confidence == 0.85
    # Sorted by confidence descending
    assert candidates[0].confidence >= candidates[1].confidence


def test_parse_highlight_response_duration_out_of_bounds():
    """Candidates outside MIN/MAX duration should be filtered."""
    response = json.dumps({
        "candidates": [
            {"start_seconds": 10.0, "end_seconds": 15.0, "reason": "Too short",
             "suggested_title": "Short", "suggested_description": "Desc", "confidence": 0.9},
            {"start_seconds": 20.0, "end_seconds": 50.0, "reason": "Valid",
             "suggested_title": "Valid", "suggested_description": "Desc", "confidence": 0.8},
            {"start_seconds": 100.0, "end_seconds": 200.0, "reason": "Too long",
             "suggested_title": "Long", "suggested_description": "Desc", "confidence": 0.7},
        ]
    })
    candidates = parse_highlight_response(response, 20.0, 60.0, 300.0)
    assert len(candidates) == 1
    assert candidates[0].duration == 30.0


def test_parse_highlight_response_timestamp_out_of_video():
    """Candidates outside video duration should be filtered."""
    response = json.dumps({
        "candidates": [
            {"start_seconds": -5.0, "end_seconds": 30.0, "reason": "Negative start",
             "suggested_title": "Bad", "suggested_description": "Desc", "confidence": 0.9},
            {"start_seconds": 10.0, "end_seconds": 35.0, "reason": "Valid",
             "suggested_title": "Good", "suggested_description": "Desc", "confidence": 0.8},
            {"start_seconds": 100.0, "end_seconds": 400.0, "reason": "Past end",
             "suggested_title": "Bad", "suggested_description": "Desc", "confidence": 0.7},
        ]
    })
    candidates = parse_highlight_response(response, 20.0, 60.0, 300.0)
    assert len(candidates) == 1
    assert candidates[0].start_seconds == 10.0


def test_parse_highlight_response_code_fence():
    """Should extract JSON from markdown code fences."""
    response = """```json
    {"candidates": [{"start_seconds": 10.0, "end_seconds": 40.0, "reason": "Test",
    "suggested_title": "Test", "suggested_description": "Desc", "confidence": 0.8}]}
    ```"""
    candidates = parse_highlight_response(response, 20.0, 60.0, 300.0)
    assert len(candidates) == 1


def test_parse_highlight_response_missing_field():
    response = json.dumps({
        "candidates": [{
            "start_seconds": 10.0,
            "end_seconds": 40.0,
            # missing reason, title, etc.
        }]
    })
    with pytest.raises(ValueError, match="missing required field"):
        parse_highlight_response(response, 20.0, 60.0, 300.0)


def test_parse_highlight_response_no_candidates_key():
    response = json.dumps({"other": "data"})
    with pytest.raises(ValueError, match="missing 'candidates' key"):
        parse_highlight_response(response, 20.0, 60.0, 300.0)


def test_parse_highlight_response_all_invalid():
    response = json.dumps({
        "candidates": [
            {"start_seconds": 10.0, "end_seconds": 15.0, "reason": "Too short",
             "suggested_title": "Short", "suggested_description": "Desc", "confidence": 0.9},
        ]
    })
    with pytest.raises(ValueError, match="No valid candidates"):
        parse_highlight_response(response, 20.0, 60.0, 300.0)


# --- Crop filter ---
def test_build_crop_filter_16to9_to_9to16():
    """16:9 source (1920x1080) -> 9:16 (1080x1920) center crop."""
    filter_str = build_crop_filter("center", 1920, 1080, 1080, 1920)
    assert "crop=" in filter_str
    assert "scale=1080:1920" in filter_str
    # For 16:9 (1.78) > 9:16 (0.5625), crop sides
    # crop_w = 1080 * 0.5625 = 607.5 -> 607 or 608
    # x_offset = (1920 - 607) // 2


def test_build_crop_filter_vertical_source():
    """Vertical source (1080x1920) -> 9:16 no crop needed (just scale)."""
    filter_str = build_crop_filter("center", 1080, 1920, 1080, 1920)
    assert "crop=" in filter_str
    # src_ar = 0.5625 == target_ar, so falls into else branch (crop top/bottom with crop_w=src_w)
    # Actually since src_ar <= target_ar, it takes the else branch


def test_build_crop_filter_ultrawide():
    """Ultrawide (2560x1080) -> 9:16 center crop."""
    filter_str = build_crop_filter("center", 2560, 1080, 1080, 1920)
    assert "crop=" in filter_str
    assert "scale=1080:1920" in filter_str


# --- Caption track from whisper ---
def _make_transcript(words_data: list[tuple[str, float, float]], duration: float = 100.0) -> TranscriptResult:
    """Helper to create TranscriptResult from list of (word, start, end)."""
    words = [WordTimestamp(word=w, start=s, end=e, probability=1.0) for w, s, e in words_data]
    seg = SegmentTimestamp(text=" ".join(w for w, _, _ in words_data), start=0.0, end=duration, words=words)
    return TranscriptResult(
        language="en", language_probability=0.99, duration=duration,
        segments=[seg], source_path="test.mp4"
    )


def test_build_caption_track_from_whisper_basic():
    """Basic caption generation from whisper words."""
    # Clip from 30-60s, words at 31-33, 33-35, etc.
    transcript = _make_transcript([
        ("Hello", 31.0, 31.5),
        ("world", 31.5, 32.0),
        ("this", 33.0, 33.3),
        ("is", 33.3, 33.5),
        ("a", 33.5, 33.6),
        ("test", 33.6, 34.0),
    ], duration=100.0)

    clip = ClipCandidate(
        start_seconds=30.0, end_seconds=60.0,
        reason="test", suggested_title="Test", suggested_description="Test",
        confidence=0.9
    )

    track = build_caption_track_from_whisper(transcript, clip, max_chars_per_line=42, max_lines_per_caption=2)
    assert isinstance(track, CaptionTrack)
    assert len(track.lines) >= 1
    # Check times are relative to clip (0-30s window)
    for line in track.lines:
        assert line.start >= 0.0
        assert line.end <= clip.duration


def test_build_caption_track_from_whisper_empty_window():
    """No words in clip window -> fallback behavior."""
    transcript = _make_transcript([
        ("Hello", 10.0, 11.0),  # Outside clip window
    ], duration=100.0)

    clip = ClipCandidate(
        start_seconds=30.0, end_seconds=60.0,
        reason="test", suggested_title="Test Clip", suggested_description="Test",
        confidence=0.9
    )

    track = build_caption_track_from_whisper(transcript, clip)
    # Should still return a track (fallback uses suggested_title)
    assert isinstance(track, CaptionTrack)
    if track.lines:
        assert "Test Clip" in track.lines[0].text


def test_build_caption_track_whitespace_collapse():
    """Multiple spaces in transcript should collapse."""
    transcript = _make_transcript([
        ("Hello", 31.0, 31.5),
        ("world", 31.5, 32.0),
    ], duration=100.0)

    clip = ClipCandidate(
        start_seconds=30.0, end_seconds=60.0,
        reason="test", suggested_title="Test", suggested_description="Test",
        confidence=0.9
    )

    track = build_caption_track_from_whisper(transcript, clip)
    for line in track.lines:
        # No double spaces
        assert "  " not in line.text


def test_build_caption_track_word_clamping():
    """Words at clip boundaries should be clamped."""
    transcript = _make_transcript([
        ("Early", 29.5, 30.0),   # Starts before clip
        ("OnTime", 30.0, 30.5),  # Exactly at clip start
        ("Late", 59.5, 60.5),    # Ends after clip
    ], duration=100.0)

    clip = ClipCandidate(
        start_seconds=30.0, end_seconds=60.0,
        reason="test", suggested_title="Test", suggested_description="Test",
        confidence=0.9
    )

    track = build_caption_track_from_whisper(transcript, clip)
    for line in track.lines:
        assert line.start >= 0.0
        assert line.end <= 30.0  # clip duration


def test_build_caption_track_emoji_and_punctuation():
    """Emoji and punctuation should be preserved."""
    transcript = _make_transcript([
        ("Wow! 🎉", 31.0, 31.8),
        ("Amazing.", 31.8, 32.5),
    ], duration=100.0)

    clip = ClipCandidate(
        start_seconds=30.0, end_seconds=60.0,
        reason="test", suggested_title="Test", suggested_description="Test",
        confidence=0.9
    )

    track = build_caption_track_from_whisper(transcript, clip)
    text = " ".join(line.text for line in track.lines)
    assert "🎉" in text
    assert "!" in text
    assert "." in text


# --- ClipCandidate duration property ---
def test_clip_candidate_duration():
    c = ClipCandidate(10.0, 45.0, "reason", "title", "desc", 0.8)
    assert c.duration == 35.0


# --- Pipeline exit code behavior (mocked) ---
def test_pipeline_exits_nonzero_on_transcribe_failure(tmp_path):
    """CLI should return non-zero on transcription failure."""
    from app.clipper.__main__ import main

    # Non-existent file
    rc = main(["/nonexistent/video.mp4"])
    assert rc == 2  # File not found


def test_pipeline_exits_nonzero_on_max_clips_invalid(tmp_path):
    """CLI should return non-zero on invalid --max-clips."""
    from app.clipper.__main__ import main

    # Create dummy file
    dummy = tmp_path / "dummy.mp4"
    dummy.write_bytes(b"fake")

    rc = main([str(dummy), "--max-clips", "5"])
    assert rc == 2


# --- Duplicate check for clips ---
def test_clip_duplicate_check_only_against_clipped():
    """Duplicate check should only compare against source_type='clipped' videos."""
    from app.storage.database import Database
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(f"{tmpdir}/test.db")
        try:
            # Insert a generated video (should NOT be checked)
            db.insert("videos", {
                "topic": "AI generated", "title": "Generated Video", "description": "",
                "video_path": "/fake.mp4", "status": "PUBLISHED", "source_type": "generated",
                "youtube_video_id": "gen123", "created_at": "2026-01-01T00:00:00"
            })
            # Insert a clipped video (SHOULD be checked)
            db.insert("videos", {
                "topic": "clip:source", "title": "My Clipped Short", "description": "From my video",
                "video_path": "/fake2.mp4", "status": "PUBLISHED", "source_type": "clipped",
                "youtube_video_id": "clip123", "created_at": "2026-01-01T00:00:00"
            })

            from app.clipper.pipeline import ClipperPipeline
            from app.config.settings import Settings
            settings = Settings()
            pipeline = ClipperPipeline(db=db, settings=settings)

            # This title is similar to the clipped video
            result = pipeline._check_clip_duplicate("My Clipped Short", "From my video", "job1")
            assert result.is_duplicate is True
            assert result.reason != "no duplicates"

            # This title is similar to the GENERATED video only - should NOT match
            result2 = pipeline._check_clip_duplicate("AI generated", "Something else", "job1")
            # Generated videos are excluded from the check
            assert result2.is_duplicate is False
            assert result2.reason == "no duplicates"
        finally:
            db.close()


# --- Cookie validation ---
def test_validate_cookies_file_valid():
    """Valid Netscape-format cookies file passes validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cookies_path = Path(tmpdir) / "cookies.txt"
        # Valid 7-field lines (domain, flag, path, secure, expiration, name, value)
        cookies_path.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tTRUE\t1822391581\tPREF\tf6=40000000\n"
            ".youtube.com\tTRUE\t/\tFALSE\t1822330270\tLOGIN_INFO\tAFmmF2swRQIgO5Rh\n"
        )

        count, auth_ok = validate_cookies_file(cookies_path)
        assert count == 2
        # auth_ok will be False in test env (no real yt-dlp auth), but format passes
        assert isinstance(auth_ok, bool)


def test_validate_cookies_file_malformed_line():
    """Line with wrong field count raises ValueError with line number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cookies_path = Path(tmpdir) / "cookies.txt"
        # Line 3 has only 6 fields (missing tab between expiration and name = LOGIN_INFOmissing_tab as one field)
        cookies_path.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tTRUE\t1822391581\tPREF\tf6=40000000\n"
            ".youtube.com\tTRUE\t/\tFALSE\t1822330270\tLOGIN_INFOmissing_tab\n"
        )

        with pytest.raises(ValueError, match=r"Invalid cookie line 3: expected 7 tab-separated fields, got 6"):
            validate_cookies_file(cookies_path)


def test_validate_cookies_file_crlf_normalized():
    """CRLF line endings are normalized to LF and don't break parsing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cookies_path = Path(tmpdir) / "cookies.txt"
        # Write with CRLF (Windows-style)
        cookies_path.write_bytes(
            b"# Netscape HTTP Cookie File\r\n"
            b".youtube.com\tTRUE\t/\tTRUE\t1822391581\tPREF\tf6=40000000\r\n"
            b".youtube.com\tTRUE\t/\tFALSE\t1822330270\tLOGIN_INFO\tAFmmF2swRQIgO5Rh\r\n"
        )

        count, auth_ok = validate_cookies_file(cookies_path)
        assert count == 2


def test_validate_cookies_file_empty_raises():
    """File with only comments/blanks raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cookies_path = Path(tmpdir) / "cookies.txt"
        cookies_path.write_text("# Just a comment\n\n# Another comment\n")

        with pytest.raises(ValueError, match="No valid cookie data lines found"):
            validate_cookies_file(cookies_path)


def test_build_dynamic_crop_expr():
    from app.clipper.cut import _build_dynamic_crop_expr
    from app.clipper.face_tracker import ShotPlan

    # 1. Empty shots
    assert _build_dynamic_crop_expr([]) == "0"

    # 2. Single shot
    single = [ShotPlan(0.0, 10.0, 200, 0, 405, 720)]
    assert _build_dynamic_crop_expr(single) == "200"

    # 3. 2 shots
    shots2 = [
        ShotPlan(0.0, 15.0, 180, 0, 405, 720),
        ShotPlan(15.0, 30.0, 720, 0, 405, 720),
    ]
    assert _build_dynamic_crop_expr(shots2) == "if(lt(t,15.00),180,720)"

    # 4. 3 shots
    shots3 = [
        ShotPlan(0.0, 10.0, 100, 0, 405, 720),
        ShotPlan(10.0, 25.0, 500, 0, 405, 720),
        ShotPlan(25.0, 40.0, 800, 0, 405, 720),
    ]
    assert _build_dynamic_crop_expr(shots3) == "if(lt(t,10.00),100,if(lt(t,25.00),500,800))"


def test_build_crop_filter_dynamic_mode():
    from app.clipper.cut import build_crop_filter
    from app.clipper.face_tracker import FramingPlan, ShotPlan

    shots = [
        ShotPlan(0.0, 12.0, 150, 0, 405, 720),
        ShotPlan(12.0, 30.0, 650, 0, 405, 720),
    ]
    plan = FramingPlan(
        mode="dynamic",
        crop_x=150,
        crop_y=0,
        crop_w=405,
        crop_h=720,
        shots=shots,
    )
    filter_str = build_crop_filter("auto", 1280, 720, 1080, 1920, framing_plan=plan)
    assert "if(lt(t,12.00),150,650)" in filter_str
    assert "scale=1080:1920" in filter_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])