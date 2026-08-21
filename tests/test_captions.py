"""Phase 7 tests: caption generation, SRT, ASS output."""
import tempfile
from pathlib import Path

import pytest

from app.media.captions import (
    CaptionLine, CaptionTrack,
    split_into_caption_lines, to_srt, to_ass, write_caption_files,
    estimate_word_duration,
)


def test_estimate_word_duration():
    assert estimate_word_duration(150) == 0.4
    assert estimate_word_duration(180) == 60/180


def test_split_into_caption_lines_basic():
    script = "Hello world this is a test script for captions"
    track = split_into_caption_lines(script, total_duration=10.0)
    assert isinstance(track, CaptionTrack)
    assert len(track.lines) > 0
    # Last caption should end at total_duration
    assert track.lines[-1].end == 10.0


def test_split_respects_max_chars():
    script = "a " * 50  # many short words
    track = split_into_caption_lines(script, total_duration=20.0, max_chars_per_line=20)
    # Individual wrapped lines (split by \n) should respect max_chars approximately
    for line in track.lines:
        for wrapped in line.text.split("\n"):
            assert len(wrapped) <= 30  # generous bound for word-wrapping


def test_split_handles_emphasis():
    script = "Normal text *EMPHASIS* and MORE emphasis"
    track = split_into_caption_lines(script, total_duration=5.0)
    assert any(line.emphasis for line in track.lines)


def test_srt_format():
    track = CaptionTrack(lines=[
        CaptionLine(1, 0.0, 2.5, "Hello world"),
        CaptionLine(2, 2.5, 5.0, "This is a test"),
    ])
    srt = to_srt(track)
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "Hello world" in srt
    assert "This is a test" in srt
    # Should have blank lines between entries
    assert "\n\n" in srt


def test_ass_format():
    track = CaptionTrack(lines=[
        CaptionLine(1, 0.0, 2.5, "Hello world"),
        CaptionLine(2, 2.5, 5.0, "Emphasis *here*"),
    ])
    ass = to_ass(track)
    assert "Dialogue:" in ass
    assert "Hello world" in ass
    # Emphasis line should use Emphasis style
    assert "Emphasis" in ass


def test_write_caption_files():
    track = CaptionTrack(lines=[
        CaptionLine(1, 0.0, 2.0, "Test"),
    ])
    with tempfile.TemporaryDirectory() as td:
        paths = write_caption_files(track, str(Path(td) / "captions"), formats=["srt", "ass"])
        assert "srt" in paths
        assert "ass" in paths
        assert Path(paths["srt"]).exists()
        assert Path(paths["ass"]).exists()
        # Content check
        srt_content = Path(paths["srt"]).read_text()
        assert "Test" in srt_content


def test_caption_timing_monotonic():
    script = "word " * 100
    track = split_into_caption_lines(script, total_duration=30.0)
    for i in range(1, len(track.lines)):
        assert track.lines[i].start >= track.lines[i-1].start
        assert track.lines[i].end >= track.lines[i].end


def test_empty_script():
    track = split_into_caption_lines("", total_duration=10.0)
    assert len(track.lines) == 0