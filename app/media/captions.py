"""Local caption generation.

Generates timed captions from script text by estimating word timing.
Outputs SRT and ASS formats for burning with FFmpeg.
No paid APIs - fully local, deterministic.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CaptionLine:
    index: int
    start: float      # seconds
    end: float        # seconds
    text: str
    emphasis: bool = False  # highlight key words


@dataclass
class CaptionTrack:
    lines: list[CaptionLine]


def estimate_word_duration(words_per_minute: float = 150) -> float:
    """Seconds per word at given WPM."""
    return 60.0 / words_per_minute


def split_into_caption_lines(
    script: str,
    total_duration: float,
    max_chars_per_line: int = 42,
    max_lines_per_caption: int = 2,
    words_per_minute: float = 150,
) -> CaptionTrack:
    """Split script into timed caption lines.

    Algorithm:
    1. Split script into words
    2. Estimate time per word from total_duration / word_count
    3. Group words into lines (max chars)
    4. Group lines into caption blocks (max 2 lines)
    5. Assign start/end times proportionally
    """
    words = script.split()
    if not words:
        return CaptionTrack(lines=[])

    word_dur = total_duration / len(words)
    lines = []
    current_line_words = []
    current_line_chars = 0

    for word in words:
        wlen = len(word) + 1  # +1 for space
        if current_line_chars + wlen > max_chars_per_line and current_line_words:
            lines.append(" ".join(current_line_words))
            current_line_words = [word]
            current_line_chars = wlen
        else:
            current_line_words.append(word)
            current_line_chars += wlen

    if current_line_words:
        lines.append(" ".join(current_line_words))

    # Group into caption blocks (max 2 lines each)
    caption_blocks = []
    for i in range(0, len(lines), max_lines_per_caption):
        block = lines[i:i + max_lines_per_caption]
        caption_blocks.append("\n".join(block))

    # Assign timings proportionally
    caption_lines = []
    words_per_caption = len(words) / max(len(caption_blocks), 1)
    word_idx = 0

    for i, block in enumerate(caption_blocks):
        block_word_count = len(block.split())
        start_time = word_idx * word_dur
        end_time = min((word_idx + block_word_count) * word_dur, total_duration)
        word_idx += block_word_count

        # Detect emphasis words (ALL CAPS or *wrapped*)
        emphasis = bool(re.search(r'\*[^*]+\*|\b[A-Z]{3,}\b', block))

        caption_lines.append(CaptionLine(
            index=i + 1,
            start=round(start_time, 2),
            end=round(end_time, 2),
            text=block,
            emphasis=emphasis,
        ))

    # Ensure last caption ends exactly at total_duration
    if caption_lines:
        caption_lines[-1].end = round(total_duration, 2)

    logger.info(f"generated {len(caption_lines)} caption lines for {total_duration:.1f}s",
                extra={"stage": "captions", "status": "generated"})
    return CaptionTrack(lines=caption_lines)


# --- SRT Formatter ---

def to_srt(track: CaptionTrack) -> str:
    """Convert to SRT format."""

    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    parts = []
    for line in track.lines:
        parts.append(str(line.index))
        parts.append(f"{fmt(line.start)} --> {fmt(line.end)}")
        parts.append(line.text)
        parts.append("")  # blank line
    return "\n".join(parts)


# --- ASS Formatter (advanced styling) ---

ASS_HEADER = """[Script Info]
Title: YouTube Shorts Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,20,20,180,1
Style: Emphasis,Arial,76,&H00FFFF00,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,20,20,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def to_ass(track: CaptionTrack) -> str:
    """Convert to ASS format with styling for mobile readability."""

    def fmt_ass(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int((t - int(t)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    parts = [ASS_HEADER]
    for line in track.lines:
        style = "Emphasis" if line.emphasis else "Default"
        text = line.text.replace("\n", "\\N")
        parts.append(f"Dialogue: 0,{fmt_ass(line.start)},{fmt_ass(line.end)},{style},,0,0,0,,{text}")
    return "\n".join(parts)


def write_caption_files(
    track: CaptionTrack,
    base_path: str,
    formats: list[str] = ("srt", "ass"),
) -> dict[str, str]:
    """Write caption files and return paths."""
    out = {}
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    if "srt" in formats:
        srt_path = base.with_suffix(".srt")
        srt_path.write_text(to_srt(track), encoding="utf-8")
        out["srt"] = str(srt_path)

    if "ass" in formats:
        ass_path = base.with_suffix(".ass")
        ass_path.write_text(to_ass(track), encoding="utf-8")
        out["ass"] = str(ass_path)

    logger.info(f"wrote captions: {list(out.keys())}",
                extra={"stage": "captions", "status": "written"})
    return out