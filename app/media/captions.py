"""Local caption generation.

Generates timed captions from script text by estimating word timing.
Outputs SRT and ASS formats for burning with FFmpeg.
No paid APIs - fully local, deterministic.

Supports word-level karaoke highlighting when word boundaries are available
(from edge-tts), falling back to phrase-level estimation otherwise.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
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
    # Word-level timing for karaoke (list of (word, start, end) in seconds)
    words: list[tuple[str, float, float]] = field(default_factory=list)


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
    word_boundaries: Optional[list[dict]] = None,  # from edge-tts: [{"text": "word", "offset": 100ns, "duration": 100ns}]
) -> CaptionTrack:
    """Split script into timed caption lines.

    Algorithm:
    1. Split script into words
    2. If word_boundaries provided (from edge-tts), use exact timings
       Otherwise estimate time per word from total_duration / word_count
    3. Group words into lines (max chars)
    4. Group lines into caption blocks (max 2 lines)
    5. Assign start/end times proportionally (or from word boundaries)
    """
    words = script.split()
    if not words:
        return CaptionTrack(lines=[])

    # If we have exact word boundaries from edge-tts, use them
    if word_boundaries:
        return _split_with_word_boundaries(script, total_duration, max_chars_per_line,
                                           max_lines_per_caption, word_boundaries)

    # Fallback: proportional estimation
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

    logger.info(f"generated {len(caption_lines)} caption lines for {total_duration:.1f}s (estimated)",
                extra={"stage": "captions", "status": "generated"})
    return CaptionTrack(lines=caption_lines)


def _split_with_word_boundaries(
    script: str,
    total_duration: float,
    max_chars_per_line: int,
    max_lines_per_caption: int,
    word_boundaries: list[dict],
) -> CaptionTrack:
    """Build captions using exact word timings from edge-tts.

    word_boundaries: list of {"text": "word", "offset": 100ns, "duration": 100ns}
    """
    # Convert 100ns units to seconds
    wb_words = [(wb["text"], wb["offset"] / 1e7, (wb["offset"] + wb["duration"]) / 1e7)
                for wb in word_boundaries]

    # Group words into lines by character count
    lines = []
    current_line_words = []
    current_line_chars = 0

    for word_text, w_start, w_end in wb_words:
        wlen = len(word_text) + 1  # +1 for space
        if current_line_chars + wlen > max_chars_per_line and current_line_words:
            lines.append(current_line_words)
            current_line_words = [(word_text, w_start, w_end)]
            current_line_chars = wlen
        else:
            current_line_words.append((word_text, w_start, w_end))
            current_line_chars += wlen

    if current_line_words:
        lines.append(current_line_words)

    # Group into caption blocks (max 2 lines each)
    caption_blocks = []
    for i in range(0, len(lines), max_lines_per_caption):
        caption_blocks.append(lines[i:i + max_lines_per_caption])

    # Build CaptionLine objects with word-level timing for karaoke
    caption_lines = []
    for i, block_lines in enumerate(caption_blocks):
        # Flatten words from all lines in this block
        block_words = []
        for line_words in block_lines:
            block_words.extend(line_words)

        block_text = " ".join(w[0] for w in block_words)
        start_time = block_words[0][1] if block_words else 0
        end_time = block_words[-1][2] if block_words else total_duration

        # Detect emphasis words
        emphasis = bool(re.search(r'\*[^*]+\*|\b[A-Z]{3,}\b', block_text))

        # Prepare words list for karaoke (relative to block start)
        karaoke_words = []
        for word_text, w_start, w_end in block_words:
            karaoke_words.append((word_text, round(w_start - start_time, 3), round(w_end - start_time, 3)))

        caption_lines.append(CaptionLine(
            index=i + 1,
            start=round(start_time, 2),
            end=round(end_time, 2),
            text=block_text,
            emphasis=emphasis,
            words=karaoke_words,
        ))

    # Ensure last caption ends exactly at total_duration
    if caption_lines:
        caption_lines[-1].end = round(total_duration, 2)

    logger.info(f"generated {len(caption_lines)} caption lines for {total_duration:.1f}s (edge-tts word boundaries)",
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


# --- ASS Formatter (karaoke styling for Shorts) ---

# ASS Style optimized for YouTube Shorts (9:16, mobile viewing)
# Key parameters:
# - Fontsize: 84 (≈4.4% of 1920px; CapCut/Opus use ~8-10% but we keep it readable)
# - Bold: yes (-1)
# - Outline: 4px (thick black outline for contrast)
# - Shadow: 2px (subtle depth)
# - BackColour: semi-transparent black box (&H80000000) for max readability
# - MarginV: 220 (safe zone: avoid bottom 15% ≈ 288px, YouTube UI buttons on right)
# - Alignment: 2 (bottom-center)
ASS_HEADER = """[Script Info]
Title: YouTube Shorts Captions
ScriptType: v4.00+
WrapStyle: 1
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,90,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,10,0,2,40,40,350,1
Style: Emphasis,Montserrat,96,&H0000FF00,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,10,0,2,40,40,350,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _word_to_karaoke(word: str, duration_cs: int) -> str:
    """Convert a single word to ASS karaoke tag.

    duration_cs: duration in centiseconds (1/100s)
    Uses \\k tag (karaoke) which highlights word as it's spoken.
    """
    # Escape ASS special characters
    escaped = word.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
    return f"{{\\k{duration_cs}}}{escaped}"


def _line_to_karaoke_ass(line: CaptionLine) -> str:
    """Convert a CaptionLine to ASS dialogue with karaoke highlighting.

    If line.words has timing data, use \\k tags for per-word highlight.
    Otherwise fall back to simple style-based emphasis.
    """
    def fmt_ass(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int((t - int(t)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    style = "Emphasis" if line.emphasis else "Default"

    if line.words:
        # Build karaoke text with per-word timing
        karaoke_parts = []
        for word_text, w_start, w_end in line.words:
            dur_cs = max(1, int((w_end - w_start) * 100))  # centiseconds, min 1
            karaoke_parts.append(_word_to_karaoke(word_text, dur_cs))
        text = " ".join(karaoke_parts)
        # No inline style override needed - karaoke uses Default style with \k
        return f"Dialogue: 0,{fmt_ass(line.start)},{fmt_ass(line.end)},Default,,0,0,0,,{text}"
    else:
        # Fallback: simple style switching
        text = line.text.replace("\n", "\\N")
        return f"Dialogue: 0,{fmt_ass(line.start)},{fmt_ass(line.end)},{style},,0,0,0,,{text}"


def to_ass(track: CaptionTrack) -> str:
    """Convert to ASS format with karaoke highlighting for mobile readability."""
    parts = [ASS_HEADER]
    for line in track.lines:
        parts.append(_line_to_karaoke_ass(line))
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