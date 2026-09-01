"""Caption generation for clips using real whisper timestamps.

Reuses the SRT/ASS formatters from app.media.captions but builds
CaptionTrack from actual word-level timestamps aligned to the clip segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.clipper.highlight import ClipCandidate
from app.clipper.transcribe import TranscriptResult, SegmentTimestamp, WordTimestamp
from app.media.captions import (
    CaptionLine, CaptionTrack, to_srt, to_ass, write_caption_files,
    split_into_caption_lines
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClipCaptionResult:
    """Caption files for a clip."""
    clip_candidate: ClipCandidate
    srt_path: str
    ass_path: str
    track: CaptionTrack


def build_caption_track_from_whisper(
    transcript: TranscriptResult,
    clip: ClipCandidate,
    max_chars_per_line: int = 20,
    max_lines_per_caption: int = 2,
) -> CaptionTrack:
    """Build CaptionTrack from whisper word timestamps, aligned to clip.

    Args:
        transcript: Full transcript from source video.
        clip: ClipCandidate with start/end seconds relative to source.
        max_chars_per_line: Max chars per caption line.
        max_lines_per_caption: Max lines per caption block.

    Returns:
        CaptionTrack with timings relative to clip start (0 = clip start).
    """
    clip_start = clip.start_seconds
    clip_end = clip.end_seconds
    clip_duration = clip.duration

    # Collect all words within the clip window, with times shifted to clip-relative
    clip_words: list[tuple[str, float, float]] = []  # (word, start_rel, end_rel)

    for seg in transcript.segments:
        if seg.end <= clip_start:
            continue
        if seg.start >= clip_end:
            break
        for w in seg.words:
            # Shift to clip-relative time
            w_start = w.start - clip_start
            w_end = w.end - clip_start
            # Only include words that fall within clip bounds (with small tolerance)
            if w_end <= 0 or w_start >= clip_duration:
                continue
            # Clamp to clip bounds
            w_start = max(0.0, w_start)
            w_end = min(clip_duration, w_end)
            clip_words.append((w.word, w_start, w_end))

    if not clip_words:
        logger.warning(f"no whisper words found in clip window [{clip_start:.1f}-{clip_end:.1f}], falling back to estimated",
                       extra={"stage": "captions", "status": "fallback"})
        # Fallback: use estimated timing from full clip text
        full_text = " ".join(w for w, _, _ in clip_words) if clip_words else clip.suggested_title
        return split_into_caption_lines(full_text, clip_duration, max_chars_per_line, max_lines_per_caption)

    # Group words into lines by max_chars_per_line
    lines: list[list[tuple[str, float, float]]] = []
    current_line: list[tuple[str, float, float]] = []
    current_chars = 0

    for word, w_start, w_end in clip_words:
        wlen = len(word) + 1  # +1 for space
        if current_chars + wlen > max_chars_per_line and current_line:
            lines.append(current_line)
            current_line = [(word, w_start, w_end)]
            current_chars = wlen
        else:
            current_line.append((word, w_start, w_end))
            current_chars += wlen

    if current_line:
        lines.append(current_line)

    # Group lines into caption blocks (max_lines_per_caption)
    caption_blocks = []
    for i in range(0, len(lines), max_lines_per_caption):
        block = lines[i:i + max_lines_per_caption]
        caption_blocks.append(block)

    # Build CaptionLine objects with actual whisper timings
    caption_lines = []
    for i, block in enumerate(caption_blocks):
        # Flatten words in block
        block_words = [w for line in block for w in line]
        block_text = " ".join(w for w, _, _ in block_words)

        # Use first word start and last word end for timing
        start_time = block_words[0][1]
        end_time = block_words[-1][2]

        # Detect emphasis (ALL CAPS or *wrapped*)
        import re
        emphasis = bool(re.search(r'\*[^*]+\*|\b[A-Z]{3,}\b', block_text))

        caption_lines.append(CaptionLine(
            index=i + 1,
            start=round(start_time, 2),
            end=round(end_time, 2),
            text=block_text,
            emphasis=emphasis,
        ))

    # Ensure last caption ends before CTA
    if caption_lines:
        caption_lines[-1].end = min(round(clip_duration, 2), round(clip_duration - 2.0, 2))

    # Append CTA in the last 2 seconds
    cta_start = max(0.0, round(clip_duration - 2.0, 2))
    cta_end = round(clip_duration, 2)
    caption_lines.append(CaptionLine(
        index=len(caption_lines) + 1,
        start=cta_start,
        end=cta_end,
        text="SUBSCRIBE FOR MORE!",
        emphasis=True,
    ))

    logger.info(f"built {len(caption_lines)} caption lines from whisper timestamps for clip [{clip_start:.1f}-{clip_end:.1f}]",
                extra={"stage": "captions", "status": "built", "word_count": len(clip_words)})

    return CaptionTrack(lines=caption_lines)


def generate_clip_captions(
    transcript: TranscriptResult,
    clip: ClipCandidate,
    output_base: str,
    max_chars_per_line: int = 42,
    max_lines_per_caption: int = 2,
    formats: list[str] = ("srt", "ass"),
) -> ClipCaptionResult:
    """Generate caption files for a clip from whisper transcript.

    Args:
        transcript: Full TranscriptResult from source video.
        clip: ClipCandidate defining the segment.
        output_base: Base path (without extension) for output files.
        max_chars_per_line: Max chars per caption line.
        max_lines_per_caption: Max lines per caption block.
        formats: Which formats to write ("srt", "ass").

    Returns:
        ClipCaptionResult with paths and track.
    """
    track = build_caption_track_from_whisper(
        transcript, clip, max_chars_per_line, max_lines_per_caption
    )

    paths = write_caption_files(track, output_base, formats=formats)

    return ClipCaptionResult(
        clip_candidate=clip,
        srt_path=paths.get("srt", ""),
        ass_path=paths.get("ass", ""),
        track=track,
    )