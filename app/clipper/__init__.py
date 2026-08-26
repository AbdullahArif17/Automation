"""Clipper package: video-to-shorts pipeline."""
from __future__ import annotations

__version__ = "0.1.0"

from app.clipper.pipeline import ClipperPipeline, PipelineResult
from app.clipper.transcribe import (
    transcribe_video, transcribe_or_load, TranscriptResult, WordTimestamp, SegmentTimestamp
)
from app.clipper.highlight import select_highlights, ClipCandidate
from app.clipper.cut import cut_segment, cut_all_candidates, CutResult
from app.clipper.captions import generate_clip_captions, ClipCaptionResult

__all__ = [
    "ClipperPipeline", "PipelineResult",
    "transcribe_video", "transcribe_or_load", "TranscriptResult", "WordTimestamp", "SegmentTimestamp",
    "select_highlights", "ClipCandidate",
    "cut_segment", "cut_all_candidates", "CutResult",
    "generate_clip_captions", "ClipCaptionResult",
]