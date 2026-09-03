"""Transcription using faster-whisper (local, zero-cost).

Outputs word-level timestamps for caption alignment.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WordTimestamp:
    """Single word with timing."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class SegmentTimestamp:
    """A segment (sentence/phrase) with timing."""
    text: str
    start: float
    end: float
    words: list[WordTimestamp]


@dataclass
class TranscriptResult:
    """Full transcription result."""
    language: str
    language_probability: float
    duration: float
    segments: list[SegmentTimestamp]
    source_path: str

    @property
    def word_count(self) -> int:
        """Total number of words across all segments."""
        return sum(len(s.words) for s in self.segments)

    def to_json(self) -> str:
        return json.dumps({
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "words": [asdict(w) for w in s.words],
                }
                for s in self.segments
            ],
            "source_path": self.source_path,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "TranscriptResult":
        d = json.loads(data)
        return cls(
            language=d["language"],
            language_probability=d["language_probability"],
            duration=d["duration"],
            segments=[
                SegmentTimestamp(
                    text=s["text"],
                    start=s["start"],
                    end=s["end"],
                    words=[WordTimestamp(**w) for w in s["words"]],
                )
                for s in d["segments"]
            ],
            source_path=d["source_path"],
        )


def get_whisper_model(model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
    """Load faster-whisper model (cached)."""
    from faster_whisper import WhisperModel
    logger.info(f"loading faster-whisper model '{model_size}' on {device}",
                extra={"stage": "transcribe", "status": "loading"})
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_video(
    video_path: str,
    model_size: str = "base.en",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    word_timestamps: bool = True,
) -> TranscriptResult:
    """Transcribe a video file with word-level timestamps.

    Args:
        video_path: Path to source video file.
        model_size: Whisper model size (tiny, base, small, medium, large).
                    Use 'base.en' for English-only (faster).
        device: 'cpu' or 'cuda'.
        compute_type: 'int8' (fastest, less accurate), 'int8_float16', 'float16', 'float32'.
        beam_size: Beam size for decoding.
        word_timestamps: Whether to return word-level timestamps.

    Returns:
        TranscriptResult with segments and word timestamps.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    model = get_whisper_model(model_size, device, compute_type)

    logger.info(f"transcribing {video_path}", extra={"stage": "transcribe", "status": "start"})

    segments_iter, info = model.transcribe(
        video_path,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=True,  # Filter out non-speech
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments: list[SegmentTimestamp] = []
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                ))
        segments.append(SegmentTimestamp(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            words=words,
        ))

    result = TranscriptResult(
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=segments,
        source_path=video_path,
    )

    logger.info(f"transcribed {len(segments)} segments, {sum(len(s.words) for s in segments)} words in {info.language} ({info.language_probability:.2f})",
                extra={"stage": "transcribe", "status": "done", "duration": info.duration})
    return result


def save_transcript(result: TranscriptResult, output_path: str) -> None:
    """Save transcript to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(result.to_json(), encoding="utf-8")
    logger.info(f"saved transcript to {output_path}", extra={"stage": "transcribe", "status": "saved"})


def load_transcript(input_path: str) -> TranscriptResult:
    """Load transcript from JSON file."""
    data = Path(input_path).read_text(encoding="utf-8")
    return TranscriptResult.from_json(data)


def get_transcript_cache_path(video_path: str, model_size: str) -> str:
    """Get cache path for transcript (next to source video)."""
    video = Path(video_path)
    return str(video.parent / f"{video.stem}_transcript_{model_size.replace('/', '_')}.json")


def transcribe_or_load(
    video_path: str,
    model_size: str = "base.en",
    device: str = "cpu",
    compute_type: str = "int8",
    force_refresh: bool = False,
) -> TranscriptResult:
    """Transcribe video, using cached transcript if available and not stale."""
    cache_path = get_transcript_cache_path(video_path, model_size)

    if not force_refresh and Path(cache_path).exists():
        # Check if cache is newer than source
        cache_mtime = Path(cache_path).stat().st_mtime
        video_mtime = Path(video_path).stat().st_mtime
        if cache_mtime >= video_mtime:
            logger.info(f"loading cached transcript from {cache_path}",
                        extra={"stage": "transcribe", "status": "cache_hit"})
            return load_transcript(cache_path)
        logger.info(f"cache stale, re-transcribing", extra={"stage": "transcribe", "status": "cache_stale"})

    result = transcribe_video(video_path, model_size, device, compute_type)
    save_transcript(result, cache_path)
    return result