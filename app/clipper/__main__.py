"""CLI entry point for the YouTube Shorts Clipper.

Usage:
    python -m app.clipper <source_video_path> [options]

Converts a long-form source video into one or more YouTube Shorts
by transcribing, selecting highlights, cutting, captioning, and optionally uploading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.clipper.pipeline import ClipperPipeline
from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-shorts-clipper",
        description="Convert long-form videos into YouTube Shorts (local, zero-cost).",
    )
    parser.add_argument("source", type=str, help="Path to source video file")
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload generated Shorts to YouTube (requires AUTO_UPLOAD=true in .env)"
    )
    parser.add_argument(
        "--max-clips", type=int, default=1,
        help="Maximum number of Shorts to produce from the source (1-3, default 1)"
    )
    parser.add_argument(
        "--force-retranscribe", action="store_true",
        help="Ignore cached transcript and re-transcribe"
    )
    parser.add_argument(
        "--model-size", type=str, default=None,
        help=f"Whisper model size (default from WHISPER_MODEL_SIZE env: base.en). Options: tiny, base, small, medium, large, or base.en"
    )
    parser.add_argument(
        "--crop-mode", type=str, default=None,
        help="Crop mode for 9:16 conversion (default from CLIP_CROP_MODE env: center)"
    )
    parser.add_argument(
        "--mock-llm", action="store_true",
        help="Use mock LLM provider (no API key, no cost) for testing"
    )
    parser.add_argument(
        "--list-candidates", action="store_true",
        help="Only transcribe and show highlight candidates, don't cut/upload"
    )
    return parser


def _make_mock_provider(duration: float) -> "MockProvider":
    """Create a MockProvider with a valid highlight response for the given video duration."""
    from app.ai.provider import MockProvider
    start = min(5.0, duration * 0.1)
    end = min(start + 30.0, duration * 0.8, duration - 1.0)
    if end - start < 20.0:  # ensure minimum duration
        end = min(start + 20.0, duration)
    mock_response = json.dumps({
        "candidates": [{
            "start_seconds": start,
            "end_seconds": end,
            "reason": "Mock highlight for testing",
            "suggested_title": "Mock Short Title",
            "suggested_description": "Mock description for testing",
            "confidence": 0.9
        }]
    })
    return MockProvider([mock_response])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()

    # Override settings from CLI args
    if args.model_size:
        settings.whisper_model_size = args.model_size
    if args.crop_mode:
        settings.clip_crop_mode = args.crop_mode

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[error] Source video not found: {source_path}")
        return 2

    if args.max_clips < 1 or args.max_clips > 3:
        print(f"[error] --max-clips must be between 1 and 3")
        return 2

    print(f"[info] Processing: {source_path}")
    print(f"[info] Max clips: {args.max_clips}")
    print(f"[info] Upload: {args.upload}")
    print(f"[info] Whisper model: {settings.whisper_model_size}")
    print(f"[info] Crop mode: {settings.clip_crop_mode}")

    try:
        if args.list_candidates:
            # Just show candidates
            from app.clipper.transcribe import transcribe_or_load
            from app.clipper.highlight import select_highlights

            transcript = transcribe_or_load(
                str(source_path),
                model_size=settings.whisper_model_size,
                force_refresh=args.force_retranscribe,
            )

            if args.mock_llm:
                provider = _make_mock_provider(transcript.duration)
            else:
                provider = None

            candidates = select_highlights(
                transcript,
                provider=provider,
                min_dur=settings.min_video_duration,
                max_dur=settings.max_video_duration,
                max_candidates=args.max_clips,
            )
            print(f"\n=== Highlight Candidates ({len(candidates)}) ===")
            for i, c in enumerate(candidates):
                print(f"  {i+1}. [{c.start_seconds:.1f}-{c.end_seconds:.1f}] ({c.duration:.1f}s)")
                print(f"     Reason: {c.reason}")
                print(f"     Title: {c.suggested_title}")
                print(f"     Confidence: {c.confidence:.2f}")
                print()
            return 0

        # Full pipeline
        pipeline = ClipperPipeline()

        if args.mock_llm:
            # Pre-transcribe to get duration for mock
            from app.clipper.transcribe import transcribe_or_load
            transcript = transcribe_or_load(
                str(source_path),
                model_size=settings.whisper_model_size,
                force_refresh=args.force_retranscribe,
            )
            pipeline.provider = _make_mock_provider(transcript.duration)

        result = pipeline.run(
            str(source_path),
            upload=args.upload,
            max_clips=args.max_clips,
            force_retranscribe=args.force_retranscribe,
        )

        # Print summary
        print(f"\n=== Clipper Result ===")
        print(f"Source: {result.source_path}")
        print(f"Candidates found: {len(result.candidates)}")
        print(f"Clips processed: {len(result.outcomes)}")

        success_count = 0
        for i, outcome in enumerate(result.outcomes):
            if outcome.error:
                print(f"  Clip {i+1}: FAILED - {outcome.error}")
            else:
                print(f"  Clip {i+1}: OK")
                print(f"    Segment: [{outcome.candidate.start_seconds:.1f}-{outcome.candidate.end_seconds:.1f}] ({outcome.candidate.duration:.1f}s)")
                print(f"    Title: {outcome.candidate.suggested_title}")
                if outcome.cut_result:
                    print(f"    Output: {outcome.cut_result.output_path} ({outcome.cut_result.width}x{outcome.cut_result.height})")
                if outcome.published:
                    print(f"    YouTube: {outcome.youtube_video_id} (published)")
                success_count += 1

        if success_count == 0:
            print(f"\n[error] All clips failed")
            return 1

        if args.upload and success_count > 0:
            # Check if any expected upload didn't happen
            unpublished = [o for o in result.outcomes if not o.error and not o.published]
            if unpublished:
                print(f"\n[warning] {len(unpublished)} clip(s) completed but not uploaded")
                return 1

        print(f"\n[success] {success_count}/{len(result.outcomes)} clip(s) completed")
        return 0

    except Exception as exc:
        logger.error(f"clipper failed: {exc}", extra={"stage": "cli", "status": "error"})
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())