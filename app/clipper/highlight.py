"""Highlight selection using Gemini to find Shorts-worthy segments.

Feeds transcript to Gemini, asks for 1-3 candidate clips with start/end timestamps.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from app.ai.gemini import GeminiProvider
from app.ai.provider import LLMProvider
from app.clipper.transcribe import TranscriptResult
from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClipCandidate:
    """A candidate clip segment for a Short."""
    start_seconds: float
    end_seconds: float
    reason: str
    suggested_title: str
    suggested_description: str
    confidence: float  # 0-1, model's confidence this will work as a Short

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration": self.duration,
            "reason": self.reason,
            "suggested_title": self.suggested_title,
            "suggested_description": self.suggested_description,
            "confidence": self.confidence,
        }


def build_highlight_prompt(transcript: TranscriptResult, min_dur: float, max_dur: float) -> str:
    """Build the prompt for Gemini to select highlights."""
    # Concatenate all segments with timestamps for context
    full_text = ""
    for seg in transcript.segments:
        full_text += f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}\n"

    return f"""You are an expert YouTube Shorts editor. Given the transcript below with timestamps,
identify 1-3 segments that would make compelling standalone Shorts (vertical videos 20-60 seconds).

SOURCE VIDEO DURATION: {transcript.duration:.1f} seconds
TARGET SHORT DURATION: {min_dur:.0f}-{max_dur:.0f} seconds

TRANSCRIPT:
{full_text}

Return ONLY valid JSON matching this exact schema:
{{
  "candidates": [
    {{
      "start_seconds": <float>,
      "end_seconds": <float>,
      "reason": "<why this segment works as a standalone Short>",
      "suggested_title": "<highly optimized SEO title, curiosity gap hook, max 60 chars>",
      "suggested_description": "<2 sentences heavily packed with high-volume search keywords + 'Subscribe for more!' + 4 highly specific #hashtags + #shorts>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

Rules:
- Each candidate duration MUST be between {min_dur:.0f} and {max_dur:.0f} seconds
- Start/end times must exist in the transcript
- Prefer segments with: clear hook, complete thought, visual potential, self-contained
- Reject segments that need context from earlier/later parts
- For suggested_title: Use click-worthy hooks ("The truth about...", "Why...") and front-load keywords. MUST be under 60 chars.
- For suggested_description: Front-load high-search keywords, end with Subscribe CTA and exactly 5 hashtags (including #shorts).
- confidence: your estimate of how well this will perform as a Short
- Return 1-3 candidates, best first
"""


def parse_highlight_response(response: str, min_dur: float, max_dur: float, video_duration: float) -> list[ClipCandidate]:
    """Parse and validate Gemini's response."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from code fences
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise ValueError(f"Failed to parse JSON from response: {response[:200]}")

    if "candidates" not in data:
        raise ValueError("Response missing 'candidates' key")

    candidates = []
    for c in data["candidates"]:
        # Validate required fields
        for field in ("start_seconds", "end_seconds", "reason", "suggested_title", "suggested_description", "confidence"):
            if field not in c:
                raise ValueError(f"Candidate missing required field: {field}")

        start = float(c["start_seconds"])
        end = float(c["end_seconds"])
        dur = end - start

        # Validate duration bounds
        if not (min_dur <= dur <= max_dur):
            logger.warning(f"candidate duration {dur:.1f}s outside bounds [{min_dur}, {max_dur}], skipping",
                           extra={"stage": "highlight", "status": "validation_fail", "duration": dur})
            continue

        # Validate timestamps within video
        if start < 0 or end > video_duration + 1:  # +1 for floating point
            logger.warning(f"candidate timestamps [{start}, {end}] outside video duration {video_duration}, skipping",
                           extra={"stage": "highlight", "status": "validation_fail"})
            continue

        candidates.append(ClipCandidate(
            start_seconds=start,
            end_seconds=end,
            reason=c["reason"],
            suggested_title=c["suggested_title"][:100],
            suggested_description=c["suggested_description"][:5000],
            confidence=float(c["confidence"]),
        ))

    if not candidates:
        raise ValueError("No valid candidates after validation")

    # Sort by confidence descending
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates


def select_highlights(
    transcript: TranscriptResult,
    provider: Optional[LLMProvider] = None,
    min_dur: Optional[float] = None,
    max_dur: Optional[float] = None,
    max_candidates: int = 3,
    job_id: Optional[str] = None,
) -> list[ClipCandidate]:
    """Select highlight segments from transcript using Gemini.

    Args:
        transcript: TranscriptResult from transcribe step.
        provider: LLMProvider (defaults to GeminiProvider from settings).
        min_dur: Minimum clip duration (from settings if None).
        max_dur: Maximum clip duration (from settings if None).
        max_candidates: Max number of candidates to return.
        job_id: Job ID for logging.

    Returns:
        List of ClipCandidate, sorted by confidence (best first).
    """
    settings = get_settings()
    min_dur = min_dur or settings.min_video_duration
    max_dur = max_dur or settings.max_video_duration

    if provider is None:
        provider = GeminiProvider()

    prompt = build_highlight_prompt(transcript, min_dur, max_dur)

    logger.info(f"requesting highlights from LLM (video duration: {transcript.duration:.1f}s)",
                extra={"job_id": job_id, "stage": "highlight", "status": "request"})

    # Try up to 2 times for malformed responses
    for attempt in range(1, 3):
        try:
            response = provider.generate(prompt, temperature=0.3)
            candidates = parse_highlight_response(response, min_dur, max_dur, transcript.duration)

            # Log the model's reasoning for debugging
            for i, c in enumerate(candidates):
                logger.info(f"candidate {i+1}: [{c.start_seconds:.1f}-{c.end_seconds:.1f}] {c.reason} (conf={c.confidence:.2f})",
                            extra={"job_id": job_id, "stage": "highlight", "status": "candidate"})

            logger.info(f"selected {len(candidates)} valid highlight(s)",
                        extra={"job_id": job_id, "stage": "highlight", "status": "done"})
            return candidates[:max_candidates]

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"highlight selection attempt {attempt} failed: {exc}",
                           extra={"job_id": job_id, "stage": "highlight", "status": "retry", "attempt": attempt})
            if attempt == 2:
                raise RuntimeError(f"Gemini failed to return valid highlights after 2 attempts: {exc}")

    raise RuntimeError("highlight selection failed unexpectedly")