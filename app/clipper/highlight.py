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
    crop_mode: str = "center"  # 'center' or 'blur'

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
            "crop_mode": self.crop_mode,
        }


def build_highlight_prompt(transcript: TranscriptResult, min_dur: float, max_dur: float) -> str:
    """Build the prompt for Gemini to select highlights."""
    # Concatenate all segments with timestamps for context
    full_text = ""
    for seg in transcript.segments:
        full_text += f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}\n"

    return f"""You are an elite YouTube Shorts curator and viral video editor.
Given the timestamped transcript below from a long-form video, identify 1-3 segments that will make powerful, self-contained standalone Shorts (25-45 seconds is the sweet spot).

The most important rule: ANY VIEWER who has never seen this podcast or video before MUST be hooked within the first 3 seconds. The clip must feel like a complete, satisfying mini-story or argument, NOT a random chopped fragment.

SOURCE VIDEO DURATION: {transcript.duration:.1f} seconds
TARGET SHORT DURATION: {min_dur:.0f}-{max_dur:.0f} seconds (optimal: 25-45s)
TARGET AUDIENCE: United States, Canada, and United Kingdom. Prioritize moments that grip Western audiences: recognizable celebrities, entrepreneurs, intense debates, shocking admissions, or universally relatable humor.

TRANSCRIPT:
{full_text}

Return ONLY valid JSON matching this exact schema:
{{
  "candidates": [
    {{
      "start_seconds": <float>,
      "end_seconds": <float>,
      "reason": "<explain the context, who is speaking, what the core idea/punchline is, and why it works as a standalone Short>",
      "suggested_title": "<punchy curiosity hook naming person/topic, max 50 chars for mobile>",
      "suggested_description": "<2 context-rich sentences explaining who is talking and what happened + high-volume search keywords + 'Subscribe for more!' + 4 specific #hashtags + #shorts>",
      "confidence": <0.0-1.0>,
      "crop_mode": "<'center' or 'blur'>"
    }}
  ]
}}

STRICT QUALITY RULES:
1. THE 3-SECOND HOOK (CRITICAL):
   - The first sentence spoken MUST be an instant hook: a surprising claim, dramatic question, paradox, or intense emotion.
   - STRICTLY BANNED: Never start with conversational runway like 'Well...', 'You know...', 'So basically...', 'In my opinion...', 'And so...', or polite small talk. The viewer must be stopped from swiping within the first 2 seconds.
2. STANDALONE COMPLETION (CRITICAL):
   - The clip MUST finish at the natural end of a sentence delivering the payoff, punchline, debate conclusion, or reaction.
   - NEVER cut off mid-sentence or right before the climax.
3. MOBILE-OPTIMIZED TITLE:
   - Must explicitly name the person, topic, or conflict (e.g., 'Joe Rogan on the 1994 Own Goal' or 'Ronaldo Explains Why He Left').
   - Keep under 50 chars so the title is never cut off by '...' on mobile screens.
4. STRICT DURATION BOUNDS (CRITICAL):
   - Duration MUST be between {min_dur:.0f} and {max_dur:.0f} seconds (optimal sweet spot is 28-45s for 80%+ completion rate).
   - Snippets under {min_dur:.0f}s or over {max_dur:.0f}s will be rejected.
5. CROP MODE:
   - Use 'center' for interviews, podcasts, gym, and centered subjects.
   - Use 'blur' for gaming or wide group panels where edges matter.
6. Return 1-3 candidates, best first.
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
        # Validate required fields (supporting common LLM aliases)
        if "start_seconds" not in c and "start" not in c:
            raise ValueError("Candidate missing required field: start_seconds")
        if "end_seconds" not in c and "end" not in c:
            raise ValueError("Candidate missing required field: end_seconds")
        if "reason" not in c and "explanation" not in c and "why" not in c:
            raise ValueError("Candidate missing required field: reason")
        if "suggested_title" not in c and "title" not in c and "headline" not in c:
            raise ValueError("Candidate missing required field: suggested_title")
        if "suggested_description" not in c and "description" not in c and "summary" not in c:
            raise ValueError("Candidate missing required field: suggested_description")
        if "confidence" not in c and "score" not in c:
            raise ValueError("Candidate missing required field: confidence")

        start = float(c.get("start_seconds") if "start_seconds" in c else c["start"])
        end = float(c.get("end_seconds") if "end_seconds" in c else c["end"])
        dur = end - start

        # If slightly over max_dur (e.g. 60.5s or 63s), clamp to max_dur gracefully
        if dur > max_dur and dur <= max_dur + 5.0:
            end = start + max_dur
            dur = max_dur

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

        # Extract title and description with flexible key fallbacks
        title = (c.get("suggested_title") or c.get("title") or c.get("headline") or "").strip()
        if not title:
            title = "Viral Highlight Clip"
        if "#shorts" not in title.lower() and len(title) <= 45:
            title = f"{title} #shorts"
        title = title[:60].strip()

        desc = (c.get("suggested_description") or c.get("description") or c.get("summary") or "").strip()
        if not desc:
            desc = f"{title}\n\nSubscribe for more daily clips! #shorts"

        reason = c.get("reason") or c.get("explanation") or c.get("why") or "Standalone highlight"

        try:
            conf = float(c.get("confidence", 0.85))
        except (ValueError, TypeError):
            conf = 0.85

        c_mode = c.get("crop_mode", "center")
        if c_mode not in ("center", "blur"):
            c_mode = "center"

        candidates.append(ClipCandidate(
            start_seconds=start,
            end_seconds=end,
            reason=reason,
            suggested_title=title,
            suggested_description=desc[:5000],
            confidence=conf,
            crop_mode=c_mode,
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