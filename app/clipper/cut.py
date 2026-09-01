"""Video cutting and vertical reframing for Shorts.

Uses ffmpeg with accurate seeking (re-encode at cut point) and
configurable crop mode for 9:16 conversion.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.clipper.highlight import ClipCandidate
from app.config.settings import get_settings
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)


@dataclass
class CutResult:
    """Result of cutting a clip segment."""
    output_path: str
    start_seconds: float
    end_seconds: float
    duration: float
    width: int
    height: int


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_video_info(path: str) -> tuple[int, int, float]:
    """Get video width, height, duration via ffprobe."""
    import json
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "json", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    data = json.loads(result.stdout)
    w = data["streams"][0]["width"]
    h = data["streams"][0]["height"]
    dur = float(data["format"]["duration"])
    return w, h, dur


def build_crop_filter(crop_mode: str, src_w: int, src_h: int, target_w: int = 1080, target_h: int = 1920) -> str:
    """Build ffmpeg crop filter for 9:16 conversion.

    Args:
        crop_mode: 'center' (hard crop), 'blur' (preserves full video with blurred background)
        src_w, src_h: Source video dimensions
        target_w, target_h: Output dimensions (default 1080x1920 = 9:16)

    Returns:
        Filter string for -filter_complex
    """
    if crop_mode == "blur":
        # Split video into background and foreground.
        # Background: scale to fill, crop, and heavily blur.
        # Foreground: scale to fit (letterbox) and overlay on center.
        return (
            f"split[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=40[bg_blurred];"
            f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg_scaled];"
            f"[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2"
        )
    elif crop_mode == "center":
        # Determine crop to get 9:16 from source
        src_ar = src_w / src_h
        target_ar = target_w / target_h  # 0.5625

        if src_ar > target_ar:
            # Source is wider than 9:16 (e.g., 16:9 = 1.78) -> crop sides
            crop_h = src_h
            crop_w = int(src_h * target_ar)
            x_offset = (src_w - crop_w) // 2
            y_offset = 0
        else:
            # Source is taller than 9:16 -> crop top/bottom
            crop_w = src_w
            crop_h = int(src_w / target_ar)
            x_offset = 0
            y_offset = (src_h - crop_h) // 2

        return f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"
    else:
        # Default: just scale to fit (letterbox)
        return f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"


def cut_segment(
    source_path: str,
    candidate: ClipCandidate,
    output_path: str,
    crop_mode: Optional[str] = None,
    job_id: Optional[str] = None,
    ass_path: Optional[str] = None,
) -> CutResult:
    """Cut a segment from source video with accurate seeking and 9:16 reframe.

    Uses -ss before -i for fast seek, then -ss after -i for accurate seek,
    and re-encodes to avoid keyframe issues at cut boundaries.

    Args:
        source_path: Path to source video.
        candidate: ClipCandidate with start/end timestamps.
        output_path: Where to write the clipped video.
        crop_mode: 'center' (default from settings) or other future modes.
        job_id: Job ID for logging.
        ass_path: Optional path to .ass subtitle file to burn into video.

    Returns:
        CutResult with output path and metadata.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg/ffprobe not found in PATH")

    settings = get_settings()
    crop_mode = crop_mode or settings.clip_crop_mode
    target_w = settings.video_width
    target_h = settings.video_height

    src_w, src_h, src_dur = get_video_info(source_path)

    # Validate timestamps
    if candidate.start_seconds < 0 or candidate.end_seconds > src_dur:
        raise ValueError(f"candidate timestamps [{candidate.start_seconds}, {candidate.end_seconds}] outside source duration {src_dur}")

    duration = candidate.end_seconds - candidate.start_seconds

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build filter chain
    crop_filter = build_crop_filter(crop_mode, src_w, src_h, target_w, target_h)

    if ass_path:
        safe_ass = str(Path(ass_path).absolute()).replace("\\", "/").replace(":", "\\:")
        crop_filter += f",subtitles='{safe_ass}'"

    # ffmpeg command:
    # -ss before -i: fast seek to nearest keyframe before start
    # -ss after -i: accurate seek from keyframe to exact start (re-encodes)
    # -t: duration
    # -filter_complex: crop/scale/blur to 9:16
    # -c:v libx264 -preset fast -crf 18: re-encode video (High quality)
    # -c:a aac -b:a 128k: re-encode audio
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(candidate.start_seconds),  # fast seek (before -i)
        "-i", source_path,
        "-ss", "0",  # accurate seek from keyframe (after -i, offset 0 since we already seeked)
        "-t", str(duration),
        "-filter_complex", crop_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-r", "30",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        output_path,
    ]

    logger.info(f"cutting segment [{candidate.start_seconds:.1f}-{candidate.end_seconds:.1f}] -> {output_path}",
                extra={"job_id": job_id, "stage": "cut", "status": "start"})

    def _run():
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg cut failed: {result.stderr[-2000:]}")
        return result

    retry(_run, max_attempts=2, retry_on=(subprocess.TimeoutExpired,))

    # Verify output
    out_w, out_h, out_dur = get_video_info(output_path)

    logger.info(f"cut complete: {out_dur:.1f}s {out_w}x{out_h} -> {output_path}",
                extra={"job_id": job_id, "stage": "cut", "status": "done"})

    return CutResult(
        output_path=output_path,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
        duration=out_dur,
        width=out_w,
        height=out_h,
    )


def cut_all_candidates(
    source_path: str,
    candidates: list[ClipCandidate],
    output_dir: str,
    crop_mode: Optional[str] = None,
    job_id: Optional[str] = None,
) -> list[CutResult]:
    """Cut multiple candidates from the same source video.

    Output files named: {source_stem}_clip_{index}.mp4
    """
    results = []
    source_stem = Path(source_path).stem
    for i, cand in enumerate(candidates):
        out_path = str(Path(output_dir) / f"{source_stem}_clip_{i+1}.mp4")
        result = cut_segment(source_path, cand, out_path, crop_mode, job_id)
        results.append(result)
    return results