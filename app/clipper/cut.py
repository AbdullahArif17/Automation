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
from typing import Optional, Any

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


def get_video_info(path: str) -> tuple[int, int, float, float]:
    """Get video width, height, duration, and fps via ffprobe."""
    import json
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
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
    fps = 30.0
    fps_str = data["streams"][0].get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/", 1)
        try:
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = 30.0
    return w, h, dur, fps


def build_crop_filter(
    crop_mode: str,
    src_w: int,
    src_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
    framing_plan: Optional[Any] = None,
) -> str:
    """Build ffmpeg crop filter for 9:16 conversion with high-fidelity Lanczos scaling.

    Args:
        crop_mode: 'auto'/'smart' (face tracking), 'center' (hard crop), 'blur' (blurred background)
        src_w, src_h: Source video dimensions
        target_w, target_h: Output dimensions (default 1080x1920 = 9:16)
        framing_plan: Optional FramingPlan from face detection analysis

    Returns:
        Filter string for -filter_complex
    """
    if framing_plan is not None:
        if framing_plan.mode == "split":
            top_h = target_h // 2
            return (
                f"split[vtop_in][vbot_in];"
                f"[vtop_in]crop={framing_plan.top_w}:{framing_plan.top_h}:{framing_plan.top_x}:{framing_plan.top_y},"
                f"scale={target_w}:{top_h}:flags=lanczos[top_panel];"
                f"[vbot_in]crop={framing_plan.bottom_w}:{framing_plan.bottom_h}:{framing_plan.bottom_x}:{framing_plan.bottom_y},"
                f"scale={target_w}:{top_h}:flags=lanczos[bottom_panel];"
                f"[top_panel][bottom_panel]vstack=inputs=2"
            )
        elif framing_plan.mode == "single":
            return (
                f"crop={framing_plan.crop_w}:{framing_plan.crop_h}:{framing_plan.crop_x}:{framing_plan.crop_y},"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"
            )

    if crop_mode == "blur":
        # Split video into background and foreground.
        # Background: scale to fill, crop, and heavily blur.
        # Foreground: scale to fit (letterbox) and overlay on center with lanczos sharpness.
        return (
            f"split[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=40[bg_blurred];"
            f"[fg]scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease[fg_scaled];"
            f"[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2"
        )
    elif crop_mode in ("center", "auto", "smart", "face"):
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

        return f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"
    else:
        # Default: scale with lanczos
        return f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"


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
        crop_mode: 'auto' (default: smart AI face tracking), 'center', 'blur'.
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

    src_w, src_h, src_dur, src_fps = get_video_info(source_path)
    target_fps = 60 if src_fps >= 55.0 else 30

    # Validate timestamps
    if candidate.start_seconds < 0 or candidate.end_seconds > src_dur:
        raise ValueError(f"candidate timestamps [{candidate.start_seconds}, {candidate.end_seconds}] outside source duration {src_dur}")

    duration = candidate.end_seconds - candidate.start_seconds

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Perform smart AI face tracking if mode is auto/smart/face
    framing_plan = None
    if crop_mode in ("auto", "smart", "face"):
        try:
            from app.clipper.face_tracker import analyze_clip_framing
            framing_plan = analyze_clip_framing(
                video_path=source_path,
                start_seconds=candidate.start_seconds,
                duration=duration,
                src_w=src_w,
                src_h=src_h,
                target_w=target_w,
                target_h=target_h,
            )
            logger.info(f"AI framing plan determined for job {job_id}: mode={framing_plan.mode}")
        except Exception as exc:
            logger.warning(f"Face tracking analysis failed, falling back to standard crop: {exc}")
            framing_plan = None

    # Build filter chain
    crop_filter = build_crop_filter(crop_mode, src_w, src_h, target_w, target_h, framing_plan=framing_plan)

    if ass_path:
        safe_ass = str(Path(ass_path).absolute()).replace("\\", "/").replace(":", "\\:")
        crop_filter += f",subtitles='{safe_ass}'"

    # ffmpeg command with studio-grade settings:
    # -ss before -i: fast seek to nearest keyframe before start
    # -ss after -i: accurate seek from keyframe to exact start (re-encodes)
    # -t: duration
    # -filter_complex: crop/scale/blur with Lanczos interpolation
    # -c:v libx264 -preset medium -crf 17: visually lossless broadcast quality
    # -pix_fmt yuv420p: universal mobile player compatibility
    # -r target_fps: preserves up to 60fps for silky smooth motion
    # -c:a aac -b:a 192k -ar 48000: pristine 48kHz stereo audio
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(candidate.start_seconds),  # fast seek (before -i)
        "-i", source_path,
        "-ss", "0",  # accurate seek from keyframe (after -i, offset 0 since we already seeked)
        "-t", str(duration),
        "-filter_complex", crop_filter,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        output_path,
    ]

    logger.info(f"cutting segment [{candidate.start_seconds:.1f}-{candidate.end_seconds:.1f}] -> {output_path} (quality: crf=17, fps={target_fps})",
                extra={"job_id": job_id, "stage": "cut", "status": "start"})

    def _run():
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg cut failed: {result.stderr[-2000:]}")
        return result

    retry(_run, max_attempts=2, retry_on=(subprocess.TimeoutExpired,))

    # Verify output
    out_w, out_h, out_dur, _ = get_video_info(output_path)

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