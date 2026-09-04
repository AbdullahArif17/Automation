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


def _build_dynamic_crop_expr(shots: list[Any]) -> str:
    """Build a nested FFmpeg time expression for dynamic multi-shot crop X coordinate.

    Example: 3 shots ending at t=12.5s (x=180), t=28.0s (x=720), and t=45.0s (x=240):
    Returns: 'if(lt(t,12.50),180,if(lt(t,28.00),720,240))'
    """
    if not shots:
        return "0"
    if len(shots) == 1:
        return str(shots[0].crop_x)

    expr = str(shots[-1].crop_x)
    for shot in reversed(shots[:-1]):
        expr = f"if(lt(t,{shot.end_time:.2f}),{shot.crop_x},{expr})"
    return expr


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
        elif framing_plan.mode == "dynamic" and getattr(framing_plan, "shots", None):
            x_expr = _build_dynamic_crop_expr(framing_plan.shots)
            return (
                f"crop={framing_plan.crop_w}:{framing_plan.crop_h}:'{x_expr}':{framing_plan.crop_y},"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"
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


def detect_hardcoded_subtitles(
    video_path: str,
    start_time: float,
    duration: float,
    max_samples: int = 10,
) -> bool:
    """Detect if a video has real dynamic burned-in subtitles.

    Distinguishes actual changing subtitle text from static objects
    (laptops, desks, logos, podiums) using temporal variance:
    - Subtitles appear, change words, and disappear as speech progresses.
    - Laptops, desks, and logos remain in the exact same position with static pixels.
    """
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280

        # Subtitle region: lower third (55% to 92% of height, 10% to 90% of width)
        y1, y2 = int(h * 0.55), int(h * 0.92)
        x1, x2 = int(w * 0.10), int(w * 0.90)

        masks = []
        sample_interval = max(0.8, duration / 10.0)
        t = start_time + 0.5
        end_t = start_time + duration - 0.5
        samples_taken = 0

        while t < end_t and samples_taken < max_samples:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                t += sample_interval
                continue

            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Subtitles have high brightness (white > 200 or yellow) + high gradient (dark outlines)
            _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            _, edges = cv2.threshold(grad, 35, 255, cv2.THRESH_BINARY)

            text_candidate = cv2.bitwise_and(bright, edges)

            # Filter small noise: must have at least 50 active pixels
            if np.sum(text_candidate > 0) >= 50:
                masks.append((t, text_candidate))
            else:
                masks.append((t, None))

            samples_taken += 1
            t += sample_interval

        cap.release()

        if len(masks) < 2:
            return False

        dynamic_changes = 0
        static_matches = 0

        for i in range(len(masks) - 1):
            t_a, m_a = masks[i]
            t_b, m_b = masks[i + 1]

            if m_a is None and m_b is None:
                continue
            if (m_a is None) != (m_b is None):
                # Text appeared or disappeared! Subtitle behavior.
                dynamic_changes += 1
                continue

            area_a = np.sum(m_a > 0)
            area_b = np.sum(m_b > 0)
            diff = cv2.absdiff(m_a, m_b)
            diff_pixels = np.sum(diff > 0)
            max_area = max(area_a, area_b)

            if max_area > 0:
                diff_ratio = diff_pixels / max_area
                if diff_ratio > 0.45:
                    # Text pixels changed substantially (different words spoken)
                    dynamic_changes += 1
                elif diff_ratio < 0.15:
                    # Static object (laptop/desk/logo)
                    static_matches += 1

        is_subtitles = (dynamic_changes >= 2 and dynamic_changes > static_matches)
        return is_subtitles
    except Exception as exc:
        logger.warning(f"Subtitle pre-detection check failed, defaulting to burning subtitles: {exc}")
        return False


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
        burn_mode = (os.getenv("CLIP_BURN_SUBTITLES") or getattr(settings, "clip_burn_subtitles", "auto")).lower()
        should_burn = True

        if burn_mode == "never":
            should_burn = False
            logger.info(f"Skipping subtitle burn for job {job_id} (CLIP_BURN_SUBTITLES=never)")
        elif burn_mode == "always":
            should_burn = True
        else:  # auto
            has_real_subs = detect_hardcoded_subtitles(source_path, candidate.start_seconds, candidate.duration)
            if has_real_subs:
                should_burn = False
                logger.info(f"Pre-existing dynamic subtitles detected in source for job {job_id}; skipping subtitle burn to avoid double subtitles")

        if should_burn:
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
        "-preset", "fast",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-threads", "0",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        output_path,
    ]

    logger.info(f"cutting segment [{candidate.start_seconds:.1f}-{candidate.end_seconds:.1f}] -> {output_path} (quality: crf=17, fps={target_fps})",
                extra={"job_id": job_id, "stage": "cut", "status": "start"})

    def _run():
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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