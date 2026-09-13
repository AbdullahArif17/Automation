"""Video cutting and vertical reframing for Shorts.

Uses ffmpeg with accurate seeking (re-encode at cut point) and
configurable crop mode for 9:16 conversion.
"""
from __future__ import annotations

import os
import re
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
        if framing_plan.mode == "blur":
            return (
                f"split[bg][fg];"
                f"[bg]scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=40[bg_blurred];"
                f"[fg]scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease[fg_scaled];"
                f"[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2"
            )
        elif framing_plan.mode == "split":
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
    max_samples: int = 12,
    expected_text: Optional[str] = None,
) -> bool:
    """Robustly detect if a video segment has pre-existing burned-in subtitles.

    Combines OCR transcript cross-matching (when pytesseract is available)
    with multi-color CV word-cluster morphology. Accurately distinguishes real
    dialogue subtitles (white, yellow, boxed) from static TV watermarks, laptops,
    desks, and temporary lower-third banners.
    """
    try:
        import cv2
        import numpy as np
        import re

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dur = total_frames / fps if total_frames > 0 else (start_time + duration)

        # Determine subtitle zone based on aspect ratio
        is_vertical = (w / h) < 1.1
        if is_vertical:
            # Vertical (9:16) - subtitles often in middle-lower area
            y1, y2 = int(h * 0.40), int(h * 0.88)
            x1, x2 = int(w * 0.08), int(w * 0.92)
        else:
            # Widescreen (16:9 / 4:3) - lower 28% and middle 75%
            y1, y2 = int(h * 0.68), int(h * 0.94)
            x1, x2 = int(w * 0.12), int(w * 0.88)

        scale_factor = h / 720.0

        # Build set of expected spoken words (excluding common short stopwords)
        key_tokens = set()
        if expected_text:
            spoken_tokens = set(re.findall(r'[a-zA-Z]{3,}', expected_text.lower()))
            stopwords = {"the", "and", "that", "this", "with", "for", "you", "was", "are", "have", "had"}
            key_tokens = spoken_tokens - stopwords

        # Try importing pytesseract for high-precision OCR matching
        has_ocr = False
        try:
            import pytesseract
            has_ocr = True
        except (ImportError, Exception):
            has_ocr = False

        # Evenly sample frames across candidate segment
        eff_dur = min(duration, max(1.0, video_dur - start_time))
        num_samples = min(max_samples, max(4, int(eff_dur * 1.5)))
        sample_interval = max(0.4, (eff_dur - 0.8) / float(max(num_samples, 1)))
        t = start_time + 0.4
        end_t = start_time + eff_dur - 0.3

        samples = []
        ocr_matches = []

        while t < end_t and len(samples) < max_samples:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                t += sample_interval
                continue

            sub_frame = frame[y1:y2, x1:x2]
            gray_sub = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)

            # --- Strategy 1: OCR Text Extraction (if available) ---
            if has_ocr:
                try:
                    _, ocr_thresh = cv2.threshold(gray_sub, 180, 255, cv2.THRESH_BINARY)
                    raw_ocr = pytesseract.image_to_string(ocr_thresh, config='--psm 6').lower()
                    if not raw_ocr or len(raw_ocr.strip()) < 3:
                        raw_ocr = pytesseract.image_to_string(gray_sub, config='--psm 6').lower()

                    ocr_tokens = set(re.findall(r'[a-zA-Z]{3,}', raw_ocr))
                    if key_tokens:
                        matched = ocr_tokens.intersection(key_tokens)
                        if matched:
                            ocr_matches.append((t, matched))
                    else:
                        if len(ocr_tokens) >= 2:
                            ocr_matches.append((t, ocr_tokens))
                except Exception:
                    pass

            # --- Strategy 2: Multi-Color CV Word-Cluster Analysis ---
            # 1. White text with sharp stroke/edges
            _, white_bright = cv2.threshold(gray_sub, 190, 255, cv2.THRESH_BINARY)
            grad = cv2.morphologyEx(gray_sub, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            _, edges = cv2.threshold(grad, 25, 255, cv2.THRESH_BINARY)
            white_text = cv2.bitwise_and(white_bright, edges)

            # 2. Yellow text (common in viral clips/podcasts)
            hsv_sub = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv_sub, np.array([18, 65, 130]), np.array([38, 255, 255]))
            yellow_text = cv2.bitwise_and(yellow_mask, edges)

            # Combined candidate text pixels
            combined = cv2.bitwise_or(white_text, yellow_text)

            # 3. Morphological close along horizontal axis to group letters into words
            close_w = max(5, int(8 * scale_factor))
            close_h = max(2, int(3 * scale_factor))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
            connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            # 4. Connected components analysis to filter word-like shapes
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected)

            min_h = max(6, int(8 * scale_factor))
            max_h = max(30, int(70 * scale_factor))
            min_w = max(8, int(10 * scale_factor))
            max_w = max(100, int(380 * scale_factor))
            min_area = max(25, int(45 * (scale_factor ** 2)))

            word_clusters = 0
            text_pixel_count = 0
            for s in stats[1:]:
                cw = s[cv2.CC_STAT_WIDTH]
                ch = s[cv2.CC_STAT_HEIGHT]
                area = s[cv2.CC_STAT_AREA]
                if min_h <= ch <= max_h and min_w <= cw <= max_w and area >= min_area:
                    word_clusters += 1
                    text_pixel_count += area

            samples.append({
                "t": t,
                "words": word_clusters,
                "text_pixels": text_pixel_count,
                "mask": connected,
            })
            t += sample_interval

        cap.release()

        # Decision rule 1: Direct OCR match against spoken words
        if ocr_matches:
            if key_tokens:
                if len(ocr_matches) >= 2 or (len(ocr_matches) >= 1 and len(ocr_matches[0][1]) >= 2):
                    logger.info(f"Subtitles detected via OCR transcript match ({len(ocr_matches)} frames matched)")
                    return True
            else:
                if len(ocr_matches) >= 2:
                    logger.info(f"Subtitles detected via general OCR words ({len(ocr_matches)} frames matched)")
                    return True

        if len(samples) < 2:
            return False

        # Decision rule 2: CV Word-Cluster Analysis (density & dynamics)
        frames_with_words = sum(1 for s in samples if s["words"] >= 1)
        frames_with_multi_words = sum(1 for s in samples if s["words"] >= 2)
        avg_pixels = np.mean([s["text_pixels"] for s in samples])
        total_s = len(samples)

        pct_words = frames_with_words / total_s
        pct_multi = frames_with_multi_words / total_s

        dynamic_changes = 0
        static_matches = 0
        for i in range(len(samples) - 1):
            s1 = samples[i]
            s2 = samples[i + 1]
            act1 = s1["text_pixels"]
            act2 = s2["text_pixels"]
            if act1 >= 30 or act2 >= 30:
                diff = cv2.absdiff(s1["mask"], s2["mask"])
                diff_px = int(np.sum(diff > 0))
                max_px = max(act1, act2)
                if max_px > 0:
                    ratio = diff_px / max_px
                    if ratio > 0.30:
                        dynamic_changes += 1
                    elif ratio < 0.10:
                        static_matches += 1

        min_pixel_threshold = 200 * (scale_factor ** 2)
        is_subtitles = (
            pct_words >= 0.70 and
            pct_multi >= 0.45 and
            avg_pixels >= min_pixel_threshold and
            dynamic_changes >= 1 and
            (static_matches == 0 or dynamic_changes >= static_matches or pct_words >= 0.80)
        )

        if is_subtitles:
            logger.info(f"Subtitles detected via CV word clusters (words={pct_words:.0%}, multi={pct_multi:.0%}, avg_px={avg_pixels:.0f}, changes={dynamic_changes})")
            return True

        return False
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

    # Guarantee broadcast visual quality: reject low-res sources (<720p) that look blurry when cropped to 9:16
    if src_h < 720:
        raise ValueError(
            f"Source video resolution too low ({src_w}x{src_h} < 720p). "
            f"Skipping to ensure only high-definition source footage is clipped."
        )

    # Validate timestamps
    if candidate.start_seconds < 0 or candidate.end_seconds > src_dur:
        raise ValueError(f"candidate timestamps [{candidate.start_seconds}, {candidate.end_seconds}] outside source duration {src_dur}")

    duration = candidate.end_seconds - candidate.start_seconds

    # Check if candidate requested blur mode (e.g. for group panels/multi-person scenes)
    candidate_crop_mode = getattr(candidate, "crop_mode", "").lower()
    if candidate_crop_mode == "blur":
        crop_mode = "blur"

    # Check for reaction context (e.g. reactor webcam, response, watching clip)
    # Cropping tightly into a reactor's facecam cuts off the actual video/lift/fail they are reacting to!
    context_text = f"{Path(source_path).name} {getattr(candidate, 'suggested_title', '')} {getattr(candidate, 'reason', '')} {getattr(candidate, 'hook_headline', '')}".lower()
    if any(k in context_text for k in ("react", "reaction", "reacts", "reacting", "watching", "response to", "breakdown")):
        logger.info(f"Reaction context detected for job {job_id}; automatically engaging blur mode so both reactor and source content are visible")
        crop_mode = "blur"

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
                preferred_crop_mode=crop_mode,
            )
            logger.info(f"AI framing plan determined for job {job_id}: mode={framing_plan.mode}")
        except Exception as exc:
            logger.warning(f"Face tracking analysis failed, falling back to standard crop: {exc}")
            framing_plan = None
    elif crop_mode == "blur":
        from app.clipper.face_tracker import FramingPlan
        framing_plan = FramingPlan(mode="blur")
        logger.info(f"Aesthetic blur mode engaged for job {job_id} (full group frame preserved)")

    # Build filter chain
    crop_filter = build_crop_filter(crop_mode, src_w, src_h, target_w, target_h, framing_plan=framing_plan)

    # Dynamic camera punch-in zoom (0-2.2s) to break mobile thumb-swipe inertia
    intro_punchin = os.getenv("CLIP_INTRO_PUNCHIN", "true").lower() in ("true", "1", "yes")
    if intro_punchin and duration >= 3.0:
        crop_filter += (
            f",crop=w='if(lte(t\\,2.2)\\,{target_w}/(1.07-0.07*(t/2.2))\\,{target_w})':"
            f"h='if(lte(t\\,2.2)\\,{target_h}/(1.07-0.07*(t/2.2))\\,{target_h})':"
            f"x='(in_w-out_w)/2':y='(in_h-out_h)/2',scale={target_w}:{target_h}:flags=lanczos"
        )

    # Subtitle and Top Hook Banner burning
    target_ass = ass_path
    if not target_ass:
        hook_text = getattr(candidate, "hook_headline", "").strip() or getattr(candidate, "suggested_title", "").strip()
        if hook_text:
            try:
                from app.media.captions import create_hook_only_ass
                fallback_ass_path = str(Path(output_path).with_suffix(".hook.ass"))
                target_ass = create_hook_only_ass(hook_text, duration, fallback_ass_path)
            except Exception:
                target_ass = None

    if target_ass:
        burn_mode = (os.getenv("CLIP_BURN_SUBTITLES") or getattr(settings, "clip_burn_subtitles", "auto")).lower()
        should_burn = True
        has_real_subs = False

        if burn_mode == "never":
            should_burn = False
            logger.info(f"Skipping subtitle burn for job {job_id} (CLIP_BURN_SUBTITLES=never)")
        elif burn_mode == "always":
            should_burn = True
        else:  # auto
            # Extract expected spoken words from ASS or candidate to guide OCR cross-matching
            expected_words: list[str] = []
            if ass_path and os.path.exists(ass_path):
                try:
                    with open(ass_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.startswith("Dialogue:") and "TopHook" not in line:
                                parts = line.split(",", 9)
                                if len(parts) >= 10:
                                    clean_text = re.sub(r'\{[^}]*\}', '', parts[9])
                                    expected_words.extend(re.findall(r'[a-zA-Z]{3,}', clean_text))
                except Exception:
                    pass
            if not expected_words:
                cand_text = f"{getattr(candidate, 'suggested_title', '')} {getattr(candidate, 'hook_headline', '')}"
                expected_words = re.findall(r'[a-zA-Z]{3,}', cand_text)

            expected_text = " ".join(expected_words) if expected_words else None
            has_real_subs = detect_hardcoded_subtitles(
                source_path,
                candidate.start_seconds,
                candidate.duration,
                expected_text=expected_text,
            )
            if has_real_subs:
                should_burn = False
                logger.info(f"Pre-existing dynamic subtitles detected in source for job {job_id}; skipping dialogue subtitle burn")

        active_ass = None
        if should_burn:
            active_ass = target_ass
        elif has_real_subs:
            # Source video already has dialogue subtitles at the bottom, but we STILL burn
            # the viral Top Hook banner at the top for high-CTR thumbnail & first-frame retention!
            hook_text = getattr(candidate, "hook_headline", "").strip()
            if not hook_text and getattr(candidate, "suggested_title", ""):
                words = [w for w in candidate.suggested_title.split() if not w.startswith("#")]
                hook_text = " ".join(words[:4]).upper() + " 😳"
            if not hook_text:
                hook_text = "WAIT FOR THE END... 🤯"

            try:
                from app.media.captions import create_hook_only_ass
                hook_ass_path = str(Path(target_ass).with_name(f"{Path(target_ass).stem}_hook_only.ass"))
                active_ass = create_hook_only_ass(hook_text, duration, hook_ass_path)
                logger.info(f"Burning top hook headline banner only for job {job_id}: '{hook_text}'")
            except Exception as exc:
                logger.warning(f"Failed to generate hook-only ASS for job {job_id}: {exc}")
                active_ass = None

        if active_ass and os.path.exists(active_ass):
            safe_ass = str(Path(active_ass).absolute()).replace("\\", "/").replace(":", "\\:")
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
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-af", "loudnorm=I=-14:LRA=7:TP=-1.5",
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