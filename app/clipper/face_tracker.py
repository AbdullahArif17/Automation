"""Smart Face Detection and Dynamic Reframing for vertical Shorts.

Analyzes horizontal (16:9) video segments using lightweight computer vision to
automatically detect subjects and construct optimal 9:16 vertical crop layouts:
1. Single Subject: Centers framing on the speaker's face with smooth panning.
2. Two Subjects (Podcasts/Debates): Creates a vertical split-screen stacking
   both speakers (Speaker 1 on top, Speaker 2 on bottom).
3. Fallback: Reverts to clean center-crop if no human faces are present.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import subprocess

from app.utils.logging import get_logger

logger = get_logger(__name__)

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_CACHE_DIR = Path.home() / ".cache" / "yunet"
MODEL_PATH = MODEL_CACHE_DIR / "face_detection_yunet_2023mar.onnx"


YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def _ensure_yunet_model() -> Optional[str]:
    """Download YuNet ONNX model to local cache if not present, with checksum validation."""
    import hashlib

    if MODEL_PATH.exists():
        try:
            content = MODEL_PATH.read_bytes()
            if hashlib.sha256(content).hexdigest() == YUNET_MODEL_SHA256:
                return str(MODEL_PATH)
            logger.warning("Cached YuNet model checksum mismatch; re-downloading...")
            MODEL_PATH.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"Error validating cached YuNet model: {exc}")
            MODEL_PATH.unlink(missing_ok=True)

    try:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading YuNet face detection model to {MODEL_PATH}...")
        tmp_path = MODEL_PATH.with_suffix(".tmp")
        urllib.request.urlretrieve(YUNET_MODEL_URL, str(tmp_path))
        downloaded = tmp_path.read_bytes()
        if hashlib.sha256(downloaded).hexdigest() != YUNET_MODEL_SHA256:
            logger.warning("Downloaded YuNet model checksum mismatch; falling back to Haar")
            tmp_path.unlink(missing_ok=True)
            return None
        tmp_path.replace(MODEL_PATH)
        return str(MODEL_PATH)
    except Exception as exc:
        logger.warning(f"Could not download YuNet model, will fallback to Haar cascade: {exc}")
        return None


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    conf: float = 1.0

    @property
    def center_x(self) -> int:
        return self.x + self.w // 2

    @property
    def center_y(self) -> int:
        return self.y + self.h // 2


@dataclass
class ShotPlan:
    """Represents framing for a discrete camera shot / scene within the clip."""
    start_time: float  # relative to clip start (seconds)
    end_time: float    # relative to clip start (seconds)
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int


@dataclass
class FramingPlan:
    mode: str  # "single", "dynamic", "split", "blur", or "center"
    has_subtitles: bool = False  # True if source video already has hardcoded subtitles
    # For single / dynamic mode:
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    shots: list[ShotPlan] = field(default_factory=list)
    # For split mode (top & bottom crops):
    top_x: int = 0
    top_y: int = 0
    top_w: int = 0
    top_h: int = 0
    bottom_x: int = 0
    bottom_y: int = 0
    bottom_w: int = 0
    bottom_h: int = 0





class FaceDetector:
    """Lightweight dual-backend face detector (YuNet DNN with Haar Cascade fallback)."""

    def __init__(self):
        self.yunet = None
        self.haar = None
        
        model_file = _ensure_yunet_model()
        if model_file and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model_file,
                    "",
                    (320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5,
                )
            except Exception as exc:
                logger.warning(f"Failed to initialize YuNet detector: {exc}")

        # Optional Haar Cascade fallback if supported by OpenCV build
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self.haar = cv2.CascadeClassifier(cascade_path)
            except Exception as exc:
                logger.debug(f"Haar cascade initialization skipped: {exc}")

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        """Detect faces in a BGR frame."""
        h, w = frame.shape[:2]
        faces: list[FaceBox] = []

        # Try YuNet first
        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w, h))
                _, detected_faces = self.yunet.detect(frame)
                if detected_faces is not None:
                    for f in detected_faces:
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        conf = float(f[14])
                        if fw > 20 and fh > 20 and conf >= 0.55:
                            faces.append(FaceBox(x=max(0, fx), y=max(0, fy), w=fw, h=fh, conf=conf))
                    return faces
            except Exception as exc:
                logger.debug(f"YuNet inference error: {exc}")

        # Fallback to Haar cascade
        if self.haar is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.haar.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)
            )
            for (fx, fy, fw, fh) in detected:
                faces.append(FaceBox(x=fx, y=fy, w=fw, h=fh, conf=0.8))

        return faces


def _extract_frame_ffmpeg(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Extract a single frame as a numpy BGR image using ffmpeg fallback when OpenCV decoding fails."""
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{timestamp:.2f}",
            "-i", video_path,
            "-vframes", "1",
            "-f", "image2",
            "-c:v", "mjpeg",
            "pipe:1",
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=5)
        if res.returncode == 0 and res.stdout:
            return cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def analyze_clip_framing(
    video_path: str,
    start_seconds: float,
    duration: float,
    src_w: int,
    src_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
    sample_interval: float = 0.5,
    preferred_crop_mode: Optional[str] = None,
) -> FramingPlan:
    """Analyze video frames across the clip segment to determine optimal 9:16 framing.

    Supports:
    1. Aesthetic Blur Mode: for panels, 3+ people, or wide multi-person scenes to keep all subjects visible.
    2. Dynamic Multi-Shot AI Editing: cuts/pans between speakers on camera angle changes.
    3. Side-by-Side Split Screen: stacks 2 distinct speakers (top & bottom) for wide podcast frames.
    4. Single Speaker Tracking: centers on primary speaker.
    5. Center Crop Fallback: for non-face / B-roll footage.
    """
    if preferred_crop_mode == "blur":
        logger.info("Preferred crop mode is 'blur'; using 4:5 portrait blur framing")
        return _make_blur_plan(src_w, src_h, target_w=target_w, target_h=target_h)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open {video_path} for face analysis, using center crop")
        return _make_center_plan(src_w, src_h, target_w, target_h)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector()

    @dataclass
    class _FrameSample:
        time: float
        faces: list[int]
        face_boxes: list[tuple[int, int, int, int]]
        thumb: np.ndarray

    samples: list[_FrameSample] = []
    cut_timestamps: list[float] = [start_seconds]
    prev_thumb: Optional[np.ndarray] = None
    last_cut = start_seconds

    total_sampled = 0
    current_time = start_seconds
    end_time = start_seconds + duration
    consecutive_fails = 0

    # Sample frames across clip duration (e.g., every 0.5s)
    try:
        while current_time < end_time:
            frame_idx = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback to ffmpeg for codecs (e.g. AV1 in WSL) where OpenCV fails
                frame = _extract_frame_ffmpeg(video_path, current_time)

            if frame is None:
                consecutive_fails += 1
                if consecutive_fails >= 5 and total_sampled == 0:
                    break
                current_time += sample_interval
                continue

            consecutive_fails = 0
            total_sampled += 1

            # Resize for faster face detection (skip if source is already small)
            if src_w > 640:
                scale = 640.0 / src_w
                detect_h = int(src_h * scale)
                small_frame = cv2.resize(frame, (640, detect_h))
            else:
                scale = 1.0
                small_frame = frame

            detected = detector.detect(small_frame)
            frame_centers = []
            frame_boxes = []
            for face in detected:
                orig_cx = int(face.center_x / scale)
                orig_cy = int(face.center_y / scale)
                orig_w = int(face.w / scale)
                orig_h = int(face.h / scale)
                frame_centers.append(orig_cx)
                frame_boxes.append((orig_cx, orig_cy, orig_w, orig_h))

            # Scene change / camera cut detection:
            # Downscale grayscale to (160, 90) for fast difference check
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (160, 90))

            if prev_thumb is not None:
                diff = float(np.mean(cv2.absdiff(thumb, prev_thumb)))
                # A camera switch between different angles/people yields diff > 28
                # Minimum shot length = 1.5s to prevent jitter on quick movement
                if diff > 28.0 and (current_time - last_cut) >= 1.5:
                    cut_timestamps.append(current_time)
                    last_cut = current_time

            prev_thumb = thumb
            samples.append(_FrameSample(time=current_time, faces=sorted(frame_centers), face_boxes=frame_boxes, thumb=thumb))
            current_time += sample_interval
    finally:
        cap.release()

    all_face_centers = [s.faces for s in samples if s.faces]
    if not all_face_centers:
        logger.info("No faces detected in clip; falling back to center crop")
        return _make_center_plan(src_w, src_h, target_w, target_h)

    # Check for corner webcam / reaction video layout (PiP):
    # In reaction videos, a small webcam of the reactor is pinned in a corner while the main video plays.
    # Cropping tightly into the corner webcam hides what they are reacting to!
    corner_webcam_hits = 0
    for s in samples:
        for cx, cy, w, h in s.face_boxes:
            if (cx < src_w * 0.35 or cx > src_w * 0.65) and (cy < src_h * 0.38 or cy > src_h * 0.62) and w < (src_w * 0.30):
                corner_webcam_hits += 1
                break
    if corner_webcam_hits >= max(2, int(len(samples) * 0.18)):
        logger.info(
            f"Detected corner webcam / reaction video layout ({corner_webcam_hits}/{len(samples)} frames); "
            f"automatically using 4:5 portrait blur framing so both reactor and content are visible"
        )
        flattened_faces = [x for s in samples for x in s.faces]
        return _make_blur_plan(src_w, src_h, flattened_faces or None, target_w, target_h)

    # 1. Check for 2 or more people continuously in frame (Permanent panels/couches across the whole clip)
    # If 2+ faces are present in the majority of frames (>45%), engage 4:5 taller blur mode centered on speakers.
    # Otherwise, proceed to Dynamic Scene-Aware Framing which cuts between speakers across shots.
    multi_face_frames = [f for f in all_face_centers if len(f) >= 2]
    if len(multi_face_frames) >= max(3, int(len(all_face_centers) * 0.45)):
        flattened_faces = [x for f in all_face_centers for x in f]
        logger.info(
            f"Detected continuous multi-person scene ({len(multi_face_frames)}/{len(all_face_centers)} frames with 2+ faces); "
            f"engaging 4:5 taller portrait blur framing centered on subjects"
        )
        return _make_blur_plan(src_w, src_h, flattened_faces, target_w, target_h)

    # 2. Dynamic Scene-Aware Framing: analyze camera shots
    cut_timestamps.append(end_time)
    cuts = sorted(list(set(cut_timestamps)))
    shot_plans: list[ShotPlan] = []
    default_plan = _make_single_plan(src_w, src_h, src_w // 2, target_w, target_h)
    crop_w, crop_h, crop_y = default_plan.crop_w, default_plan.crop_h, default_plan.crop_y

    for i in range(len(cuts) - 1):
        t_start = cuts[i]
        t_end = cuts[i + 1]
        shot_samples = [s for s in samples if t_start <= s.time < t_end]
        shot_faces = []
        for s in shot_samples:
            if s.faces:
                p_face = s.faces[0] if len(s.faces) == 1 else s.faces[int(np.argmin(np.abs(np.array(s.faces) - src_w // 2)))]
                shot_faces.append(p_face)

        if shot_faces:
            shot_median_x = int(np.median(shot_faces))
        else:
            shot_median_x = src_w // 2

        shot_single = _make_single_plan(src_w, src_h, shot_median_x, target_w, target_h)
        rel_start = max(0.0, t_start - start_seconds)
        rel_end = max(rel_start + 0.1, t_end - start_seconds)

        shot_plans.append(ShotPlan(
            start_time=rel_start,
            end_time=rel_end,
            crop_x=shot_single.crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
        ))

    # Check if multiple shots actually have distinct framing (diff >= 8% of width)
    distinct_positions = False
    if len(shot_plans) > 1:
        xs = [sp.crop_x for sp in shot_plans]
        if (max(xs) - min(xs)) >= (src_w * 0.08):
            distinct_positions = True

    if distinct_positions:
        shot_info = [(round(sp.start_time, 1), round(sp.end_time, 1), sp.crop_x) for sp in shot_plans]
        logger.info(
            f"AI Editor: detected {len(shot_plans)} camera shots in clip; applying dynamic multi-shot framing: {shot_info}",
            extra={"stage": "face_tracker", "shots": shot_info}
        )
        plan = FramingPlan(
            mode="dynamic",
            crop_x=shot_plans[0].crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            shots=shot_plans,
        )
        return plan

    # 3. Single-speaker tracking mode fallback
    primary_centers = [f[0] if len(f) == 1 else f[int(np.argmin(np.abs(np.array(f) - src_w // 2)))] for f in all_face_centers]
    median_x = int(np.median(primary_centers))
    logger.info(f"Detected single primary speaker at x={median_x}; generating centered smart track")
    return _make_single_plan(src_w, src_h, median_x, target_w, target_h)


def _make_blur_plan(
    src_w: int = 1920,
    src_h: int = 1080,
    face_centers: Optional[list[int]] = None,
    target_w: int = 1080,
    target_h: int = 1920,
) -> FramingPlan:
    """Create 4:5 portrait blur framing plan centered on detected subjects."""
    target_fg_ar = 4.0 / 5.0
    base_crop_h = src_h
    base_crop_w = min(src_w, int(src_h * target_fg_ar))

    if face_centers:
        min_x = min(face_centers)
        max_x = max(face_centers)
        margin = int(src_w * 0.12)
        span = (max_x - min_x) + margin * 2
        if span > base_crop_w:
            crop_w = min(src_w, int(span))
            mid_x = (min_x + max_x) // 2
        else:
            crop_w = base_crop_w
            mid_x = int(np.median(face_centers))
        crop_x = max(0, min(mid_x - crop_w // 2, src_w - crop_w))
    else:
        crop_w = base_crop_w
        crop_x = (src_w - crop_w) // 2

    return FramingPlan(
        mode="blur",
        crop_x=crop_x,
        crop_y=0,
        crop_w=crop_w,
        crop_h=base_crop_h,
    )


def _make_center_plan(src_w: int, src_h: int, target_w: int, target_h: int) -> FramingPlan:
    target_ar = target_w / target_h
    crop_h = src_h
    crop_w = int(src_h * target_ar)
    crop_x = max(0, (src_w - crop_w) // 2)
    return FramingPlan(mode="center", crop_x=crop_x, crop_y=0, crop_w=crop_w, crop_h=crop_h)


def _make_single_plan(src_w: int, src_h: int, face_x: int, target_w: int, target_h: int) -> FramingPlan:
    target_ar = target_w / target_h
    crop_h = src_h
    crop_w = int(src_h * target_ar)

    # Center crop around face_x, bounded within video frame
    crop_x = face_x - (crop_w // 2)
    crop_x = max(0, min(crop_x, src_w - crop_w))

    return FramingPlan(mode="single", crop_x=crop_x, crop_y=0, crop_w=crop_w, crop_h=crop_h)


def _make_split_screen_plan(src_w: int, src_h: int, left_x: int, right_x: int, target_w: int, target_h: int) -> FramingPlan:
    # Each split half target: width=1080, height=960 (ratio 9:8 = 1.125)
    split_ar = target_w / (target_h / 2.0)  # 1.125
    half_crop_h = src_h
    half_crop_w = int(src_h * split_ar)

    if half_crop_w > src_w:
        half_crop_w = src_w
        half_crop_h = int(src_w / split_ar)

    # Top panel: speaker 1 (left)
    top_x = max(0, min(left_x - half_crop_w // 2, src_w - half_crop_w))
    top_y = (src_h - half_crop_h) // 2

    # Bottom panel: speaker 2 (right)
    bottom_x = max(0, min(right_x - half_crop_w // 2, src_w - half_crop_w))
    bottom_y = (src_h - half_crop_h) // 2

    return FramingPlan(
        mode="split",
        top_x=top_x,
        top_y=top_y,
        top_w=half_crop_w,
        top_h=half_crop_h,
        bottom_x=bottom_x,
        bottom_y=bottom_y,
        bottom_w=half_crop_w,
        bottom_h=half_crop_h,
    )
