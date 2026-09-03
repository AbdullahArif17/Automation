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


def _ensure_yunet_model() -> Optional[str]:
    """Download YuNet ONNX model to local cache if not present."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    try:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading YuNet face detection model to {MODEL_PATH}...")
        urllib.request.urlretrieve(YUNET_MODEL_URL, str(MODEL_PATH))
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
    mode: str  # "single", "dynamic", "split", or "center"
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


def _check_frame_has_subtitles(frame: np.ndarray) -> bool:
    """Detect if high-contrast horizontal subtitle text exists in the lower third."""
    try:
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.55):int(h * 0.95), int(w * 0.08):int(w * 0.92)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            _, _, cw, ch = cv2.boundingRect(cnt)
            if cw >= (w * 0.12) and 12 <= ch <= 90 and (cw / max(1, ch)) >= 2.5:
                return True
        return False
    except Exception:
        return False


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
) -> FramingPlan:
    """Analyze video frames across the clip segment to determine optimal 9:16 framing.

    Supports:
    1. Dynamic Multi-Shot AI Editing: cuts/pans between speakers on camera angle changes.
    2. Side-by-Side Split Screen: stacks 2 distinct speakers (top & bottom) for wide podcast frames.
    3. Single Speaker Tracking: centers on primary speaker.
    4. Center Crop Fallback: for non-face / B-roll footage.
    """
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
        thumb: np.ndarray

    samples: list[_FrameSample] = []
    cut_timestamps: list[float] = [start_seconds]
    prev_thumb: Optional[np.ndarray] = None
    last_cut = start_seconds

    total_sampled = 0
    subtitle_hits = 0
    current_time = start_seconds
    end_time = start_seconds + duration
    consecutive_fails = 0

    # Sample frames across clip duration (e.g., every 0.5s)
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

        # Resize for faster face detection (width=640)
        scale = 640.0 / src_w
        detect_h = int(src_h * scale)
        small_frame = cv2.resize(frame, (640, detect_h))

        # Check if this frame contains existing burned-in subtitles
        if _check_frame_has_subtitles(small_frame):
            subtitle_hits += 1

        detected = detector.detect(small_frame)
        frame_centers = []
        for face in detected:
            orig_cx = int(face.center_x / scale)
            frame_centers.append(orig_cx)

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
        samples.append(_FrameSample(time=current_time, faces=sorted(frame_centers), thumb=thumb))
        current_time += sample_interval

    cap.release()

    has_existing_subs = (subtitle_hits >= max(2, int(total_sampled * 0.25))) if total_sampled > 0 else False
    if has_existing_subs:
        logger.info(f"Pre-existing hardcoded subtitles detected in source ({subtitle_hits}/{total_sampled} frames)")

    all_face_centers = [s.faces for s in samples if s.faces]
    if not all_face_centers:
        logger.info("No faces detected in clip; falling back to center crop")
        plan = _make_center_plan(src_w, src_h, target_w, target_h)
        plan.has_subtitles = has_existing_subs
        return plan

    # 1. Check for consistent 2-person podcast wide format across the clip
    two_face_frames = [f for f in all_face_centers if len(f) >= 2 and (f[-1] - f[0]) > (src_w * 0.25)]
    if len(two_face_frames) >= max(3, int(len(all_face_centers) * 0.35)):
        left_speakers = [f[0] for f in two_face_frames]
        right_speakers = [f[-1] for f in two_face_frames]
        median_left = int(np.median(left_speakers))
        median_right = int(np.median(right_speakers))
        logger.info(f"Detected 2 distinct speakers (left={median_left}, right={median_right}); generating podcast split-screen")
        plan = _make_split_screen_plan(src_w, src_h, median_left, median_right, target_w, target_h)
        plan.has_subtitles = has_existing_subs
        return plan

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
            has_subtitles=has_existing_subs,
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
    plan = _make_single_plan(src_w, src_h, median_x, target_w, target_h)
    plan.has_subtitles = has_existing_subs
    return plan


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
