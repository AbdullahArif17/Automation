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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_CACHE_DIR = Path.home() / ".cache" / "yunet"
MODEL_PATH = MODEL_CACHE_DIR / "face_detection_yunet_2023mar.onnx"


def _ensure_yunet_model() -> Optional[str]:
    """Download YuNet ONNX model (~336KB) if not already present."""
    try:
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
            return str(MODEL_PATH)
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
class FramingPlan:
    mode: str  # "single", "split", or "center"
    # For single mode:
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
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

        # Always initialize Haar Cascade as guaranteed fallback
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(cascade_path)
        except Exception as exc:
            logger.warning(f"Failed to initialize Haar cascade: {exc}")

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
    """Analyze video frames across the clip segment to determine optimal 9:16 framing."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open {video_path} for face analysis, using center crop")
        return _make_center_plan(src_w, src_h, target_w, target_h)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector()

    all_face_centers: list[list[int]] = []  # per-frame list of X centers
    current_time = start_seconds
    end_time = start_seconds + duration

    # Sample frames across clip duration (e.g., every 0.5s)
    while current_time < end_time:
        frame_idx = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Resize for faster face detection (width=640)
        scale = 640.0 / src_w
        detect_h = int(src_h * scale)
        small_frame = cv2.resize(frame, (640, detect_h))

        detected = detector.detect(small_frame)
        frame_centers = []
        for face in detected:
            # Scale back to original coordinates
            orig_cx = int(face.center_x / scale)
            frame_centers.append(orig_cx)

        if frame_centers:
            all_face_centers.append(sorted(frame_centers))

        current_time += sample_interval

    cap.release()

    if not all_face_centers:
        logger.info("No faces detected in clip; falling back to center crop")
        return _make_center_plan(src_w, src_h, target_w, target_h)

    # Check for consistent 2-person podcast format:
    # If in >= 40% of sampled frames we see 2 distinct faces separated by at least 25% of screen width
    two_face_frames = [f for f in all_face_centers if len(f) >= 2 and (f[-1] - f[0]) > (src_w * 0.25)]
    if len(two_face_frames) >= max(3, int(len(all_face_centers) * 0.35)):
        # Podcast / 2-speaker split screen mode
        left_speakers = [f[0] for f in two_face_frames]
        right_speakers = [f[-1] for f in two_face_frames]

        median_left = int(np.median(left_speakers))
        median_right = int(np.median(right_speakers))

        logger.info(f"Detected 2 distinct speakers (left={median_left}, right={median_right}); generating podcast split-screen")
        return _make_split_screen_plan(src_w, src_h, median_left, median_right, target_w, target_h)

    # Single-speaker tracking mode
    primary_centers = [f[0] if len(f) == 1 else f[np.argmin(np.abs(np.array(f) - src_w // 2))] for f in all_face_centers]
    median_x = int(np.median(primary_centers))

    logger.info(f"Detected single primary speaker at x={median_x}; generating centered smart track")
    return _make_single_plan(src_w, src_h, median_x, target_w, target_h)


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
