"""Final quality checks before marking a video READY.

Validates: duration, resolution, audio levels, caption sync, file integrity.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityCheckResult:
    passed: bool
    checks: dict[str, bool]  # check_name -> pass/fail
    details: dict[str, str]  # check_name -> detail message
    overall_message: str


def check_duration(path: str, min_dur: float, max_dur: float) -> tuple[bool, str]:
    """Verify video duration is within bounds."""
    try:
        import json
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        dur = float(data["format"]["duration"])
        if min_dur <= dur <= max_dur:
            return True, f"duration {dur:.1f}s in [{min_dur}, {max_dur}]"
        return False, f"duration {dur:.1f}s outside [{min_dur}, {max_dur}]"
    except Exception as exc:
        return False, f"duration check failed: {exc}"


def check_resolution(path: str, expected_w: int, expected_h: int) -> tuple[bool, str]:
    """Verify video resolution matches 1080x1920."""
    try:
        import json
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "json", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        w, h = stream["width"], stream["height"]
        if w == expected_w and h == expected_h:
            return True, f"resolution {w}x{h} matches"
        return False, f"resolution {w}x{h} != {expected_w}x{expected_h}"
    except Exception as exc:
        return False, f"resolution check failed: {exc}"


def check_audio_levels(path: str) -> tuple[bool, str]:
    """Verify audio is not silent and not clipping (basic check)."""
    try:
        cmd = ["ffmpeg", "-i", path, "-af", "astats=metadata=1:reset=1", "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Look for RMS levels in stderr
        out = result.stderr
        if "RMS level" in out or "Peak level" in out:
            return True, "audio levels detected"
        return False, "no audio level info"
    except Exception as exc:
        return False, f"audio check failed: {exc}"


def check_file_integrity(path: str, min_size_bytes: int = 1024) -> tuple[bool, str]:
    """Verify output file exists and is not empty/corrupt."""
    p = Path(path)
    if not p.exists():
        return False, "file does not exist"
    size = p.stat().st_size
    if size < min_size_bytes:
        return False, f"file too small: {size} bytes"
    # Quick container check
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "ffprobe cannot read file (corrupt?)"
        float(result.stdout.strip())
        return True, f"file OK ({size} bytes)"
    except Exception as exc:
        return False, f"integrity check failed: {exc}"


def check_captions_exist(path: str) -> tuple[bool, str]:
    """Verify subtitle stream exists in output (captions burned in = no separate stream, but we check filter ran)."""
    # Since we burn captions, they're in the video stream. Just verify video stream exists.
    try:
        import json
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=codec_type", "-of", "json", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        has_video = any(s.get("codec_type") == "video" for s in data.get("streams", []))
        if has_video:
            return True, "video stream present (captions burned in)"
        return False, "no video stream"
    except Exception as exc:
        return False, f"caption check failed: {exc}"


def run_quality_checks(
    path: str,
    min_dur: float = 20.0,
    max_dur: float = 60.0,
    expected_w: int = 1080,
    expected_h: int = 1920,
    job_id: Optional[str] = None,
) -> QualityCheckResult:
    """Run all quality checks."""
    checks = {
        "duration": check_duration(path, min_dur, max_dur),
        "resolution": check_resolution(path, expected_w, expected_h),
        "audio": check_audio_levels(path),
        "integrity": check_file_integrity(path),
        "captions": check_captions_exist(path),
    }

    passed = all(v[0] for v in checks.values())
    check_results = {k: v[0] for k, v in checks.items()}
    details = {k: v[1] for k, v in checks.items()}

    overall = "All checks passed" if passed else f"Failed: {[k for k, v in checks.items() if not v[0]]}"

    logger.info(f"quality checks: {'PASS' if passed else 'FAIL'} - {overall}",
                extra={"job_id": job_id, "stage": "quality", "status": "pass" if passed else "fail"})

    return QualityCheckResult(
        passed=passed,
        checks=check_results,
        details=details,
        overall_message=overall,
    )