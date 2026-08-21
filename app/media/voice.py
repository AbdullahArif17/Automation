"""Local/free TTS providers.

Abstraction so any TTS backend can be used. Defaults to:
- Piper TTS (high quality, requires piper binary + voice model)
- espeak-ng (lightweight, widely available)
- MockProvider (zero-cost testing, no binary needed)

Architecture allows paid APIs to be added later without rewrites.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.ai.provider import LLMProvider
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)


@dataclass
class VoiceResult:
    audio_path: str
    duration: float
    sample_rate: int
    channels: int


class VoiceProvider(ABC):
    """Interface every TTS backend must implement."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str, job_id: Optional[str] = None) -> VoiceResult:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider can run (binary installed, model present)."""
        pass


class MockVoiceProvider(VoiceProvider):
    """Generates a silent WAV of appropriate duration for testing (no binary needed)."""

    def __init__(self, words_per_minute: int = 150):
        self.wpm = words_per_minute

    def is_available(self) -> bool:
        return True

    def synthesize(self, text: str, output_path: str, job_id: Optional[str] = None) -> VoiceResult:
        # Estimate duration from word count
        words = len(text.split())
        duration = max(1.0, (words / self.wpm) * 60)

        # Create a minimal valid WAV file (silence)
        import wave
        sample_rate = 22050
        channels = 1
        n_frames = int(duration * sample_rate)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * n_frames)

        logger.info(f"mock voice: {duration:.1f}s -> {output_path}",
                    extra={"job_id": job_id, "stage": "voice", "status": "mock"})
        return VoiceResult(output_path, duration, sample_rate, channels)


class EspeakVoiceProvider(VoiceProvider):
    """espeak-ng TTS - lightweight, robotic but always works if installed."""

    def __init__(self, voice: str = "en-us", speed: int = 175):
        self.voice = voice
        self.speed = speed

    def is_available(self) -> bool:
        return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None

    def synthesize(self, text: str, output_path: str, job_id: Optional[str] = None) -> VoiceResult:
        if not self.is_available():
            raise RuntimeError("espeak-ng not found in PATH")

        cmd = ["espeak-ng", "-v", self.voice, "-s", str(self.speed), "-w", output_path, text]
        # Use subprocess with timeout
        def _run():
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"espeak-ng failed: {result.stderr}")
            return result

        retry(_run, max_attempts=2, retry_on=(subprocess.TimeoutExpired,))

        # Probe duration
        duration = self._probe_duration(output_path)
        logger.info(f"espeak voice: {duration:.1f}s -> {output_path}",
                    extra={"job_id": job_id, "stage": "voice", "status": "espeak"})
        return VoiceResult(output_path, duration, 22050, 1)

    def _probe_duration(self, path: str) -> float:
        try:
            import wave
            with wave.open(path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 5.0  # fallback


class PiperVoiceProvider(VoiceProvider):
    """Piper TTS - high quality neural TTS, local, free.

    Requires:
    - piper binary in PATH (or PIPER_BIN env var)
    - Voice model (.onnx) + config (.json) in PIPER_MODELS_DIR or default locations
    """

    DEFAULT_MODEL = "en_US-lessac-medium"  # good quality, ~47MB

    def __init__(self, model: str | None = None, model_dir: str | None = None,
                 piper_bin: str | None = None):
        self.model_name = model or os.getenv("PIPER_MODEL", self.DEFAULT_MODEL)
        self.model_dir = Path(model_dir or os.getenv("PIPER_MODELS_DIR", Path.home() / ".local/share/piper"))
        self.piper_bin = piper_bin or os.getenv("PIPER_BIN", "piper")
        self._model_path: Path | None = None
        self._config_path: Path | None = None

    def is_available(self) -> bool:
        if not shutil.which(self.piper_bin):
            return False
        self._resolve_model()
        return self._model_path is not None and self._model_path.exists()

    def _resolve_model(self) -> None:
        if self._model_path is not None:
            return
        # Look for model.onnx and model.onnx.json
        for name in [self.model_name, f"{self.model_name}.onnx"]:
            cand = self.model_dir / name
            if cand.exists():
                self._model_path = cand
                self._config_path = cand.with_suffix(cand.suffix + ".json")
                if not self._config_path.exists():
                    self._config_path = cand.with_name(cand.stem + ".json")
                break
            # Also check with .onnx.json directly
            cand_json = self.model_dir / f"{name}.json"
            if cand_json.exists():
                self._config_path = cand_json
                self._model_path = cand_json.with_suffix("")
                break

    def synthesize(self, text: str, output_path: str, job_id: Optional[str] = None) -> VoiceResult:
        if not self.is_available():
            raise RuntimeError("Piper not available (binary or model missing)")

        self._resolve_model()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.piper_bin,
            "--model", str(self._model_path),
            "--output_file", output_path,
        ]
        # piper reads text from stdin
        def _run():
            result = subprocess.run(cmd, input=text.encode("utf-8"),
                                    capture_output=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"piper failed: {result.stderr.decode()}")
            return result

        retry(_run, max_attempts=2, retry_on=(subprocess.TimeoutExpired,))

        duration = self._probe_duration(output_path)
        logger.info(f"piper voice: {duration:.1f}s -> {output_path}",
                    extra={"job_id": job_id, "stage": "voice", "status": "piper"})
        return VoiceResult(output_path, duration, 22050, 1)

    def _probe_duration(self, path: str) -> float:
        try:
            import wave
            with wave.open(path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 5.0


def get_voice_provider(preferred: str = "auto") -> VoiceProvider:
    """Factory: returns the best available provider.

    preferred: "piper" | "espeak" | "mock" | "auto"
    auto tries Piper -> espeak -> mock.
    """
    if preferred in ("piper", "auto"):
        p = PiperVoiceProvider()
        if p.is_available():
            logger.info("using Piper TTS", extra={"stage": "voice", "status": "piper"})
            return p
        if preferred == "piper":
            raise RuntimeError("Piper requested but not available")

    if preferred in ("espeak", "auto"):
        e = EspeakVoiceProvider()
        if e.is_available():
            logger.info("using espeak-ng TTS", extra={"stage": "voice", "status": "espeak"})
            return e
        if preferred == "espeak":
            raise RuntimeError("espeak-ng requested but not available")

    logger.info("using MockVoiceProvider (no TTS binary)", extra={"stage": "voice", "status": "mock"})
    return MockVoiceProvider()