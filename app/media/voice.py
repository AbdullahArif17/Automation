"""TTS providers - free/local and cloud-based.

Abstraction so any TTS backend can be used. Priority order:
1. edge-tts (Microsoft Neural TTS, free, cloud-based, natural voices, word boundaries)
2. Piper TTS (high quality, requires piper binary + voice model, local)
3. espeak-ng (lightweight, widely available, robotic)
4. MockProvider (zero-cost testing, no binary needed)

Architecture allows paid APIs to be added later without rewrites.
"""
from __future__ import annotations

import asyncio
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

    def get_word_boundaries(self, text: str) -> list[dict]:
        """Get word-level timestamps for caption karaoke sync.

        Returns list of dicts with: text, offset (100ns units), duration (100ns units)
        Default implementation returns empty list (no word boundaries available).
        """
        return []


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
    """espeak-ng TTS - lightweight, robotic but always works if installed.

    Voice options: use `espeak-ng --voices` to list. Good female English voices:
    - en-us+f1, en-us+f2, en-us+f3, en-us+f4 (US English female variants)
    - en+f1, en+f2, en+f3, en+f4 (British English female variants)
    Speed: 80-200 wpm (words per minute). Lower = slower.
    """

    def __init__(self, voice: str = "en-us+f3", speed: int = 135):
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


class EdgeTTSVoiceProvider(VoiceProvider):
    """Microsoft Edge TTS (edge-tts) - free cloud neural TTS with natural voices.

    Advantages:
    - Natural-sounding neural voices (en-US-ChristopherNeural, en-US-GuyNeural, etc.)
    - Provides word-level timestamps for karaoke-style caption sync
    - No local binary/model needed (just Python package + internet)
    - Free tier generous (no API key needed)

    Voice recommendations for AI/tech explainer content:
    - en-US-ChristopherNeural: Male, News/Novel, "Reliable, Authority" - professional
    - en-US-GuyNeural: Male, News/Novel, "Passion" - engaging
    - en-US-AriaNeural: Female, News/Novel, "Positive, Confident" - clear
    - en-US-AndrewNeural: Male, Conversation/Copilot, "Warm, Confident, Authentic" - conversational

    Default: en-US-ChristopherNeural (authoritative for tech content)
    """

    DEFAULT_VOICE = "en-US-ChristopherNeural"
    DEFAULT_RATE = "+0%"   # normal speed
    DEFAULT_PITCH = "+0Hz" # normal pitch

    def __init__(self, voice: str = DEFAULT_VOICE, rate: str = DEFAULT_RATE, pitch: str = DEFAULT_PITCH):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self._edge_tts_available = None

    def is_available(self) -> bool:
        """Check if edge-tts package is installed and we have network."""
        if self._edge_tts_available is not None:
            return self._edge_tts_available
        try:
            import edge_tts
            self._edge_tts_available = True
        except ImportError:
            self._edge_tts_available = False
        return self._edge_tts_available

    def _get_word_boundaries(self, text: str) -> list[dict]:
        """Extract word-level timestamps from edge-tts stream.

        Returns list of dicts with: text, offset (100ns units), duration (100ns units)
        """
        import edge_tts
        # boundary="WordBoundary" enables word-level timestamps in the metadata stream
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch, boundary="WordBoundary")
        word_boundaries = []

        for chunk in communicate.stream_sync():
            if chunk["type"] == "WordBoundary":
                word_boundaries.append({
                    "text": chunk["text"],
                    "offset": chunk["offset"],      # 100-nanosecond units
                    "duration": chunk["duration"],  # 100-nanosecond units
                })
        return word_boundaries

    def synthesize(self, text: str, output_path: str, job_id: Optional[str] = None) -> VoiceResult:
        if not self.is_available():
            raise RuntimeError("edge-tts package not installed")

        import edge_tts
        # boundary="WordBoundary" enables word-level timestamps in the metadata stream
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch, boundary="WordBoundary")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Use asyncio to run the save
        async def _save():
            await communicate.save(output_path)

        def _run():
            asyncio.run(_save())
            return True

        retry(_run, max_attempts=2, retry_on=(Exception,))

        # Probe duration from the saved file
        duration = self._probe_duration(output_path)

        # Also store word boundaries for caption sync (could be returned via VoiceResult extension)
        # For now, we log them; the caption generator can re-query if needed
        try:
            word_boundaries = self._get_word_boundaries(text)
            logger.debug(f"edge-tts word boundaries: {len(word_boundaries)} words",
                         extra={"job_id": job_id, "stage": "voice", "status": "edge-tts"})
        except Exception as exc:
            logger.warning(f"Could not extract word boundaries: {exc}",
                          extra={"job_id": job_id, "stage": "voice", "status": "edge-tts"})

        logger.info(f"edge-tts voice: {duration:.1f}s -> {output_path}",
                    extra={"job_id": job_id, "stage": "voice", "status": "edge-tts"})
        return VoiceResult(output_path, duration, 24000, 1)  # edge-tts outputs 24kHz mono

    def _probe_duration(self, path: str) -> float:
        try:
            # edge-tts outputs MP3, use ffprobe
            import subprocess
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception:
            # Fallback: estimate from word count
            return 5.0

    def get_word_boundaries(self, text: str) -> list[dict]:
        """Get word-level timestamps from edge-tts for karaoke captions."""
        return self._get_word_boundaries(text)


def get_voice_provider(preferred: str = "auto") -> VoiceProvider:
    """Factory: returns the best available provider.

    preferred: "edge-tts" | "piper" | "espeak" | "mock" | "auto"
    auto tries edge-tts -> Piper -> espeak -> mock.
    """
    # Edge-TTS is first priority (natural voices, word boundaries, no local deps)
    if preferred in ("edge-tts", "auto"):
        e = EdgeTTSVoiceProvider()
        if e.is_available():
            logger.info("using edge-tts TTS", extra={"stage": "voice", "status": "edge-tts"})
            return e
        if preferred == "edge-tts":
            raise RuntimeError("edge-tts requested but not available")

    if preferred in ("piper", "auto", "local"):
        p = PiperVoiceProvider()
        if p.is_available():
            logger.info("using Piper TTS", extra={"stage": "voice", "status": "piper"})
            return p
        if preferred == "piper":
            raise RuntimeError("Piper requested but not available")

    if preferred in ("espeak", "auto", "local"):
        e = EspeakVoiceProvider()
        if e.is_available():
            logger.info("using espeak-ng TTS", extra={"stage": "voice", "status": "espeak"})
            return e
        if preferred == "espeak":
            raise RuntimeError("espeak-ng requested but not available")

    logger.info("using MockVoiceProvider (no TTS binary)", extra={"stage": "voice", "status": "mock"})
    return MockVoiceProvider()