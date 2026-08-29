"""Phase 6 tests: voice providers (mock, espeak, piper factory)."""
import tempfile
from pathlib import Path

import pytest

from app.media.voice import (
    VoiceProvider, VoiceResult, MockVoiceProvider,
    EspeakVoiceProvider, PiperVoiceProvider, EdgeTTSVoiceProvider, get_voice_provider
)


def test_mock_voice_provider():
    p = MockVoiceProvider()
    assert p.is_available()

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "test.wav"
        res = p.synthesize("Hello world test", str(out))
        assert isinstance(res, VoiceResult)
        assert out.exists()
        assert res.duration > 0
        assert res.sample_rate == 22050
        assert res.channels == 1
        # WAV header check
        with open(out, "rb") as f:
            assert f.read(4) == b"RIFF"
            f.seek(8)
            assert f.read(4) == b"WAVE"


def test_mock_voice_duration_scales_with_text():
    p = MockVoiceProvider(words_per_minute=150)
    with tempfile.TemporaryDirectory() as td:
        short = p.synthesize("Hi", str(Path(td) / "s.wav"))
        long = p.synthesize("Hello " * 50, str(Path(td) / "l.wav"))
        assert long.duration > short.duration


def test_espeak_provider_availability():
    p = EspeakVoiceProvider()
    # Just test construction; actual binary may not exist
    assert isinstance(p.is_available(), bool)


def test_piper_provider_availability():
    p = PiperVoiceProvider()
    # Just test construction; actual binary/model may not exist
    assert isinstance(p.is_available(), bool)


def test_get_voice_provider_auto_returns_mock():
    # In CI/test environment, edge-tts is now available (installed via pip)
    # So auto should pick edge-tts first, then Piper, then espeak, then mock
    provider = get_voice_provider("auto")
    assert isinstance(provider, (EdgeTTSVoiceProvider, PiperVoiceProvider, EspeakVoiceProvider, MockVoiceProvider))
    assert provider.is_available()


def test_get_voice_provider_explicit_mock():
    provider = get_voice_provider("mock")
    assert isinstance(provider, MockVoiceProvider)


def test_get_voice_provider_unknown_falls_back_to_mock():
    # Unknown preference should gracefully fall back to mock
    provider = get_voice_provider("nonexistent")
    assert isinstance(provider, MockVoiceProvider)