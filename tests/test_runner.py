"""Phase 12 integration test: full pipeline run with keyword-driven mock LLM."""
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from app.ai.provider import LLMProvider, MockProvider
from app.config.settings import get_settings
from app.media.asset_manager import AssetManager, AssetRecord
from app.media.captions import CaptionLine, CaptionTrack
from app.media.voice import MockVoiceProvider, VoiceResult
from app.research.sources import TopicCandidate
from app.storage.database import Database, JobState
from app.video.editor import RenderResult, VideoEditor
from app.runner import Pipeline, build_pipeline, _now


class KeywordMockProvider(LLMProvider):
    """Returns appropriate JSON for each prompt type by keyword detection."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        self.calls.append(prompt)
        p = prompt.lower()

        # Research
        if "research" in p and "primary-source" in p:
            return json.dumps({
                "summary": "AI coding tools like Cursor and GitHub Copilot are transforming dev workflows.",
                "sources": ["https://blog.example.com/ai-coding-2026"],
                "facts": ["Cursor uses GPT-4 for code generation", "GitHub Copilot has 1.3M paid users"],
                "publication_dates": ["2026-01-15"],
                "confidence": 0.9,
            })

        # Fact verification
        if "verify each fact" in p:
            return json.dumps({
                "verified_facts": [
                    {"fact": "Cursor uses GPT-4 for code generation", "verified": True, "confidence": 0.95, "notes": "source 1"},
                    {"fact": "GitHub Copilot has 1.3M paid users", "verified": True, "confidence": 0.9, "notes": "source 1"},
                ]
            })

        # Topic scoring
        if "score this topic" in p or ("score" in p and "topic" in p and "axis" in p):
            return json.dumps({
                "trend": 0.9, "interest": 0.85, "novelty": 0.8,
                "visual": 0.7, "shorts": 0.9, "source_quality": 0.85,
            })

        # Script generation (check for "scriptwriter" - unique to script.txt)
        if "scriptwriter" in p:
            return json.dumps({
                "script": "AI coding tools are changing how developers write code. Cursor and Copilot lead the pack.",
                "hook": "Stop writing boilerplate by hand.",
                "duration_estimate_seconds": 35,
            })

        # Quality evaluation
        if "quality evaluator" in p or "score the script" in p:
            return json.dumps({
                "hook": 9, "accuracy": 9, "clarity": 9, "retention": 9,
                "novelty": 8, "pacing": 9, "visual_potential": 8,
                "naturalness": 9, "policy_risk": 0,
                "total": 8.7, "verdict": "pass", "notes": "strong script",
            })

        # Visual plan
        if "visual planner" in p or "break the script into scenes" in p:
            return json.dumps({
                "duration": 35.0,
                "scenes": [
                    {"start": 0.0, "end": 12.0, "visual_query": "AI coding assistant", "visual_type": "image", "motion": "zoom_in"},
                    {"start": 12.0, "end": 23.0, "visual_query": "developer typing", "visual_type": "image", "motion": "pan_right"},
                    {"start": 23.0, "end": 35.0, "visual_query": "code on screen", "visual_type": "image", "motion": "static"},
                ],
            })

        # Metadata
        if "metadata" in p and "youtube metadata" in p:
            return json.dumps({
                "titles": ["AI Coding Tools 2026", "Cursor vs Copilot", "Dev Tools That Write Code"],
                "description": "AI coding tools are transforming development. Here's what you need to know.",
                "hashtags": ["#ai", "#coding", "#programming", "#devtools"],
            })

        # Default fallback
        return json.dumps({
            "script": "Mock script about the topic.",
            "hook": "Hook",
            "duration_estimate_seconds": 30,
        })


class FakeAssetManager:
    """Returns valid AssetRecords without network calls."""

    def __init__(self, db: Database):
        self.db = db

    def get_or_fetch_image(self, query: str, job_id: Optional[str] = None) -> AssetRecord:
        return AssetRecord(
            id=0, source="mock", source_url="https://example.com/img.jpg",
            license="CC0", local_path=str(Path("assets") / f"mock_{query}.jpg"),
            type="image", duration=5.0, width=1080, height=1920,
            hash="abc123",
        )

    def get_or_fetch_video(self, query: str, job_id: Optional[str] = None) -> AssetRecord:
        return AssetRecord(
            id=0, source="mock", source_url="https://example.com/vid.mp4",
            license="CC0", local_path=str(Path("assets") / f"mock_{query}.mp4"),
            type="video", duration=5.0, width=1080, height=1920,
            hash="def456",
        )


class FakeEditor:
    """Returns a RenderResult without calling ffmpeg."""

    def __init__(self):
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"

    def check_available(self) -> bool:
        return True

    def render(
        self,
        plan,
        scene_assets,
        voice: VoiceResult,
        captions: CaptionTrack,
        music_path: str | None = None,
        music_volume: float = 0.1,
        output_path: str | None = None,
        job_id: str | None = None,
    ) -> RenderResult:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Write a tiny valid MP4-ish file so quality checks can stat it.
        Path(output_path).write_bytes(b"fake-mp4-content")
        return RenderResult(output_path=output_path, duration=voice.duration,
                            width=1080, height=1920)


def _make_db(tmp_path):
    return Database(str(tmp_path / "test_runner.db"))


def test_pipeline_full_run(tmp_path):
    db = _make_db(tmp_path)
    provider = KeywordMockProvider()

    # Build pipeline with all fakes
    pipe = Pipeline(
        provider=provider,
        db=db,
        asset_manager=FakeAssetManager(db),
        voice_provider=MockVoiceProvider(),
        editor=FakeEditor(),
        settings=get_settings(),
    )

    # Mock quality & duplicate checks (avoid ffprobe)
    with patch("app.runner.run_quality_checks") as mock_quality, \
         patch("app.runner.run_duplicate_checks") as mock_dup:

        mock_quality.return_value = type("QC", (), {
            "passed": True,
            "checks": {"duration": True, "resolution": True, "audio": True,
                       "integrity": True, "captions": True},
            "details": {}, "overall_message": "All checks passed",
        })()

        mock_dup.return_value = type("DC", (), {
            "is_duplicate": False, "reason": "no duplicates", "similar_items": [],
        })()

        outcome = pipe.run_topic("AI coding tools 2026", upload=False)

    assert outcome.error is None, f"pipeline error: {outcome.error}"
    assert outcome.script is not None
    assert outcome.render is not None
    assert outcome.quality is not None
    assert outcome.duplicates is not None
    assert outcome.metadata is not None
    assert not outcome.published  # AUTO_UPLOAD=false by default
    assert outcome.youtube_video_id is None

    # DB state checks
    job = db.fetchone("SELECT * FROM publishing_jobs WHERE id=?", (outcome.job_id,))
    assert job["state"] == JobState.READY.value

    video = db.fetchone("SELECT * FROM videos WHERE id=?", (outcome.video_id,))
    assert video["status"] == JobState.READY.value
    assert video["topic"] == "AI coding tools 2026"
    assert video["title"] is not None
    assert video["video_path"] is not None

    # Topic stored with score for duplicate penalty
    topic_row = db.fetchone("SELECT * FROM topics WHERE topic=?", ("AI coding tools 2026",))
    assert topic_row is not None
    assert topic_row["final_score"] > 0


def test_pipeline_resumable_state(tmp_path):
    """If pipeline fails mid-way, job state shows where it stopped."""
    db = _make_db(tmp_path)
    provider = KeywordMockProvider()

    pipe = Pipeline(
        provider=provider,
        db=db,
        asset_manager=FakeAssetManager(db),
        voice_provider=MockVoiceProvider(),
        editor=FakeEditor(),
        settings=get_settings(),
    )

    # Make asset_manager fail on image fetch to force failure at ASSET_COLLECTION
    class FailAssetManager(FakeAssetManager):
        def get_or_fetch_image(self, query: str, job_id: Optional[str] = None):
            raise RuntimeError("network down")

    pipe.assets = FailAssetManager(db)

    with patch("app.runner.run_quality_checks") as mock_quality, \
         patch("app.runner.run_duplicate_checks") as mock_dup:
        mock_quality.return_value = type("QC", (), {"passed": True})()
        mock_dup.return_value = type("DC", (), {"is_duplicate": False})()

        outcome = pipe.run_topic("Fail topic", upload=False)

    assert outcome.error is not None
    assert "network down" in outcome.error
    job = db.fetchone("SELECT * FROM publishing_jobs WHERE id=?", (outcome.job_id,))
    assert job["state"] == JobState.FAILED.value


def test_build_pipeline_mock_llm():
    """build_pipeline() with mock_llm=True returns a Pipeline with MockProvider."""
    pipe = build_pipeline(mock_llm=True)
    assert isinstance(pipe.provider, MockProvider)
    assert pipe.editor is not None
    assert pipe.assets is not None
    assert pipe.voice is not None


def test_build_pipeline_real_llm_requires_key(monkeypatch):
    """build_pipeline() without mock raises if no GEMINI_API_KEY."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_pipeline(mock_llm=False)