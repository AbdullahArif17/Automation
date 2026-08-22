"""Phase 9 tests: metadata, duplicate detection, quality checks."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.content.metadata import MetadataGenerator, VideoMetadata
from app.content.similarity import (
    jaccard_similarity, check_topic_duplicate, check_script_duplicate,
    check_title_duplicate, run_duplicate_checks, DuplicateCheckResult
)
from app.video.quality import (
    run_quality_checks, QualityCheckResult, check_duration, check_resolution
)
from app.ai.provider import MockProvider
from app.storage.database import Database


def test_metadata_generator_mock():
    p = MockProvider([json.dumps({
        "titles": ["Title 1", "Title 2", "Title 3"],
        "description": "Great video about AI. #ai #tech",
        "hashtags": ["#ai", "#tech", "#programming"]
    })])
    gen = MetadataGenerator(p)
    meta = gen.generate("AI topic", "script text", 30.0)
    assert isinstance(meta, VideoMetadata)
    assert len(meta.titles) == 3
    assert all(len(t) <= 60 for t in meta.titles)
    assert len(meta.hashtags) >= 3
    assert all(h.startswith("#") for h in meta.hashtags)


def test_metadata_title_truncation():
    p = MockProvider([json.dumps({
        "titles": ["A" * 80, "B" * 80, "C" * 80],
        "description": "desc", "hashtags": []
    })])
    gen = MetadataGenerator(p)
    meta = gen.generate("topic", "script", 30.0)
    assert all(len(t) <= 60 for t in meta.titles)


def test_jaccard_similarity():
    assert jaccard_similarity("hello world", "hello world") == 1.0
    assert jaccard_similarity("hello", "world") == 0.0
    assert jaccard_similarity("a b c", "b c d") == 0.5
    assert jaccard_similarity("", "") == 1.0
    assert jaccard_similarity("", "a") == 0.0


def test_check_topic_duplicate_exact():
    mock_db = MagicMock(spec=Database)
    mock_db.fetchall.return_value = [{"id": 1, "topic": "AI coding tools"}]
    result = check_topic_duplicate(mock_db, "AI coding tools")
    assert result is not None
    assert result["similarity"] == 1.0


def test_check_topic_duplicate_high_sim():
    mock_db = MagicMock(spec=Database)
    # Use same words, different order -> high Jaccard
    mock_db.fetchall.return_value = [{"id": 1, "topic": "coding tools AI"}]
    result = check_topic_duplicate(mock_db, "AI coding tools")
    assert result is not None
    assert result["similarity"] >= 0.85


def test_check_script_duplicate():
    mock_db = MagicMock(spec=Database)
    mock_db.fetchall.return_value = [{"id": 1, "content": "Google shipped a local model"}]
    result = check_script_duplicate(mock_db, "Google shipped a local model")
    assert result is not None
    assert result["similarity"] == 1.0


def test_check_title_duplicate():
    mock_db = MagicMock(spec=Database)
    # Exact same words, different order
    mock_db.fetchall.return_value = [{"id": 1, "title": "model AI Google released"}]
    result = check_title_duplicate(mock_db, "Google AI model released")
    assert result is not None
    assert result["similarity"] == 1.0


def test_run_duplicate_checks_no_dup():
    mock_db = MagicMock(spec=Database)
    mock_db.fetchall.return_value = []
    result = run_duplicate_checks(mock_db, "new topic", "new script", "new title", "{}")
    assert result.is_duplicate is False
    assert result.similar_items == []


def test_run_duplicate_checks_with_dup():
    mock_db = MagicMock(spec=Database)
    mock_db.fetchall.return_value = [{"id": 1, "topic": "same topic", "title": "same title", "content": "same script"}]
    result = run_duplicate_checks(mock_db, "same topic", "same script", "same title", "{}")
    assert result.is_duplicate is True
    assert len(result.similar_items) > 0


# Quality checks - mock ffprobe
def test_check_duration():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout='{"format": {"duration": "35.0"}}')
        ok, msg = check_duration("fake.mp4", 20, 60)
        assert ok
        assert "35.0" in msg

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout='{"format": {"duration": "10.0"}}')
        ok, msg = check_duration("fake.mp4", 20, 60)
        assert not ok


def test_check_resolution():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout='{"streams": [{"width": 1080, "height": 1920}]}')
        ok, msg = check_resolution("fake.mp4", 1080, 1920)
        assert ok

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout='{"streams": [{"width": 1920, "height": 1080}]}')
        ok, msg = check_resolution("fake.mp4", 1080, 1920)
        assert not ok


def test_run_quality_checks_all_pass():
    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = 1000000

        def _mock_run(args, **kwargs):
            cmd = " ".join(args)
            if "format=duration" in cmd and "json" in cmd:
                return MagicMock(stdout='{"format": {"duration": "35.0"}}')
            elif "stream=width,height" in cmd:
                return MagicMock(stdout='{"streams": [{"width": 1080, "height": 1920}]}')
            elif "astats" in cmd:
                return MagicMock(stderr="RMS level: -20 dB")
            elif "format=duration" in cmd and "default" in cmd:
                return MagicMock(stdout="35.0", returncode=0)
            elif "stream=codec_type" in cmd:
                return MagicMock(stdout='{"streams": [{"codec_type": "video"}]}')
            return MagicMock(stdout="")

        mock_run.side_effect = _mock_run

        result = run_quality_checks("fake.mp4")
        assert result.passed
        assert all(result.checks.values())


def test_run_quality_checks_fail_duration():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout='{"format": {"duration": "10.0"}}')
        result = run_quality_checks("fake.mp4")
        assert not result.passed
        assert not result.checks["duration"]