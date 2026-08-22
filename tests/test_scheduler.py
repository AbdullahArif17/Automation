"""Phase 12 tests: scheduler components."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.scheduler import DailyRunner, install_cron, install_windows_task
from app.runner import Pipeline
from app.storage.database import Database, JobState


def _make_db(tmp_path):
    return Database(str(tmp_path / "test_sched.db"))


def test_daily_runner_needs_generation(tmp_path):
    db = _make_db(tmp_path)
    # No jobs today
    runner = DailyRunner(pipeline=MagicMock(spec=Pipeline), posts_per_day=2, max_generations=2)
    runner.pipeline.db = db
    assert runner._needs_generation() is True

    # Add a job from today with READY state
    today = datetime.now(timezone.utc).date().isoformat()
    db.execute(
        "INSERT INTO publishing_jobs (topic, state, created_at, updated_at) VALUES (?,?,?,?)",
        ("test", JobState.READY.value, f"{today}T10:00:00", f"{today}T10:00:00"),
    )
    assert runner._needs_generation() is True  # 1 < 2

    # Add second job
    db.execute(
        "INSERT INTO publishing_jobs (topic, state, created_at, updated_at) VALUES (?,?,?,?)",
        ("test2", JobState.PUBLISHED.value, f"{today}T11:00:00", f"{today}T11:00:00"),
    )
    assert runner._needs_generation() is False  # 2 >= 2


def test_daily_runner_pick_topic_fallback(tmp_path):
    db = _make_db(tmp_path)
    pipeline = MagicMock(spec=Pipeline)
    pipeline.db = db
    pipeline.provider = MagicMock()

    runner = DailyRunner(pipeline=pipeline, posts_per_day=2, max_generations=2)

    # No analytics, no candidates -> None
    with patch("app.research.sources.discover_candidates", return_value=[]):
        assert runner._pick_topic() is None

    # With candidates but selector returns None -> fallback to first candidate
    cand = MagicMock()
    cand.title = "fallback topic"
    with patch("app.research.sources.discover_candidates", return_value=[cand]):
        with patch("app.content.topic_selector.TopicSelector") as mock_selector:
            mock_selector.return_value.select_best.return_value = None
            topic = runner._pick_topic()
            assert topic == "fallback topic"


def test_install_cron_outputs_entry(capsys):
    install_cron()
    out = capsys.readouterr().out
    assert "crontab" in out
    assert "python" in out
    assert "app.scheduler" in out


def test_install_windows_task_no_crash():
    # Just verify it doesn't crash; actual schtasks needs admin
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        install_windows_task()
        assert mock_run.call_count == 2


def test_scheduler_run_loop_calls_run_once(tmp_path):
    """run_loop calls run_once and sleeps; we test by mocking sleep to exit."""
    db = _make_db(tmp_path)
    pipeline = MagicMock(spec=Pipeline)
    pipeline.db = db

    runner = DailyRunner(pipeline=pipeline, posts_per_day=1, max_generations=1)
    runner.pipeline.db = db

    call_count = {"n": 0}

    def mock_run_once():
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise KeyboardInterrupt  # exit loop
        return "topic"

    runner.run_once = mock_run_once

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = lambda _: None  # no-op
        try:
            runner.run_loop(interval_seconds=1)
        except KeyboardInterrupt:
            pass

    assert call_count["n"] == 2


def test_daily_runner_run_once_respects_limit(tmp_path):
    db = _make_db(tmp_path)
    pipeline = MagicMock(spec=Pipeline)
    pipeline.db = db
    pipeline.provider = MagicMock()

    # Use None to avoid fallback to settings.default (0 is falsy -> falls back)
    runner = DailyRunner(pipeline=pipeline, posts_per_day=0, max_generations=None)

    # Mock discover to avoid network calls
    with patch("app.research.sources.discover_candidates", return_value=[]):
        result = runner.run_once()
    assert result is None  # limit already reached