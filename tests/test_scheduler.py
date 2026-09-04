"""Phase 12 tests: scheduler components."""
import os
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
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

    with patch.dict(os.environ, {"TOPIC_NICHE": ""}):
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
    # Just verify it doesn't crash; actual schtasks needs admin (2 gen + 2 clip tasks)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        install_windows_task()
        assert mock_run.call_count == 4


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


def _new_runner(db, auto_upload):
    """Build a DailyRunner over `db` with auto_upload forced on/off.

    Returns (runner, prev_auto_upload) so the caller can restore the shared
    settings singleton in a finally block.
    """
    pipeline = MagicMock(spec=Pipeline)
    pipeline.db = db
    runner = DailyRunner(pipeline=pipeline, posts_per_day=1, max_generations=1)
    runner.pipeline.db = db
    prev = runner.settings.auto_upload
    runner.settings.auto_upload = auto_upload
    return runner, prev


# --- _recover_stuck_jobs -----------------------------------------------------

def test_recover_stuck_jobs_preserves_ready_when_upload_off(tmp_path):
    db = _make_db(tmp_path)
    runner, prev = _new_runner(db, auto_upload=False)
    try:
        old = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        db.execute(
            "INSERT INTO publishing_jobs (topic, state, current_step, payload_json, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            ("ready-topic", JobState.READY.value, "ready", "{}", old, old),
        )
        db.execute(
            "INSERT INTO publishing_jobs (topic, state, current_step, payload_json, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            ("mid-topic", JobState.RESEARCHING.value, "research", "{}", old, old),
        )
        n = runner._recover_stuck_jobs(threshold_minutes=30)
        # Only the in-progress (RESEARCHING) job is swept; READY is protected
        # because it is the deliberate end-state when uploads are disabled.
        assert n == 1
        ready_row = db.fetchone("SELECT state FROM publishing_jobs WHERE topic='ready-topic'")
        mid_row = db.fetchone("SELECT state FROM publishing_jobs WHERE topic='mid-topic'")
        assert ready_row["state"] == JobState.READY.value
        assert mid_row["state"] == JobState.FAILED.value
    finally:
        runner.settings.auto_upload = prev


def test_recover_stuck_jobs_fails_ready_when_upload_on(tmp_path):
    db = _make_db(tmp_path)
    runner, prev = _new_runner(db, auto_upload=True)
    try:
        old = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        db.execute(
            "INSERT INTO publishing_jobs (topic, state, current_step, payload_json, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            ("ready-topic", JobState.READY.value, "ready", "{}", old, old),
        )
        n = runner._recover_stuck_jobs(threshold_minutes=30)
        # With uploads enabled, a READY job past the threshold is genuinely
        # stuck and should be marked FAILED.
        assert n == 1
        row = db.fetchone("SELECT state FROM publishing_jobs WHERE topic='ready-topic'")
        assert row["state"] == JobState.FAILED.value
    finally:
        runner.settings.auto_upload = prev


# --- _recover_ready_videos ---------------------------------------------------

def _insert_ready_video(db, video_path):
    db.execute(
        "INSERT INTO videos (topic, title, description, video_path, status, "
        "youtube_video_id, created_at) VALUES (?,?,?,?,?,?,?)",
        ("t", "Title", "Desc", str(video_path), JobState.READY.value, None,
         datetime.now(timezone.utc).isoformat()),
    )


def test_recover_ready_videos_uploads_orphan(tmp_path):
    db = _make_db(tmp_path)
    runner, prev = _new_runner(db, auto_upload=True)
    try:
        vid = tmp_path / "short.mp4"
        vid.write_bytes(b"dummy")
        _insert_ready_video(db, vid)
        with patch.object(runner.pipeline, "_maybe_upload", return_value=(True, "YT123")):
            recovered = runner._recover_ready_videos()
        assert recovered == ["YT123"]
    finally:
        runner.settings.auto_upload = prev


def test_recover_ready_videos_upload_failure_keeps_ready(tmp_path):
    db = _make_db(tmp_path)
    runner, prev = _new_runner(db, auto_upload=True)
    try:
        vid = tmp_path / "short.mp4"
        vid.write_bytes(b"dummy")
        _insert_ready_video(db, vid)
        with patch.object(runner.pipeline, "_maybe_upload", return_value=(False, None)):
            recovered = runner._recover_ready_videos()
        assert recovered == []
        row = db.fetchone("SELECT status FROM videos WHERE id=1")
        assert row["status"] == JobState.READY.value
    finally:
        runner.settings.auto_upload = prev


def test_recover_ready_videos_missing_file_marked_failed(tmp_path):
    db = _make_db(tmp_path)
    runner, prev = _new_runner(db, auto_upload=True)
    try:
        # Point at a file that does not exist on disk.
        _insert_ready_video(db, tmp_path / "gone.mp4")
        recovered = runner._recover_ready_videos()
        assert recovered == []
        row = db.fetchone("SELECT status FROM videos WHERE id=1")
        assert row["status"] == JobState.FAILED.value
    finally:
        runner.settings.auto_upload = prev


# --- _verify_production ------------------------------------------------------

def test_verify_production_success(tmp_path):
    runner, prev = _new_runner(_make_db(tmp_path), auto_upload=True)
    try:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x" * 100)
        outcome = SimpleNamespace(
            render=SimpleNamespace(output_path=str(f)),
            youtube_video_id="YT1",
        )
        assert runner._verify_production(outcome) is True
    finally:
        runner.settings.auto_upload = prev


def test_verify_production_no_render_fails(tmp_path):
    runner, prev = _new_runner(_make_db(tmp_path), auto_upload=True)
    try:
        outcome = SimpleNamespace(render=None, youtube_video_id="YT1")
        assert runner._verify_production(outcome) is False
    finally:
        runner.settings.auto_upload = prev


def test_verify_production_missing_file_fails(tmp_path):
    runner, prev = _new_runner(_make_db(tmp_path), auto_upload=True)
    try:
        outcome = SimpleNamespace(
            render=SimpleNamespace(output_path=str(tmp_path / "nope.mp4")),
            youtube_video_id="YT1",
        )
        assert runner._verify_production(outcome) is False
    finally:
        runner.settings.auto_upload = prev


def test_verify_production_no_youtube_id_fails(tmp_path):
    runner, prev = _new_runner(_make_db(tmp_path), auto_upload=True)
    try:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x" * 100)
        outcome = SimpleNamespace(
            render=SimpleNamespace(output_path=str(f)),
            youtube_video_id=None,
        )
        assert runner._verify_production(outcome) is False
    finally:
        runner.settings.auto_upload = prev