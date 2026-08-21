"""Phase 2 tests: database schema, job states, hashing, files."""
import json
from pathlib import Path

import pytest

from app.storage.database import Database, JobState
from app.utils.hashing import sha256_text, normalized_hash
from app.storage import files as filemod


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


def test_schema_created(db):
    tables = {r["name"] for r in db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {"topics", "research", "scripts", "assets", "videos",
                "publishing_jobs", "job_transitions", "youtube_videos",
                "analytics", "errors"}
    assert expected.issubset(tables)


def test_job_lifecycle(db):
    job_id = db.create_job("local AI models")
    assert db.get_job(job_id)["state"] == JobState.CREATED.value

    db.set_job_state(job_id, JobState.RESEARCHING, stage="research")
    db.set_job_state(job_id, JobState.RESEARCHED, stage="research")

    row = db.get_job(job_id)
    assert row["state"] == JobState.RESEARCHED.value

    transitions = db.fetchall(
        "SELECT * FROM job_transitions WHERE job_id=?", (job_id,))
    assert len(transitions) == 3  # CREATED + 2 transitions
    assert transitions[0]["to_state"] == JobState.CREATED.value
    assert transitions[-1]["to_state"] == JobState.RESEARCHED.value


def test_error_logging(db):
    db.log_error("render", "ffmpeg failed", "exit 1", job_id=1)
    errs = db.fetchall("SELECT * FROM errors")
    assert len(errs) == 1
    assert errs[0]["stage"] == "render"


def test_insert_serializes_json(db):
    tid = db.insert("topics", {
        "topic": "AI tools",
        "final_score": 0.8,
        "status": "candidate",
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    row = db.fetchone("SELECT * FROM topics WHERE id=?", (tid,))
    assert row["topic"] == "AI tools"
    assert row["final_score"] == 0.8


def test_hashing():
    assert sha256_text("hello") == sha256_text("hello")
    assert normalized_hash("Hello  World") == normalized_hash("hello world")
    assert normalized_hash("AI tools") != normalized_hash("weather today")


def test_file_save_and_size(tmp_path):
    p = tmp_path / "sub" / "out.bin"
    filemod.save_bytes(b"abc", p)
    assert p.read_bytes() == b"abc"
    assert filemod.file_size_ok(p)

    with pytest.raises(ValueError):
        filemod.save_bytes(b"x" * 10, tmp_path / "big.bin", max_bytes=5)
