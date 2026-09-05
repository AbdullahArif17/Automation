"""SQLite storage layer.

Single source of truth for schema initialization and all persistence
(topics, research, scripts, assets, videos, jobs, youtube_videos, analytics,
errors). Schema is created idempotently on connect.

Job states follow the spec (section 28). Every state transition is logged in
`job_transitions` so the pipeline is resumable.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


class JobState(str, Enum):
    CREATED = "CREATED"
    RESEARCHING = "RESEARCHING"
    RESEARCHED = "RESEARCHED"
    SCRIPTING = "SCRIPTING"
    SCRIPT_APPROVED = "SCRIPT_APPROVED"
    ASSET_COLLECTION = "ASSET_COLLECTION"
    VOICE_GENERATION = "VOICE_GENERATION"
    RENDERING = "RENDERING"
    QUALITY_CHECK = "QUALITY_CHECK"
    READY = "READY"
    UPLOADING = "UPLOADING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    REGENERATE = "REGENERATE"


SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    summary TEXT,
    trend_score REAL,
    novelty_score REAL,
    interest_score REAL,
    visual_score REAL,
    shorts_score REAL,
    source_quality_score REAL,
    duplicate_penalty REAL,
    final_score REAL,
    status TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER,
    summary TEXT,
    sources_json TEXT,
    facts_json TEXT,
    publication_dates_json TEXT,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS scripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER,
    content TEXT NOT NULL,
    quality_json TEXT,
    score REAL,
    version INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    source_url TEXT,
    license TEXT,
    local_path TEXT,
    type TEXT,
    duration REAL,
    width INTEGER,
    height INTEGER,
    hash TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    script TEXT,
    title TEXT,
    description TEXT,
    video_path TEXT,
    status TEXT,
    youtube_video_id TEXT,
    created_at TEXT NOT NULL,
    published_at TEXT,
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    watch_time INTEGER DEFAULT 0,
    average_view_duration REAL DEFAULT 0,
    source_type TEXT DEFAULT 'generated',  -- 'generated' | 'clipped'
    source_etag TEXT,
    source_size INTEGER
);

CREATE TABLE IF NOT EXISTS publishing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    topic TEXT,
    state TEXT NOT NULL,
    current_step TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS job_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    stage TEXT,
    message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES publishing_jobs(id)
);

CREATE TABLE IF NOT EXISTS youtube_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT UNIQUE NOT NULL,
    title TEXT,
    description TEXT,
    tags_json TEXT,
    privacy_status TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    youtube_video_id TEXT,
    views INTEGER,
    likes INTEGER,
    comments INTEGER,
    watch_time INTEGER,
    average_view_duration REAL,
    collected_at TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    stage TEXT,
    message TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        logger.info(f"database ready at {self.db_path}",
                    extra={"stage": "database", "status": "ok"})

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns that did not exist when older databases were first created.

        Runs idempotently: each ALTER TABLE is guarded by a column-existence check.
        """
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(videos)")}

        # source_type: distinguish 'generated' vs 'clipped' videos.
        if "source_type" not in existing:
            self.conn.execute(
                "ALTER TABLE videos ADD COLUMN source_type TEXT DEFAULT 'generated'"
            )
            logger.info("migrated videos table: added source_type column",
                        extra={"stage": "database", "status": "migrated"})

        # source_etag + source_size: de-dup key for clipped videos (S3 ETag + bytes).
        if "source_etag" not in existing:
            self.conn.execute("ALTER TABLE videos ADD COLUMN source_etag TEXT")
            logger.info("migrated videos table: added source_etag column",
                        extra={"stage": "database", "status": "migrated"})
        if "source_size" not in existing:
            self.conn.execute("ALTER TABLE videos ADD COLUMN source_size INTEGER")
            logger.info("migrated videos table: added source_size column",
                        extra={"stage": "database", "status": "migrated"})

        # Create performance indexes safely after tables and columns exist
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_source_etag ON videos(source_etag)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_topic ON videos(topic)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_youtube_id ON videos(youtube_video_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_publishing_jobs_state ON publishing_jobs(state)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_publishing_jobs_video_id ON publishing_jobs(video_id)")

        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    # --- Jobs ---------------------------------------------------------------
    def create_job(self, topic: str, payload: Optional[dict] = None) -> int:
        now = _now()
        cur = self.execute(
            "INSERT INTO publishing_jobs (topic, state, current_step, payload_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (topic, JobState.CREATED.value, None, json.dumps(payload or {}), now, now),
        )
        job_id = cur.lastrowid
        self.log_transition(job_id, None, JobState.CREATED.value, "created")
        return job_id

    def log_transition(self, job_id: int, from_state: Optional[str],
                       to_state: str, stage: Optional[str] = None,
                       message: Optional[str] = None) -> None:
        self.execute(
            "INSERT INTO job_transitions (job_id, from_state, to_state, stage, message, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, from_state, to_state, stage, message, _now()),
        )
        self.execute(
            "UPDATE publishing_jobs SET state=?, current_step=?, updated_at=? WHERE id=?",
            (to_state, stage, _now(), job_id),
        )
        logger.info(
            f"job {job_id}: {from_state} -> {to_state}",
            extra={"job_id": str(job_id), "stage": stage or "job", "status": to_state},
        )

    def set_job_state(self, job_id: int, state: JobState,
                      stage: Optional[str] = None) -> None:
        row = self.fetchone("SELECT state FROM publishing_jobs WHERE id=?", (job_id,))
        from_state = row["state"] if row else None
        self.log_transition(job_id, from_state, state.value, stage)

    def get_job(self, job_id: int) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM publishing_jobs WHERE id=?", (job_id,))

    # --- Errors -------------------------------------------------------------
    def log_error(self, stage: str, message: str, error: str = "",
                  job_id: Optional[int] = None) -> None:
        self.execute(
            "INSERT INTO errors (job_id, stage, message, error, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, stage, message, error, _now()),
        )
        logger.error(message, extra={"job_id": str(job_id) if job_id else None,
                                     "stage": stage, "status": "error", "error": error[:500]})

    # --- Generic insert helper ---------------------------------------------
    def insert(self, table: str, data: dict[str, Any]) -> int:
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        cur = self.execute(sql, tuple(self._serialize(v) for v in data.values()))
        return cur.lastrowid

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return v

    def get_topic_analytics_summary(self) -> dict[str, dict[str, float]]:
        """Return avg views, avg likes, avg comments and count per topic category."""
        rows = self.fetchall("""
            SELECT 
                topic,
                COUNT(*) as count,
                AVG(views) as avg_views,
                AVG(likes) as avg_likes,
                AVG(comments) as avg_comments
            FROM videos 
            WHERE youtube_video_id IS NOT NULL AND topic IS NOT NULL
            GROUP BY topic
        """)
        summary = {}
        for r in rows:
            summary[r["topic"]] = {
                "count": r["count"],
                "avg_views": r["avg_views"] or 0.0,
                "avg_likes": r["avg_likes"] or 0.0,
                "avg_comments": r["avg_comments"] or 0.0,
            }
        return summary
