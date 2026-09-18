from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobDatabase:
    def __init__(self, db_path: str | Path = "data/jobs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS seen_jobs (
                    job_id TEXT PRIMARY KEY,
                    source TEXT,
                    company TEXT,
                    title TEXT,
                    url TEXT,
                    location TEXT,
                    posted_at TEXT,
                    first_seen_at TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    job_id TEXT PRIMARY KEY,
                    company TEXT,
                    title TEXT,
                    match_score REAL,
                    is_eligible INTEGER,
                    matched_skills TEXT,
                    rejection_reason TEXT,
                    evaluated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE,
                    company TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    ats_type TEXT,
                    match_score REAL,
                    status TEXT NOT NULL,
                    cover_letter TEXT,
                    applied_at TEXT NOT NULL,
                    error_message TEXT,
                    screenshot_path TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_seen_jobs_posted ON seen_jobs(posted_at);
                CREATE INDEX IF NOT EXISTS idx_apps_status ON applications(status);
                """
            )

    def is_job_seen(self, job_id: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT 1 FROM seen_jobs WHERE job_id = ?", (job_id,))
            return cur.fetchone() is not None

    def record_seen_job(
        self,
        job_id: str,
        source: str,
        company: str,
        title: str,
        url: str,
        location: str = "",
        posted_at: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO seen_jobs 
                (job_id, source, company, title, url, location, posted_at, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, source, company, title, url, location, posted_at, now),
            )

    def record_evaluation(
        self,
        job_id: str,
        company: str,
        title: str,
        match_score: float,
        is_eligible: bool,
        matched_skills: str = "",
        rejection_reason: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations
                (job_id, company, title, match_score, is_eligible, matched_skills, rejection_reason, evaluated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, company, title, match_score, 1 if is_eligible else 0, matched_skills, rejection_reason, now),
            )

    def record_application(
        self,
        job_id: str,
        company: str,
        title: str,
        url: str,
        ats_type: str,
        match_score: float,
        status: str,
        cover_letter: str = "",
        error_message: str = "",
        screenshot_path: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO applications
                (job_id, company, title, url, ats_type, match_score, status, cover_letter, applied_at, error_message, screenshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    company,
                    title,
                    url,
                    ats_type,
                    match_score,
                    status,
                    cover_letter,
                    now,
                    error_message,
                    screenshot_path,
                ),
            )

    def get_applications_today_count(self) -> int:
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM applications WHERE applied_at LIKE ? AND status IN ('APPLIED', 'DRY_RUN')",
                (f"{today_prefix}%",),
            )
            return int(cur.fetchone()[0])
