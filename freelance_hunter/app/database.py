from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FreelanceDatabase:
    def __init__(self, db_path: str | Path = "data/freelance.db"):
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
                CREATE TABLE IF NOT EXISTS seen_projects (
                    project_id TEXT PRIMARY KEY,
                    source TEXT,
                    title TEXT,
                    url TEXT,
                    budget_str TEXT,
                    posted_at TEXT,
                    first_seen_at TEXT
                );

                CREATE TABLE IF NOT EXISTS generated_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT UNIQUE,
                    source TEXT,
                    title TEXT,
                    url TEXT,
                    proposal_text TEXT,
                    alert_sent INTEGER DEFAULT 0,
                    created_at TEXT
                );
                """
            )

    def is_project_seen(self, project_id: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT 1 FROM seen_projects WHERE project_id = ?", (project_id,))
            return cur.fetchone() is not None

    def record_seen(
        self,
        project_id: str,
        source: str,
        title: str,
        url: str,
        budget_str: str = "",
        posted_at: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO seen_projects 
                (project_id, source, title, url, budget_str, posted_at, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (project_id, source, title, url, budget_str, posted_at, now),
            )

    def record_proposal(
        self,
        project_id: str,
        source: str,
        title: str,
        url: str,
        proposal_text: str,
        alert_sent: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO generated_proposals
                (project_id, source, title, url, proposal_text, alert_sent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (project_id, source, title, url, proposal_text, 1 if alert_sent else 0, now),
            )
