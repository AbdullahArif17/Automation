"""Scheduler for automated daily runs.

Supports:
- Local loop with sleep (fallback, works everywhere)
- cron (Linux/macOS via crontab -e)
- Windows Task Scheduler (schtasks)
- Docker (cron in container)

Configuration via settings (POSTS_PER_DAY, MAX_DAILY_GENERATIONS).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config.settings import get_settings
from app.runner import Pipeline, build_pipeline
from app.storage.database import Database, JobState
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DailyRunner:
    """Runs the pipeline up to N times per day."""

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        posts_per_day: int | None = None,
        max_generations: int | None = None,
        mock_llm: bool = False,
    ):
        settings = get_settings()
        self.pipeline = pipeline or build_pipeline(mock_llm=mock_llm)
        self.posts_per_day = posts_per_day or settings.posts_per_day
        self.max_generations = max_generations or settings.max_daily_generations
        self.settings = settings

    def _topics_today(self) -> int:
        """Count successful generations today."""
        db = self.pipeline.db
        today = datetime.now(timezone.utc).date().isoformat()
        row = db.fetchone(
            "SELECT COUNT(*) as cnt FROM publishing_jobs "
            "WHERE state IN ('READY','PUBLISHED') AND date(created_at)=?",
            (today,),
        )
        return row["cnt"] if row else 0

    def _needs_generation(self) -> bool:
        return self._topics_today() < self.max_generations

    def _pick_topic(self) -> Optional[str]:
        """Select next topic: prefer high-score from analytics, else discover new."""
        from app.content.topic_selector import TopicSelector
        from app.research.sources import discover_candidates, TopicCandidate

        db = self.pipeline.db
        # 1. Try topic optimization from analytics
        try:
            from app.youtube.analytics import TopicOptimizer
            opt = TopicOptimizer(db)
            best = opt.get_best_topics(limit=3)
            if best:
                topic = best[0][0]
                logger.info(f"scheduler: using top-performing topic '{topic}'",
                            extra={"stage": "scheduler", "status": "topic_from_analytics"})
                return topic
        except Exception:
            pass

        # 2. Discover fresh candidates via RSS
        candidates = discover_candidates(max_per_source=3)
        if not candidates:
            return None

        # 3. Score and pick best
        selector = TopicSelector(self.pipeline.provider, db)
        scored = selector.select_best(candidates[:10], job_id="scheduler")
        if scored and scored.final >= 0.5:
            return scored.topic
        return candidates[0].title if candidates else None

    def run_once(self) -> Optional[str]:
        """Run one generation cycle. Returns topic or None."""
        if not self._needs_generation():
            logger.info("daily limit reached", extra={"stage": "scheduler", "status": "limit"})
            return None

        topic = self._pick_topic()
        if not topic:
            logger.warning("no topic available", extra={"stage": "scheduler", "status": "no_topic"})
            return None

        logger.info(f"scheduler: generating '{topic}'",
                    extra={"stage": "scheduler", "status": "generating"})
        outcome = self.pipeline.run_topic(topic, upload=self.settings.auto_upload)

        if outcome.error:
            logger.error(f"generation failed: {outcome.error}",
                         extra={"stage": "scheduler", "status": "error"})
        else:
            logger.info(f"generated '{topic}' -> {outcome.video_id}",
                        extra={"stage": "scheduler", "status": "done"})
        return topic

    def run_loop(self, interval_seconds: int = 3600) -> None:
        """Local blocking loop: run, sleep, repeat. For local/container use."""
        logger.info(f"scheduler loop started (interval={interval_seconds}s)",
                    extra={"stage": "scheduler", "status": "loop_start"})
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.error(f"scheduler iteration error: {exc}",
                             extra={"stage": "scheduler", "status": "error"})
            time.sleep(interval_seconds)


# --- cron / Task Scheduler installation --------------------------------------

CRON_ENTRY = """# YouTube Shorts automation
# Runs at 06:00 and 18:00 local time (adjust as needed)
0 6,18 * * * {python} -m app.scheduler run-once >> {log_file} 2>&1
"""

WINDOWS_TASK = """schtasks /Create /TN "YouTubeShortsAuto" /TR "{python} -m app.scheduler run-once" /SC DAILY /ST 06:00 /F
schtasks /Create /TN "YouTubeShortsAuto_Evening" /TR "{python} -m app.scheduler run-once" /SC DAILY /ST 18:00 /F
"""


def install_cron(log_file: str | None = None) -> str:
    """Generate crontab entry. User must add to crontab manually."""
    py = sys.executable
    log = log_file or str(Path.home() / "shorts_cron.log")
    entry = CRON_ENTRY.format(python=py, log_file=log)
    print("Add this to your crontab (crontab -e):")
    print(entry)
    return entry


def install_windows_task() -> None:
    """Create Windows scheduled tasks (requires admin)."""
    py = sys.executable.replace("\\", "\\\\")
    cmds = [
        f'schtasks /Create /TN "YouTubeShortsAuto" /TR "{py} -m app.scheduler run-once" /SC DAILY /ST 06:00 /F',
        f'schtasks /Create /TN "YouTubeShortsAuto_Evening" /TR "{py} -m app.scheduler run-once" /SC DAILY /ST 18:00 /F',
    ]
    print("Run these commands as Administrator:")
    for c in cmds:
        print(c)
    # Optionally execute if admin
    for c in cmds:
        try:
            subprocess.run(c, shell=True, check=True)
            logger.info(f"installed task: {c}")
        except subprocess.CalledProcessError as exc:
            logger.warning(f"could not install task (need admin?): {exc}")


def uninstall_windows_task() -> None:
    for name in ["YouTubeShortsAuto", "YouTubeShortsAuto_Evening"]:
        subprocess.run(f'schtasks /Delete /TN "{name}" /F', shell=True, check=False)


# --- CLI entry ---------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="scheduler", description="Daily automation runner", add_help=False)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Helper to add mock-llm to a subparser
    def add_mock_llm(p):
        p.add_argument("--mock-llm", action="store_true",
                       help="Use mock LLM provider (no API keys, no cost)")

    run_once_parser = sub.add_parser("run-once", help="Run one generation cycle")
    add_mock_llm(run_once_parser)

    loop_parser = sub.add_parser("run-loop", help="Run blocking loop (local/container)")
    add_mock_llm(loop_parser)
    loop_parser.add_argument("--interval", type=int, default=3600,
                             help="Seconds between runs (default 3600)")

    install_sub = sub.add_parser("install", help="Print/install scheduler config")
    install_sub.add_argument("--cron", action="store_true", help="Print crontab entry")
    install_sub.add_argument("--windows", action="store_true", help="Install Windows tasks (admin)")
    install_sub.add_argument("--log-file", type=str, help="Log file for cron")

    args = parser.parse_args(argv)

    runner = DailyRunner(mock_llm=getattr(args, "mock_llm", False))

    if args.cmd == "run-once":
        topic = runner.run_once()
        return 0 if topic else 1

    if args.cmd == "run-loop":
        runner.run_loop(interval_seconds=args.interval)
        return 0

    if args.cmd == "install":
        if args.cron or (not args.windows and platform.system() != "Windows"):
            install_cron(args.log_file)
        if args.windows or platform.system() == "Windows":
            install_windows_task()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())