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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
from typing import Optional

from app.config.settings import get_settings
from app.runner import Pipeline, build_pipeline
from app.storage.database import Database, JobState
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RunStatus(str, Enum):
    """Outcome of a single run_once() cycle, used by the CLI for exit codes."""
    OK = "OK"                         # generated (and uploaded, if enabled)
    NOOP_LIMIT = "NOOP_LIMIT"         # daily limit reached; nothing to do
    NOOP_NO_TOPIC = "NOOP_NO_TOPIC"   # no candidate topic available
    FAILED = "FAILED"                 # generation or upload failed


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
        self.last_run: RunStatus = RunStatus.OK

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
        """Select next topic: prefer user niche, else analytics, else discover new."""
        import os
        import random
        from app.utils.logging import get_logger
        logger = get_logger(__name__)

        niche = os.getenv("TOPIC_NICHE")
        if niche:
            clean_niche = niche.strip(' "\'')
            niche_list = [n.strip(' "\'') for n in clean_niche.split("|") if n.strip(' "\'')]
            selected_niche = random.choice(niche_list) if niche_list else clean_niche
            prompt = (
                f"You are a viral YouTube Shorts creative director. "
                f"Brainstorm ONE specific, true, and mind-blowing story topic within the niche: '{selected_niche}'. "
                f"It MUST focus on a real event, person, discovery, or crazy moment that has clear context, high stakes, and a surprising twist. "
                f"Avoid generic or vague topics like 'Top 5 moments' or 'General tips'. "
                f"Instead, pick a specific true story, for example: 'The 1998 football match where a team had to score an own goal to advance' "
                f"or 'The mathematician who proved a casino was cheating using physics'. "
                f"Return ONLY the concise topic title string (max 80 chars), with no quotes or extra text."
            )
            topic = self.pipeline.provider.generate(prompt).strip(' "\'*\n')
            logger.info(f"scheduler: selected niche '{selected_niche}', brainstormed topic '{topic}'",
                        extra={"stage": "scheduler", "status": "topic_from_niche"})
            return topic

        db = self.pipeline.db
        # 1. Try topic optimization from analytics (pick randomly among top to avoid repeating)
        try:
            from app.youtube.analytics import TopicOptimizer
            opt = TopicOptimizer(db)
            best = opt.get_best_topics(limit=5)
            if best:
                topic = random.choice(best)[0]
                logger.info(f"scheduler: using top-performing topic '{topic}'",
                            extra={"stage": "scheduler", "status": "topic_from_analytics"})
                return topic
        except Exception:
            pass

        # 2. Discover fresh candidates via RSS
        from app.research.sources import discover_candidates, TopicCandidate
        candidates = discover_candidates(max_per_source=3)
        if not candidates:
            return None

        # 3. Score and pick best
        from app.content.topic_selector import TopicSelector
        selector = TopicSelector(self.pipeline.provider, db)
        scored = selector.select_best(candidates[:10], job_id="scheduler")
        if scored and scored.final >= 0.5:
            return scored.topic
        return candidates[0].title if candidates else None

    def _recover_ready_videos(self) -> list[str]:
        """Upload READY videos that never received a youtube_video_id.

        Orphan recovery for the daily-scheduler path: `run_once` always starts
        a NEW topic, so a video that finished rendering but failed upload in a
        prior run (or whose upload succeeded at YouTube but was interrupted
        before the DB row was updated) would otherwise sit un-uploaded forever,
        since nothing in the run-once flow ever re-sweeps it.

        Also promotes any stuck UPLOADING row (NULL youtube_video_id) back to
        READY so it can be retried. Runs only when auto_upload is enabled.
        """
        if not self.settings.auto_upload:
            return []
        db = self.pipeline.db

        # Promote stuck UPLOADING rows back to READY for a clean retry.
        stuck = db.fetchall(
            "SELECT id FROM videos WHERE status='UPLOADING' AND youtube_video_id IS NULL"
        )
        for row in stuck:
            db.execute("UPDATE videos SET status=? WHERE id=?",
                       (JobState.READY.value, row["id"]))
            logger.warning(f"recovered stuck UPLOADING video #{row['id']} -> READY",
                           extra={"video_id": row["id"], "stage": "scheduler",
                                  "status": "recovered"})

        orphans = db.fetchall(
            "SELECT id, title, description, video_path FROM videos "
            "WHERE status='READY' AND youtube_video_id IS NULL"
        )
        recovered: list[str] = []
        for row in orphans:
            vid_id = row["id"]
            video_path = row["video_path"]
            if not video_path or not Path(video_path).exists():
                # Local file gone (e.g. artifact retention expired): cannot
                # re-upload, mark FAILED so it stops being swept.
                db.execute("UPDATE videos SET status=? WHERE id=?",
                           (JobState.FAILED.value, vid_id))
                logger.error(f"orphan video #{vid_id} missing local file; marked FAILED",
                             extra={"video_id": vid_id, "stage": "scheduler",
                                    "status": "error"})
                continue
            job = db.fetchone(
                "SELECT id FROM publishing_jobs WHERE video_id=? ORDER BY id DESC LIMIT 1",
                (vid_id,),
            )
            job_id = job["id"] if job else None
            published, yt_id = self.pipeline._maybe_upload(
                vid_id, job_id, row["title"] or "Untitled",
                row["description"] or "", video_path,
            )
            if published:
                recovered.append(yt_id or f"video #{vid_id}")
                logger.info(f"recovered orphan video #{vid_id} -> {yt_id}",
                            extra={"video_id": vid_id, "stage": "scheduler",
                                   "status": "recovered_published"})
            else:
                logger.error(f"failed to recover orphan video #{vid_id}",
                             extra={"video_id": vid_id, "stage": "scheduler",
                                    "status": "error"})
        return recovered

    def _recover_stuck_jobs(self, threshold_minutes: int | None = None) -> int:
        """Mark non-terminal jobs untouched for N minutes as FAILED.

        If the process was killed mid-step (step timeout, OOM, SIGKILL) the last
        JobState was persisted but the work never finished. Without this, dead
        rows accumulate forever and are never retried. A job not touched in
        `threshold_minutes` (default 30, env STUCK_JOB_MINUTES) is genuinely
        stuck — run-once itself finishes in well under that.
        """
        threshold = threshold_minutes or int(os.getenv("STUCK_JOB_MINUTES", "30"))
        db = self.pipeline.db
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=threshold)).isoformat()
        # READY is terminal only when uploads are disabled: with AUTO_UPLOAD off,
        # a finished video is deliberately left at READY (Pipeline._maybe_upload
        # returns early and never moves it past), so it must not be swept as
        # "stuck". With uploads enabled a READY job should advance quickly, so it
        # stays in the sweep. All other non-terminal states (CREATED,
        # RESEARCHING, ..., UPLOADING) are always swept past the threshold.
        terminal = {JobState.PUBLISHED.value, JobState.FAILED.value}
        if not self.settings.auto_upload:
            terminal.add(JobState.READY.value)
        non_terminal = [s.value for s in JobState if s.value not in terminal]
        placeholders = ",".join("?" * len(non_terminal))
        stuck = db.fetchall(
            f"SELECT id, state FROM publishing_jobs "
            f"WHERE state IN ({placeholders}) AND updated_at < ?",
            (*non_terminal, cutoff),
        )
        for row in stuck:
            db.set_job_state(row["id"], JobState.FAILED, stage="recovery")
            logger.warning(
                f"marked stuck job #{row['id']} ({row['state']}) as FAILED",
                extra={"job_id": row["id"], "stage": "scheduler", "status": "recovered_stuck"},
            )
        if stuck:
            logger.warning(f"recovered {len(stuck)} stuck job(s)",
                           extra={"stage": "scheduler", "status": "recovered_stuck"})
        return len(stuck)

    def _verify_production(self, outcome) -> bool:
        """Guard against a 'successful' run that produced no real artifact.

        A RunOutcome counts as truly successful only if a non-empty video file
        exists on disk, and — when uploading is enabled — it was published with
        a real YouTube id. Catches the silent 'produced nothing' case (audit §4).
        """
        if outcome.render is None or not outcome.render.output_path:
            logger.error("run reported success but produced no render result",
                         extra={"stage": "scheduler", "status": "error"})
            return False
        path = Path(outcome.render.output_path)
        if not path.exists() or path.stat().st_size == 0:
            logger.error(f"run reported success but no video file at {path}",
                         extra={"stage": "scheduler", "status": "error"})
            return False
        if self.settings.auto_upload and not outcome.youtube_video_id:
            logger.error("run reported success but no youtube_video_id set",
                         extra={"stage": "scheduler", "status": "error"})
            return False
        return True

    def run_once(self) -> Optional[str]:
        """Run one generation cycle. Returns the topic on success, else None.

        The richer outcome is recorded on `self.last_run` (RunStatus) so the CLI
        can tell a normal no-op (limit reached / no topic) from a real failure —
        both return None, but only FAILED should exit non-zero.
        """
        self.last_run = RunStatus.OK
        # Recover jobs killed mid-step and videos orphaned by prior runs before
        # generating a new topic, so failures are never silently dropped.
        self._recover_stuck_jobs()
        self._recover_ready_videos()

        if not self._needs_generation():
            logger.info("daily limit reached", extra={"stage": "scheduler", "status": "limit"})
            self.last_run = RunStatus.NOOP_LIMIT
            return None

        topic = self._pick_topic()
        if not topic:
            logger.warning("no topic available", extra={"stage": "scheduler", "status": "no_topic"})
            self.last_run = RunStatus.NOOP_NO_TOPIC
            return None

        logger.info(f"scheduler: generating '{topic}'",
                    extra={"stage": "scheduler", "status": "generating"})
        outcome = self.pipeline.run_topic(topic, upload=self.settings.auto_upload)

        if outcome.error or (self.settings.auto_upload and not outcome.published):
            logger.error(f"generation failed: {outcome.error or 'upload did not complete'}",
                         extra={"stage": "scheduler", "status": "error"})
            self.last_run = RunStatus.FAILED
            return None

        if not self._verify_production(outcome):
            self.last_run = RunStatus.FAILED
            return None

        logger.info(f"generated '{topic}' -> {outcome.video_id}",
                    extra={"stage": "scheduler", "status": "done"})
        self.last_run = RunStatus.OK
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

CRON_ENTRY = """# YouTube Shorts automation (Generative)
# Runs at 06:00 and 18:00 local time (adjust as needed)
0 6,18 * * * {python} -m app.scheduler run-once >> {log_file} 2>&1

# YouTube Shorts automation (Clipping)
# Runs at 07:00 and 19:00 local time
0 7,19 * * * {python} -m app.clipper.__auto_main__ >> {log_file}_clipper 2>&1
"""

WINDOWS_TASK = """schtasks /Create /TN "YouTubeShortsAuto_Gen" /TR "{python} -m app.scheduler run-once" /SC DAILY /ST 06:00 /F
schtasks /Create /TN "YouTubeShortsAuto_Gen_Evening" /TR "{python} -m app.scheduler run-once" /SC DAILY /ST 18:00 /F
schtasks /Create /TN "YouTubeShortsAuto_Clip" /TR "{python} -m app.clipper.__auto_main__" /SC DAILY /ST 07:00 /F
schtasks /Create /TN "YouTubeShortsAuto_Clip_Evening" /TR "{python} -m app.clipper.__auto_main__" /SC DAILY /ST 19:00 /F
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
        f'schtasks /Create /TN "YouTubeShortsAuto_Gen" /TR "{py} -m app.scheduler run-once" /SC DAILY /ST 06:00 /F',
        f'schtasks /Create /TN "YouTubeShortsAuto_Gen_Evening" /TR "{py} -m app.scheduler run-once" /SC DAILY /ST 18:00 /F',
        f'schtasks /Create /TN "YouTubeShortsAuto_Clip" /TR "{py} -m app.clipper.__auto_main__" /SC DAILY /ST 07:00 /F',
        f'schtasks /Create /TN "YouTubeShortsAuto_Clip_Evening" /TR "{py} -m app.clipper.__auto_main__" /SC DAILY /ST 19:00 /F',
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
    tasks = [
        "YouTubeShortsAuto", "YouTubeShortsAuto_Evening", 
        "YouTubeShortsAuto_Gen", "YouTubeShortsAuto_Gen_Evening",
        "YouTubeShortsAuto_Clip", "YouTubeShortsAuto_Clip_Evening"
    ]
    for name in tasks:
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
        # NOOP_LIMIT / NOOP_NO_TOPIC / OK are normal; only FAILED exits non-zero.
        return 1 if runner.last_run == RunStatus.FAILED else 0

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