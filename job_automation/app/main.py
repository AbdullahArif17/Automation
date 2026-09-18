from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from app.database import JobDatabase
    from app.fetcher import fetch_all_fresh_jobs
    from app.matcher import JobMatcher
    from app.applier import JobApplier
except ImportError:
    from job_automation.app.database import JobDatabase
    from job_automation.app.fetcher import fetch_all_fresh_jobs
    from job_automation.app.matcher import JobMatcher
    from job_automation.app.applier import JobApplier


def run_job_automation(
    dry_run: bool = False,
    max_apps: int = 5,
    max_age_hours: int = 48,
    min_match_score: float = 75.0,
    db_path: str = "data/jobs.db",
    profile_path: str = "config/profile.yaml",
    resume_path: str = "assets/resume.pdf",
):
    print("=" * 60)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting Daily Job Automation")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE AUTO-APPLY'}")
    print(f"Max Applications: {max_apps} | Min Match Score: {min_match_score}% | Max Age: {max_age_hours}h")
    print("=" * 60)

    db = JobDatabase(db_path=db_path)
    matcher = JobMatcher(profile_path=profile_path)
    applier = JobApplier(profile_path=profile_path, resume_path=resume_path, dry_run=dry_run)

    today_applied = db.get_applications_today_count()
    if today_applied >= max_apps:
        print(f"[Daily Limit] Already reached daily limit of {today_applied}/{max_apps} applications. Exiting.")
        return

    remaining_slots = max_apps - today_applied
    print(f"[Quota] Remaining application slots today: {remaining_slots}")

    print(f"[Fetcher] Scanning free public feeds for jobs posted in last {max_age_hours}h...")
    jobs = fetch_all_fresh_jobs(max_age_hours=max_age_hours)
    print(f"[Fetcher] Discovered {len(jobs)} fresh job postings.")

    applied_count = 0
    matched_count = 0
    skipped_count = 0

    for job in jobs:
        if applied_count >= remaining_slots:
            print(f"[Quota] Daily application target reached ({applied_count}/{remaining_slots}). Stopping.")
            break

        # Check if already processed
        if db.is_job_seen(job.job_id):
            continue

        db.record_seen_job(
            job_id=job.job_id,
            source=job.source,
            company=job.company,
            title=job.title,
            url=job.url,
            location=job.location,
            posted_at=job.posted_at.isoformat() if job.posted_at else "",
        )

        print(f"\n--- Evaluating: {job.title} @ {job.company} [{job.source.upper()}] ---")
        match = matcher.evaluate_job(job)

        db.record_evaluation(
            job_id=job.job_id,
            company=job.company,
            title=job.title,
            match_score=match.match_score,
            is_eligible=match.is_eligible,
            matched_skills=", ".join(match.matched_skills),
            rejection_reason=match.rejection_reason,
        )

        if not match.is_eligible or match.match_score < min_match_score:
            print(f"  [DISCARD] Score: {match.match_score:.1f}% | Reason: {match.rejection_reason or 'Below threshold'}")
            skipped_count += 1
            continue

        matched_count += 1
        print(f"  [MATCH] Score: {match.match_score:.1f}% | Skills: {', '.join(match.matched_skills)}")
        print(f"  [Action] Applying to {job.url} ...")

        success, msg, screenshot = applier.apply(job)
        status = "DRY_RUN" if dry_run else ("APPLIED" if success else "FAILED")

        db.record_application(
            job_id=job.job_id,
            company=job.company,
            title=job.title,
            url=job.url,
            ats_type=job.ats_type,
            match_score=match.match_score,
            status=status,
            error_message="" if success else msg,
            screenshot_path=screenshot,
        )

        if success:
            applied_count += 1
            print(f"  [SUCCESS] {status}: {msg}")
        else:
            print(f"  [FAILURE] Could not submit: {msg}")

    print("\n" + "=" * 60)
    print("Daily Job Automation Run Summary:")
    print(f"  Fresh Jobs Scanned : {len(jobs)}")
    print(f"  High-Match Roles   : {matched_count}")
    print(f"  Skipped / Low-Fit  : {skipped_count}")
    print(f"  Applied (Today)    : {applied_count}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Automated Job Application System")
    parser.add_argument("--dry-run", action="store_true", help="Fill forms and take screenshots without submitting")
    parser.add_argument("--max-apps", type=int, default=5, help="Maximum applications to submit per day")
    parser.add_argument("--max-age-hours", type=int, default=48, help="Maximum job age in hours")
    parser.add_argument("--min-match-score", type=float, default=75.0, help="Minimum percentage score to apply")
    parser.add_argument("--db-path", type=str, default="data/jobs.db", help="Path to SQLite database")
    parser.add_argument("--profile", type=str, default="config/profile.yaml", help="Path to candidate profile YAML")
    parser.add_argument("--resume", type=str, default="assets/resume.pdf", help="Path to PDF resume")

    args = parser.parse_args()

    run_job_automation(
        dry_run=args.dry_run,
        max_apps=args.max_apps,
        max_age_hours=args.max_age_hours,
        min_match_score=args.min_match_score,
        db_path=args.db_path,
        profile_path=args.profile,
        resume_path=args.resume,
    )
