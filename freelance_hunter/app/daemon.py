from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from app.database import FreelanceDatabase
    from app.poller import fetch_all_fresh_projects
    from app.proposal_generator import ProposalGenerator
    from app.notifier import AlertNotifier
except ImportError:
    from freelance_hunter.app.database import FreelanceDatabase
    from freelance_hunter.app.poller import fetch_all_fresh_projects
    from freelance_hunter.app.proposal_generator import ProposalGenerator
    from freelance_hunter.app.notifier import AlertNotifier


def load_yaml(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_cycle(
    db: FreelanceDatabase,
    generator: ProposalGenerator,
    notifier: AlertNotifier,
    queries: list[str],
    max_age_minutes: int = 60,
) -> int:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Polling freelance feeds for {len(queries)} queries...")
    projects = fetch_all_fresh_projects(queries=queries, max_age_minutes=max_age_minutes)
    new_alerts = 0

    for proj in projects:
        if db.is_project_seen(proj.project_id):
            continue

        print(f"\n⚡ NEW PROJECT DETECTED: {proj.title} ({proj.source})")
        print(f"   Budget: {proj.budget_str} | URL: {proj.url}")

        db.record_seen(
            project_id=proj.project_id,
            source=proj.source,
            title=proj.title,
            url=proj.url,
            budget_str=proj.budget_str,
            posted_at=proj.posted_at.isoformat() if proj.posted_at else "",
        )

        # Generate proposal
        print("   Generating custom winning proposal with Gemini...")
        proposal = generator.generate_proposal(proj)

        # Send alert
        alert_sent = notifier.send_alert(proj, proposal)
        db.record_proposal(
            project_id=proj.project_id,
            source=proj.source,
            title=proj.title,
            url=proj.url,
            proposal_text=proposal,
            alert_sent=alert_sent,
        )

        new_alerts += 1

    return new_alerts


def main():
    parser = argparse.ArgumentParser(description="Freelance Fast-Responder Bot Daemon")
    parser.add_argument("--once", action="store_true", help="Run once and exit instead of continuous loop")
    parser.add_argument("--settings", type=str, default="config/settings.yaml")
    parser.add_argument("--profile", type=str, default="config/profile.yaml")
    parser.add_argument("--db", type=str, default="data/freelance.db")
    args = parser.parse_args()

    settings = load_yaml(args.settings)
    polling = settings.get("polling", {})
    interval = polling.get("interval_seconds", 180)
    max_age = polling.get("max_job_age_minutes", 60)
    queries = settings.get("filters", {}).get("queries", ["fastapi", "next.js", "python"])

    db = FreelanceDatabase(db_path=args.db)
    generator = ProposalGenerator(profile_path=args.profile)
    notifier = AlertNotifier(settings=settings)

    print("=" * 60)
    print("🚀 Freelance Fast-Responder Bot Started")
    print(f"Polling Interval: {interval}s | Max Age: {max_age}m | Target Queries: {len(queries)}")
    print(f"Telegram Alert : {'ENABLED' if settings.get('notifications', {}).get('telegram', {}).get('enabled') else 'DISABLED'}")
    print(f"Discord Alert  : {'ENABLED' if settings.get('notifications', {}).get('discord', {}).get('enabled') else 'DISABLED'}")
    print("=" * 60)

    if args.once:
        alerts = run_cycle(db, generator, notifier, queries, max_age_minutes=max_age)
        print(f"Finished one-shot check. Dispatched {alerts} alerts.")
        return

    while True:
        try:
            alerts = run_cycle(db, generator, notifier, queries, max_age_minutes=max_age)
            if alerts > 0:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Dispatched {alerts} new alert(s)!")
        except Exception as e:
            print(f"[Daemon Error] {e}")

        time.sleep(interval)


if __name__ == "__main__":
    main()
