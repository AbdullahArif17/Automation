"""CLI entry point for the YouTube Shorts automation pipeline.

Command surface: --topic, --dry-run, --generate, --upload, --analytics,
--status, --version. Publishing is disabled by default.
"""
from __future__ import annotations

import argparse
import json
import sys

from app.config.settings import get_settings
from app.storage.database import Database, JobState
from app.ai.provider import MockProvider
from app.ai.gemini import GeminiProvider
from app.content.script_generator import ScriptGenerator
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-shorts",
        description="Zero-cost automated YouTube Shorts generation pipeline.",
    )
    parser.add_argument("--topic", type=str, help="Generate a Short for a specific topic")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the pipeline without uploading or spending money")
    parser.add_argument("--generate", action="store_true",
                        help="Generate a Short (research -> script -> video)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload previously generated Shorts (auto-disabled unless enabled)")
    parser.add_argument("--analytics", action="store_true",
                        help="Collect and print YouTube analytics")
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline and job status")
    parser.add_argument("--mock-llm", action="store_true",
                        help="Use a free mock LLM (no API key, no cost) for local testing")
    parser.add_argument("--version", action="store_true", help="Print version")
    return parser


def _build_provider(settings, mock: bool):
    if mock:
        logger.info("using mock LLM provider (no cost)", extra={"stage": "cli", "status": "mock"})
        # Seed with valid script + passing-quality responses so the generate
        # flow completes without a key (3 candidates, attempt 1 passes).
        script_json = json.dumps({
            "script": "This is a mock script for the topic. It reads naturally and "
                      "stays accurate to the facts. Local, free tooling makes this possible.",
            "hook": "Mock hook line.", "duration_estimate_seconds": 30,
        })
        quality_json = json.dumps({
            "hook": 9, "accuracy": 9, "clarity": 9, "retention": 9, "novelty": 9,
            "pacing": 9, "visual_potential": 8, "naturalness": 9, "policy_risk": 0,
            "total": 8.8, "verdict": "pass", "notes": "mock evaluation",
        })
        responses = [script_json, script_json, script_json,
                     quality_json, quality_json, quality_json]
        return MockProvider(responses)
    try:
        return GeminiProvider()
    except RuntimeError as exc:
        logger.error(str(exc), extra={"stage": "cli", "status": "error"})
        print(f"[error] {exc}. Set GEMINI_API_KEY or pass --mock-llm for local testing.")
        sys.exit(2)


def _handle_generate(settings, topic: str, mock: bool) -> None:
    provider = _build_provider(settings, mock)
    generator = ScriptGenerator(provider, num_candidates=3,
                                 max_attempts=settings.max_regeneration_attempts)
    db = Database(settings.db_path)
    job_id = db.create_job(topic)
    db.set_job_state(job_id, JobState.SCRIPTING, stage="script")
    try:
        best = generator.generate(topic, summary="", facts=[], job_id=str(job_id))
        if best and best.evaluation and best.evaluation.passed:
            db.set_job_state(job_id, JobState.SCRIPT_APPROVED, stage="script")
        else:
            db.set_job_state(job_id, JobState.FAILED, stage="script")
            db.log_error("script", "no script passed quality threshold", job_id=job_id)
    finally:
        db.close()
    print("\n=== Generated Script ===")
    print(best.text)
    print(f"\nScore: {best.score} | Passed: {best.evaluation.passed if best.evaluation else False}")
    if best.evaluation:
        print("Notes:", best.evaluation.notes)


def _handle_status(settings) -> None:
    db = Database(settings.db_path)
    try:
        jobs = db.fetchall(
            "SELECT id, topic, state, updated_at FROM publishing_jobs ORDER BY id DESC LIMIT 20"
        )
        print(f"\n=== Jobs ({len(jobs)}) ===")
        for j in jobs:
            print(f"  #{j['id']} [{j['state']}] {j['topic']}  ({j['updated_at']})")
        if not jobs:
            print("  (no jobs yet)")
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()

    if args.version:
        print("youtube-shorts-automation 0.1.0")
        return 0

    if args.dry_run:
        print("[dry-run] pipeline planning is implemented in later phases.")

    if args.generate:
        if not args.topic:
            print("[error] --generate requires --topic \"your topic\".")
            return 2
        _handle_generate(settings, args.topic, args.mock_llm)
        return 0

    if args.topic and not args.generate:
        logger.info(f"target topic: {args.topic}", extra={"stage": "cli", "status": "topic"})
        print(f"[info] Topic received: {args.topic} (use --generate to produce a script)")

    if args.upload:
        print("[info] Upload is reserved for Phase 10 (disabled by default).")
    if args.analytics:
        print("[info] Analytics is reserved for Phase 11.")
    if args.status:
        _handle_status(settings)

    if not any([args.topic, args.dry_run, args.generate,
                args.upload, args.analytics, args.status, args.version]):
        parser.print_help()

    logger.info("config loaded", extra={"stage": "config", "status": "ok"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
