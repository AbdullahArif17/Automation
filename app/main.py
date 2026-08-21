"""CLI entry point for the YouTube Shorts automation pipeline.

Phase 1 implements the command surface only. Subcommands that depend on later
phases (research, generate, upload, analytics) are wired but report that the
phase is not yet implemented.
"""
from __future__ import annotations

import argparse
import sys

from app.config.settings import get_settings
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
    parser.add_argument("--version", action="store_true", help="Print version")
    return parser


def _not_implemented(phase: str) -> None:
    logger.warning(
        f"{phase} is not implemented in this phase. Continuing with Phase 1 skeleton.",
        extra={"stage": "cli", "status": "skipped"},
    )
    print(f"[Phase 1] {phase} reserved for a later phase.")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()

    if args.version:
        print("youtube-shorts-automation 0.1.0")
        return 0

    if args.topic:
        logger.info(f"target topic: {args.topic}", extra={"stage": "cli", "status": "topic"})
        print(f"[Phase 1] Topic received: {args.topic}")

    if args.dry_run:
        _not_implemented("Dry run")
    if args.generate:
        _not_implemented("Generation")
    if args.upload:
        _not_implemented("Upload")
    if args.analytics:
        _not_implemented("Analytics")
    if args.status:
        _not_implemented("Status")

    if not any([args.topic, args.dry_run, args.generate,
                args.upload, args.analytics, args.status, args.version]):
        parser.print_help()

    logger.info(
        "config loaded",
        extra={"stage": "config", "status": "ok", "job_id": None},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
