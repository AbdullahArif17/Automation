"""Clipper pipeline orchestrator.

End-to-end: transcribe -> select highlights -> cut -> caption -> quality -> upload.
Reuses existing quality check, duplicate check, uploader, and DB tracking.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.ai.gemini import GeminiProvider
from app.ai.provider import LLMProvider
from app.clipper.captions import generate_clip_captions, ClipCaptionResult
from app.clipper.cut import cut_segment, CutResult
from app.clipper.highlight import select_highlights, ClipCandidate
from app.clipper.transcribe import transcribe_or_load, TranscriptResult
from app.config.settings import get_settings
from app.content.similarity import run_duplicate_checks, DuplicateCheckResult
from app.storage.database import Database, JobState
from app.utils.logging import get_logger
from app.video.quality import run_quality_checks, QualityCheckResult
from app.youtube.auth import YouTubeAuth
from app.youtube.uploader import YouTubeUploader

logger = get_logger(__name__)


@dataclass
class ClipOutcome:
    """Outcome of processing one clip candidate."""
    candidate: ClipCandidate
    cut_result: Optional[CutResult] = None
    caption_result: Optional[ClipCaptionResult] = None
    quality_result: Optional[QualityCheckResult] = None
    duplicate_result: Optional[DuplicateCheckResult] = None
    youtube_video_id: Optional[str] = None
    published: bool = False
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of the full clipper pipeline for one source video."""
    source_path: str
    transcript: TranscriptResult
    candidates: list[ClipCandidate]
    outcomes: list[ClipOutcome]
    job_id: int
    video_id: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClipperPipeline:
    """Runs the clipper pipeline for one source video."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        db: Optional[Database] = None,
        auth: Optional[YouTubeAuth] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or GeminiProvider()
        self.db = db or Database(self.settings.db_path)
        self.auth = auth or YouTubeAuth()
        self.output_dir = Path(self.settings.clip_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        source_path: str,
        upload: bool = False,
        max_clips: int = 1,
        force_retranscribe: bool = False,
        job_id: Optional[str] = None,
    ) -> PipelineResult:
        """Run the full clipper pipeline on a source video.

        Args:
            source_path: Path to source video file.
            upload: Whether to upload to YouTube.
            max_clips: Maximum number of clips to produce (1-3).
            force_retranscribe: Ignore cached transcript.
            job_id: Optional job ID string for logging.

        Returns:
            PipelineResult with all outcomes.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"source video not found: {source_path}")

        # Create DB records
        video_id = self.db.insert("videos", {
            "topic": f"clip:{source.stem}",
            "title": "",
            "description": "",
            "video_path": "",
            "status": JobState.CREATED.value,
            "source_type": "clipped",
            "created_at": _now(),
        })
        job_db_id = self.db.create_job(f"clip:{source.stem}", payload={"video_id": video_id, "source_path": source_path})
        self.db.execute("UPDATE publishing_jobs SET video_id=? WHERE id=?", (video_id, job_db_id))

        jid = str(job_db_id) if job_id is None else job_id

        try:
            self.db.set_job_state(job_db_id, JobState.RESEARCHING, stage="transcribe")

            # 1. Transcribe (with cache)
            transcript = transcribe_or_load(
                source_path,
                model_size=self.settings.whisper_model_size,
                force_refresh=force_retranscribe,
            )
            self.db.set_job_state(job_db_id, JobState.RESEARCHED, stage="transcribe")

            # 2. Select highlights
            self.db.set_job_state(job_db_id, JobState.SCRIPTING, stage="highlight")
            candidates = select_highlights(
                transcript,
                provider=self.provider,
                min_dur=self.settings.min_video_duration,
                max_dur=self.settings.max_video_duration,
                max_candidates=max_clips,
                job_id=jid,
            )
            if not candidates:
                raise RuntimeError("no valid highlight candidates found")

            self.db.set_job_state(job_db_id, JobState.SCRIPT_APPROVED, stage="highlight")

            # 3. Process each candidate
            outcomes: list[ClipOutcome] = []

            for idx, candidate in enumerate(candidates):
                self.db.set_job_state(job_db_id, JobState.ASSET_COLLECTION, stage=f"cut_{idx+1}")

                clip_outcome = ClipOutcome(candidate=candidate)

                try:
                    # Generate captions from whisper timestamps FIRST
                    self.db.set_job_state(job_db_id, JobState.VOICE_GENERATION, stage=f"caption_{idx+1}")
                    caption_base = str(self.output_dir / f"{source.stem}_clip_{idx+1}")
                    caption_result = generate_clip_captions(transcript, candidate, caption_base)
                    clip_outcome.caption_result = caption_result

                    # Cut segment and burn subtitles
                    clip_filename = f"{source.stem}_clip_{idx+1}.mp4"
                    clip_path = str(self.output_dir / clip_filename)

                    # Decide crop mode: if settings say 'auto', use LLM's choice, else force settings
                    chosen_crop_mode = candidate.crop_mode if self.settings.clip_crop_mode == "auto" else self.settings.clip_crop_mode
                    
                    cut_result = cut_segment(source_path, candidate, clip_path, crop_mode=chosen_crop_mode, job_id=jid, ass_path=caption_result.ass_path)
                    clip_outcome.cut_result = cut_result

                    # Quality check
                    self.db.set_job_state(job_db_id, JobState.QUALITY_CHECK, stage=f"quality_{idx+1}")
                    quality = run_quality_checks(
                        cut_result.output_path,
                        min_dur=self.settings.min_video_duration,
                        max_dur=self.settings.max_video_duration,
                        expected_w=self.settings.video_width,
                        expected_h=self.settings.video_height,
                        job_id=jid,
                    )
                    clip_outcome.quality_result = quality

                    if not quality.passed:
                        raise RuntimeError(f"quality check failed: {quality.overall_message}")

                    # Duplicate check (against existing clipped videos)
                    self.db.set_job_state(job_db_id, JobState.READY, stage=f"duplicate_{idx+1}")
                    # For clips, we compare against other clipped videos' titles/descriptions
                    # Use a simplified check since there's no script
                    duplicate = self._check_clip_duplicate(
                        candidate.suggested_title,
                        candidate.suggested_description,
                        jid,
                    )
                    clip_outcome.duplicate_result = duplicate

                    if duplicate.is_duplicate:
                        logger.warning(f"clip {idx+1} duplicate detected: {duplicate.reason}",
                                       extra={"job_id": jid, "stage": "duplicate", "status": "duplicate"})
                        # Don't fail - just log. User can decide to upload or not.

                    # Update video row with this clip's info (first successful clip becomes primary)
                    if idx == 0:
                        self.db.execute(
                            "UPDATE videos SET title=?, description=?, video_path=?, status=? WHERE id=?",
                            (candidate.suggested_title, candidate.suggested_description,
                             cut_result.output_path, JobState.READY.value, video_id),
                        )

                    # Upload if requested
                    published = False
                    yt_id = None
                    if upload:
                        self.db.set_job_state(job_db_id, JobState.UPLOADING, stage=f"upload_{idx+1}")
                        published, yt_id = self._upload_clip(
                            video_id, job_db_id, candidate, cut_result.output_path
                        )
                        clip_outcome.published = published
                        clip_outcome.youtube_video_id = yt_id

                        if not published:
                            raise RuntimeError("upload did not complete")

                    outcomes.append(clip_outcome)

                except Exception as exc:
                    clip_outcome.error = str(exc)
                    outcomes.append(clip_outcome)
                    logger.error(f"clip {idx+1} failed: {exc}",
                                 extra={"job_id": jid, "stage": f"clip_{idx+1}", "status": "error"})
                    # Continue with next candidate rather than failing whole pipeline

            self.db.set_job_state(job_db_id, JobState.PUBLISHED if any(o.published for o in outcomes) else JobState.READY, stage="done")

            return PipelineResult(
                source_path=source_path,
                transcript=transcript,
                candidates=candidates,
                outcomes=outcomes,
                job_id=job_db_id,
                video_id=video_id,
            )

        except Exception as exc:
            self.db.set_job_state(job_db_id, JobState.FAILED, stage="pipeline")
            self.db.log_error("clipper", str(exc), job_id=job_db_id)
            logger.error(f"clipper pipeline failed for '{source_path}': {exc}",
                         extra={"job_id": jid, "stage": "pipeline", "status": "error"})
            # Return partial result with error
            return PipelineResult(
                source_path=source_path,
                transcript=transcript if 'transcript' in locals() else None,
                candidates=candidates if 'candidates' in locals() else [],
                outcomes=outcomes if 'outcomes' in locals() else [],
                job_id=job_db_id,
                video_id=video_id,
            )

    def _check_clip_duplicate(
        self,
        title: str,
        description: str,
        job_id: Optional[str] = None,
    ) -> DuplicateCheckResult:
        """Check for duplicates against existing clipped videos only."""
        from app.content.similarity import jaccard_similarity, DuplicateCheckResult

        # Only check against clipped videos (source_type='clipped')
        rows = self.db.fetchall(
            "SELECT id, title, description FROM videos WHERE source_type='clipped' AND youtube_video_id IS NOT NULL"
        )

        similar = []
        for row in rows:
            past_title = row["title"] or ""
            sim = jaccard_similarity(title, past_title)
            if sim >= 0.85:
                similar.append({
                    "type": "title", "id": row["id"], "similarity": sim, "content": past_title[:100]
                })

        is_dup = len(similar) > 0
        reason = "; ".join(f"{s['type']} sim={s['similarity']:.2f}" for s in similar) if similar else "no duplicates"

        return DuplicateCheckResult(is_duplicate=is_dup, reason=reason, similar_items=similar)

    def _upload_clip(
        self,
        video_id: int,
        job_db_id: int,
        candidate: ClipCandidate,
        video_path: str,
    ) -> tuple[bool, Optional[str]]:
        """Upload a clip to YouTube."""
        if not self.settings.auto_upload:
            logger.info("AUTO_UPLOAD=false; skipping publish",
                        extra={"job_id": str(job_db_id), "stage": "upload", "status": "blocked"})
            return False, None

        if not self.auth.is_configured():
            logger.error("YouTube OAuth not configured; cannot upload",
                         extra={"job_id": str(job_db_id), "stage": "upload", "status": "error"})
            self.db.set_job_state(job_db_id, JobState.FAILED, stage="upload")
            return False, None

        privacy = "public" if self.settings.auto_publish else "private"

        uploader = YouTubeUploader(self.auth)
        result = uploader.upload(
            video_path,
            candidate.suggested_title,
            candidate.suggested_description,
            candidate.suggested_title.split(),
            privacy_status=privacy,
            job_id=str(job_db_id),
        )

        self.db.execute(
            "UPDATE videos SET youtube_video_id=?, status=?, published_at=? WHERE id=?",
            (result.video_id, JobState.PUBLISHED.value, _now(), video_id),
        )
        self.db.set_job_state(job_db_id, JobState.PUBLISHED, stage="upload")

        logger.info(f"published clip {result.video_id}",
                    extra={"job_id": str(job_db_id), "stage": "upload", "status": "published"})

        return True, result.video_id