"""End-to-end pipeline orchestrator.

Chains every stage (topic select -> research -> script -> visual plan ->
assets -> voice -> captions -> render -> quality -> duplicate check ->
metadata -> optional upload) into one runnable unit. All collaborators are
injected so the pipeline is fully testable with mock providers (no API keys,
no network, no ffmpeg).

State is tracked via the `publishing_jobs` / `job_transitions` tables so a run
is resumable, and the `videos` table mirrors status so the uploader works.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from app.ai.provider import LLMProvider
from app.config.settings import Settings, get_settings
from app.content.metadata import MetadataGenerator
from app.content.script_generator import GeneratedScript, ScriptGenerator
from app.content.similarity import DuplicateCheckResult, run_duplicate_checks
from app.content.topic_selector import TopicSelector
from app.content.visual_plan import VisualPlan, VisualPlanner
from app.media.asset_manager import AssetManager
from app.media.captions import CaptionTrack, split_into_caption_lines
from app.media.voice import VoiceProvider
from app.research.sources import TopicCandidate
from app.storage.database import Database, JobState
from app.utils.logging import get_logger
from app.video.quality import QualityCheckResult, run_quality_checks
from app.video.editor import RenderResult, VideoEditor

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunOutcome:
    job_id: int
    video_id: int
    topic: str
    script: Optional[GeneratedScript]
    render: Optional[RenderResult]
    quality: Optional[QualityCheckResult]
    duplicates: Optional[DuplicateCheckResult]
    metadata: Optional[dict]
    published: bool = False
    youtube_video_id: Optional[str] = None
    error: Optional[str] = None


class Pipeline:
    """Runs the full Shorts pipeline for one or many topics."""

    def __init__(
        self,
        provider: LLMProvider,
        db: Database,
        asset_manager: AssetManager,
        voice_provider: VoiceProvider,
        editor: VideoEditor,
        settings: Settings | None = None,
        auth=None,
        quality_fn: Callable = run_quality_checks,
        dup_fn: Callable = run_duplicate_checks,
    ):
        self.provider = provider
        self.db = db
        self.assets = asset_manager
        self.voice = voice_provider
        self.editor = editor
        self.settings = settings or get_settings()
        self.auth = auth
        self.quality_fn = quality_fn
        self.dup_fn = dup_fn
        self.output_dir = Path(self.settings.output_dir or "output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- job/state helpers ---------------------------------------------------
    def _new_job(self, topic: str) -> tuple[int, int]:
        """Create a videos row + a publishing_job, return (video_id, job_id)."""
        video_id = self.db.insert(
            "videos",
            {"topic": topic, "status": JobState.CREATED.value, "created_at": _now()},
        )
        job_id = self.db.create_job(topic, payload={"video_id": video_id})
        self.db.execute("UPDATE publishing_jobs SET video_id=? WHERE id=?", (video_id, job_id))
        return video_id, job_id

    def _store_topic_score(self, topic: str, final: float) -> None:
        self.db.insert(
            "topics",
            {
                "topic": topic,
                "final_score": final,
                "status": "selected",
                "created_at": _now(),
            },
        )

    # --- one-topic run -------------------------------------------------------
    def run_topic(self, topic: str, upload: bool = False) -> RunOutcome:
        video_id, job_id = self._new_job(topic)
        jid = str(job_id)
        try:
            self.db.set_job_state(job_id, JobState.RESEARCHING, stage="research")

            # 1. Select + research + score
            selector = TopicSelector(self.provider, self.db)
            candidate = TopicCandidate(title=topic, url="", summary="", source="manual")
            scored = selector.select_best([candidate], job_id=jid)
            if scored is None:
                raise RuntimeError("topic selection returned no candidate")
            self._store_topic_score(scored.topic, scored.final)
            self.db.set_job_state(job_id, JobState.RESEARCHED, stage="research")
            research = scored.research

            # 2. Script
            self.db.set_job_state(job_id, JobState.SCRIPTING, stage="script")
            # Reduced for free tier quota (5 RPM, 20 RPD)
            gen = ScriptGenerator(self.provider, num_candidates=1, max_attempts=1)
            script = gen.generate(
                scored.topic, summary=research.summary if research else "",
                facts=research.facts if research else [], job_id=jid,
            )
            if script is None or not script.evaluation or not script.evaluation.passed:
                self.db.set_job_state(job_id, JobState.FAILED, stage="script")
                raise RuntimeError(
                    f"script failed quality check (score={script.score if script else 0})"  # noqa: E501
                )
            self.db.set_job_state(job_id, JobState.SCRIPT_APPROVED, stage="script")

            # 3. Visual plan
            planner = VisualPlanner(self.provider)
            plan = planner.plan(script.text, scored.topic, script.duration_estimate, job_id=jid)

            # 4. Assets per scene
            self.db.set_job_state(job_id, JobState.ASSET_COLLECTION, stage="asset")
            scene_assets = self._collect_assets(plan, jid)

            # 5. Voice
            self.db.set_job_state(job_id, JobState.VOICE_GENERATION, stage="voice")
            voice_path = str(self.output_dir / f"voice_{job_id}.wav")
            voice = self.voice.synthesize(script.text, voice_path, job_id=jid)

            # 6. Captions - use word boundaries if available (e.g., from edge-tts)
            word_boundaries = self.voice.get_word_boundaries(script.text)
            captions: CaptionTrack = split_into_caption_lines(
                script.text, voice.duration, word_boundaries=word_boundaries
            )

            # 7. Render
            self.db.set_job_state(job_id, JobState.RENDERING, stage="render")
            out_path = str(self.output_dir / f"short_{job_id}.mp4")
            render = self.editor.render(
                plan, scene_assets, voice, captions,
                output_path=out_path, job_id=jid,
            )

            # 8. Quality
            self.db.set_job_state(job_id, JobState.QUALITY_CHECK, stage="quality")
            quality = self.quality_fn(
                render.output_path,
                min_dur=self.settings.min_video_duration,
                max_dur=self.settings.max_video_duration,
                expected_w=self.settings.video_width,
                expected_h=self.settings.video_height,
                job_id=jid,
            )

            # 9. Metadata
            meta_gen = MetadataGenerator(self.provider)
            meta = meta_gen.generate(scored.topic, script.text, render.duration, job_id=jid)
            meta_dict = meta.to_dict()

            # 10. Duplicate check
            duplicates = self.dup_fn(
                self.db, scored.topic, script.text,
                meta.primary_title(), plan.to_json(), job_id=jid,
            )

            # Persist video row
            self.db.execute(
                "UPDATE videos SET script=?, title=?, description=?, video_path=?, "
                "status=? WHERE id=?",
                (script.text, meta.primary_title(), meta_dict.get("description", ""),
                 render.output_path, JobState.READY.value, video_id),
            )
            self.db.set_job_state(job_id, JobState.READY, stage="ready")

            published = False
            yt_id = None
            if upload:
                published, yt_id = self._maybe_upload(video_id, job_id, meta.primary_title(),
                                                      meta_dict.get("description", ""),
                                                      render.output_path)

            return RunOutcome(
                job_id=job_id, video_id=video_id, topic=scored.topic,
                script=script, render=render, quality=quality,
                duplicates=duplicates, metadata=meta_dict,
                published=published, youtube_video_id=yt_id,
            )
        except Exception as exc:  # resilience: mark failed, don't crash the run loop
            self.db.set_job_state(job_id, JobState.FAILED, stage="pipeline")
            self.db.log_error("pipeline", str(exc), job_id=job_id)
            logger.error(f"pipeline failed for '{topic}': {exc}",
                         extra={"job_id": jid, "stage": "pipeline", "status": "error"})
            return RunOutcome(
                job_id=job_id, video_id=video_id, topic=topic,
                script=None, render=None, quality=None,
                duplicates=None, metadata=None, error=str(exc),
            )

    def _collect_assets(self, plan: VisualPlan, job_id: str) -> list:
        assets = []
        for scene in plan.scenes:
            if scene.visual_type == "video":
                asset = self.assets.get_or_fetch_video(scene.visual_query, job_id=job_id)
            else:
                asset = self.assets.get_or_fetch_image(scene.visual_query, job_id=job_id)
            if asset is None:
                # Fallback to an image query so the render step still has inputs.
                asset = self.assets.get_or_fetch_image(scene.visual_query or "abstract", job_id=job_id)
            if asset is None:
                # Last resort: create a minimal placeholder asset record
                # The VideoEditor will handle missing files gracefully
                from app.media.asset_manager import AssetRecord
                asset = AssetRecord(
                    id=0, source="placeholder", source_url="", license="CC0",
                    local_path="", type="image", duration=scene.end - scene.start,
                    width=self.settings.video_width, height=self.settings.video_height,
                    hash="placeholder"
                )
            assets.append(asset)
        return assets

    def _maybe_upload(self, video_id: int, job_id: int, title: str,
                      description: str, video_path: str) -> tuple[bool, Optional[str]]:
        from app.youtube.auth import YouTubeAuth
        from app.youtube.uploader import YouTubeUploader

        if not self.settings.auto_upload:
            logger.info("AUTO_UPLOAD=false; skipping publish",
                        extra={"job_id": str(job_id), "stage": "upload", "status": "blocked"})
            return False, None
        auth = self.auth or YouTubeAuth()
        if not auth.is_configured():
            logger.error("YouTube OAuth not configured; cannot upload",
                         extra={"job_id": str(job_id), "stage": "upload", "status": "error"})
            self.db.set_job_state(job_id, JobState.FAILED, stage="upload")
            return False, None

        privacy = "public" if self.settings.auto_publish else "private"
        self.db.set_job_state(job_id, JobState.UPLOADING, stage="upload")
        try:
            uploader = YouTubeUploader(auth)
            result = uploader.upload(
                video_path, title, description, title.split(),
                privacy_status=privacy, job_id=str(job_id),
            )
            self.db.execute(
                "UPDATE videos SET youtube_video_id=?, status=?, published_at=? WHERE id=?",
                (result.video_id, JobState.PUBLISHED.value, _now(), video_id),
            )
            self.db.set_job_state(job_id, JobState.PUBLISHED, stage="upload")
            logger.info(f"published {result.video_id}",
                        extra={"job_id": str(job_id), "stage": "upload", "status": "published"})
            return True, result.video_id
        except Exception as exc:
            self.db.set_job_state(job_id, JobState.FAILED, stage="upload")
            self.db.log_error("upload", str(exc), job_id=job_id)
            logger.error(f"upload failed: {exc}",
                         extra={"job_id": str(job_id), "stage": "upload", "status": "error"})
            return False, None


def build_pipeline(settings: Settings | None = None, mock_llm: bool = False) -> Pipeline:
    """Construct a real Pipeline from settings/env (for CLI / scheduler use)."""
    from app.ai.gemini import GeminiProvider
    from app.ai.provider import MockProvider
    from app.media.asset_manager import AssetManager
    from app.media.image_provider import PexelsProvider, UnsplashSourceProvider
    from app.media.stock_provider import PexelsVideoProvider, PixabayVideoProvider
    from app.media.voice import get_voice_provider
    from app.video.editor import VideoEditor

    settings = settings or get_settings()
    provider = MockProvider() if mock_llm else GeminiProvider()
    db = Database(settings.db_path)

    # Wire up all configured free-tier providers for resilience
    unsplash = UnsplashSourceProvider()
    pexels_img = PexelsProvider(settings.pexels_api_key) if settings.pexels_api_key else None
    pexels_vid = PexelsVideoProvider(settings.pexels_api_key) if settings.pexels_api_key else None
    pixabay_vid = PixabayVideoProvider(settings.pixabay_api_key) if settings.pixabay_api_key else None

    assets = AssetManager(db, unsplash=unsplash, pexels_img=pexels_img,
                          pexels_vid=pexels_vid, pixabay_vid=pixabay_vid, settings=settings)
    voice = get_voice_provider(settings.tts_provider)
    editor = VideoEditor()
    return Pipeline(provider, db, assets, voice, editor, settings)
