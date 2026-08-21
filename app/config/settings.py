"""Configuration system for the YouTube Shorts automation pipeline.

Loads values from environment variables and an optional `.env` file, with
sensible defaults. Secrets are never hard-coded here — they are read at runtime
from the environment or `.env` (which must not be committed).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dependency guard
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Topic scoring weights — configurable, sum should be ~1.0.
DEFAULT_TOPIC_WEIGHTS: dict[str, float] = {
    "trend": 0.25,
    "interest": 0.20,
    "novelty": 0.15,
    "shorts": 0.15,
    "visual": 0.10,
    "source_quality": 0.15,
}


@dataclass
class Settings:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))

    # LLM / AI
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    # YouTube OAuth
    youtube_client_id: str = field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_ID", ""))
    youtube_client_secret: str = field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET", ""))
    youtube_refresh_token: str = field(default_factory=lambda: os.getenv("YOUTUBE_REFRESH_TOKEN", ""))
    channel_id: str = field(default_factory=lambda: os.getenv("CHANNEL_ID", ""))

    # Publishing
    posts_per_day: int = field(default_factory=lambda: _env_int("POSTS_PER_DAY", 2))
    max_daily_generations: int = field(default_factory=lambda: _env_int("MAX_DAILY_GENERATIONS", 2))
    auto_upload: bool = field(default_factory=lambda: _env_bool("AUTO_UPLOAD", False))
    auto_publish: bool = field(default_factory=lambda: _env_bool("AUTO_PUBLISH", False))

    # Video
    video_width: int = field(default_factory=lambda: _env_int("VIDEO_WIDTH", 1080))
    video_height: int = field(default_factory=lambda: _env_int("VIDEO_HEIGHT", 1920))
    video_fps: int = field(default_factory=lambda: _env_int("VIDEO_FPS", 30))
    min_video_duration: int = field(default_factory=lambda: _env_int("MIN_VIDEO_DURATION", 20))
    max_video_duration: int = field(default_factory=lambda: _env_int("MAX_VIDEO_DURATION", 60))

    # Providers
    tts_provider: str = field(default_factory=lambda: os.getenv("TTS_PROVIDER", "local"))
    media_provider: str = field(default_factory=lambda: os.getenv("MEDIA_PROVIDER", "free"))

    # Quality thresholds
    min_script_quality: float = field(default_factory=lambda: float(os.getenv("MIN_SCRIPT_QUALITY", "7.5")))
    max_regeneration_attempts: int = field(default_factory=lambda: _env_int("MAX_REGENERATION_ATTEMPTS", 3))

    # Topic scoring weights
    topic_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOPIC_WEIGHTS))

    # Paths
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    assets_dir: Path = field(default_factory=lambda: ASSETS_DIR)
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///data/app.db"))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        url = self.database_url
        if url.startswith("sqlite:///"):
            rel = url[len("sqlite:///"):]
            if rel.startswith("/"):
                return Path(rel)
            # Relative path resolves against the project root (e.g. data/app.db).
            return BASE_DIR / rel
        return self.data_dir / "app.db"

    def to_dict(self) -> dict[str, Any]:
        """Return config as a dict with secrets masked."""
        d = {}
        for k, v in self.__dict__.items():
            if "secret" in k or "token" in k or "api_key" in k:
                d[k] = "***" if v else ""
            else:
                d[k] = v
        return d


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
