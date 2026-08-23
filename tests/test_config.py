"""Phase 1 tests: configuration and CLI help."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import Settings, DEFAULT_TOPIC_WEIGHTS
from app.main import main


def test_settings_defaults():
    s = Settings(
        data_dir=Path("/tmp/yt_test_data"),
        output_dir=Path("/tmp/yt_test_out"),
        assets_dir=Path("/tmp/yt_test_assets"),
        auto_upload=False,
        auto_publish=False,
    )
    assert s.video_width == 1080
    assert s.video_height == 1920
    assert s.video_fps == 30
    assert s.min_video_duration == 20
    assert s.max_video_duration == 60
    assert s.posts_per_day == 2
    assert s.auto_upload is False
    assert s.tts_provider == "local"
    assert s.media_provider == "free"
    assert abs(sum(s.topic_weights.values()) - 1.0) < 1e-6


def test_db_path_from_url():
    s = Settings(database_url="sqlite:///data/app.db")
    assert s.db_path.name == "app.db"


def test_secrets_masked_in_dict():
    s = Settings(gemini_api_key="supersecret")
    d = s.to_dict()
    assert d["gemini_api_key"] == "***"
    assert d["video_width"] == 1080


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Zero-cost" in out


def test_cli_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    assert "0.1.0" in capsys.readouterr().out
