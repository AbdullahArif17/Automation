"""Phase 10 tests: YouTube auth, uploader (mocked), publishing guard.
Also covers YouTube trending topics fetch (Option C).
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from app.research.youtube_trending import TrendingTopic, fetch_trending_topics, fetch_trending_topics_api, fetch_trending_topics_scrape
from app.youtube.auth import YouTubeAuth, YouTubeCredentials
from app.youtube.uploader import YouTubeUploader, UploadResult


def test_auth_not_configured():
    # Patch settings to return empty values so auth is truly unconfigured
    import app.config.settings as s
    original = (s.get_settings().youtube_client_id, s.get_settings().youtube_client_secret, s.get_settings().youtube_refresh_token)
    try:
        s._settings = None
        s.get_settings().youtube_client_id = ""
        s.get_settings().youtube_client_secret = ""
        s.get_settings().youtube_refresh_token = ""
        auth = YouTubeAuth(client_id=None, client_secret=None, refresh_token=None)
        assert not auth.is_configured()
    finally:
        s._settings = None


def test_credentials_auth_header():
    creds = YouTubeCredentials(access_token="abc123")
    header = creds.auth_header()
    assert header["Authorization"] == "Bearer abc123"


def test_build_auth_url():
    url = YouTubeAuth.build_auth_url("my_client_id")
    assert "accounts.google.com" in url
    assert "client_id=my_client_id" in url
    assert "youtube.upload" in url
    assert "access_type=offline" in url


def test_get_access_token_success():
    auth = YouTubeAuth(client_id="c", client_secret="s", refresh_token="r")
    # urlopen must return a context manager yielding a response with .read()
    import contextlib
    fake_cm = contextlib.nullcontext(MagicMock(
        read=lambda: json.dumps({"access_token": "tok_123", "expires_in": 3600}).encode()
    ))

    with patch("urllib.request.urlopen", return_value=fake_cm) as mock_open:
        token = auth.get_access_token()
        assert token == "tok_123"
        # Verify request body contains refresh token (not logged, just present)
        req = mock_open.call_args[0][0]
        body = req.data.decode()
        assert "refresh_token" in body
        assert "r" in body


def test_get_access_token_not_configured():
    with patch("app.youtube.auth.get_settings") as mock_settings:
        mock_settings.return_value.youtube_client_id = ""
        mock_settings.return_value.youtube_client_secret = ""
        mock_settings.return_value.youtube_refresh_token = ""
        auth = YouTubeAuth()
        with pytest.raises(RuntimeError):
            auth.get_access_token()


def test_uploader_requires_auth():
    with patch("app.youtube.auth.get_settings") as mock_settings:
        mock_settings.return_value.youtube_client_id = ""
        mock_settings.return_value.youtube_client_secret = ""
        mock_settings.return_value.youtube_refresh_token = ""
        uploader = YouTubeUploader(auth=YouTubeAuth())
        with pytest.raises(RuntimeError):
            uploader.upload("fake.mp4", "title", "desc", ["tag"])


def test_upload_result():
    r = UploadResult("vid123", "My Title", "private", "https://youtu.be/vid123")
    assert r.video_id == "vid123"
    assert r.url == "https://youtu.be/vid123"


def test_upload_success():
    auth = YouTubeAuth(client_id="c", client_secret="s", refresh_token="r")
    uploader = YouTubeUploader(auth=auth)

    import contextlib

    init_cm = contextlib.nullcontext(MagicMock(
        headers=MagicMock(get=lambda k: "https://upload.example.com/abc")
    ))
    video_cm = contextlib.nullcontext(MagicMock(
        read=lambda: json.dumps({"id": "vid_xyz"}).encode()
    ))

    call_count = {"n": 0}
    def mock_urlopen(req, timeout=30):
        call_count["n"] += 1
        return init_cm if call_count["n"] == 1 else video_cm

    with patch("urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000000), \
         patch("builtins.open", MagicMock()), \
         patch.object(auth, "credentials", return_value=YouTubeCredentials("tok")):
        result = uploader.upload("fake.mp4", "Title", "Desc", ["tag1", "tag2"], "private")
        assert isinstance(result, UploadResult)
        assert result.video_id == "vid_xyz"
        assert result.privacy_status == "private"


def test_upload_init_failure():
    auth = YouTubeAuth(client_id="c", client_secret="s", refresh_token="r")
    uploader = YouTubeUploader(auth=auth)

    import contextlib
    init_cm = contextlib.nullcontext(MagicMock(headers=MagicMock(get=lambda k: None)))

    with patch("urllib.request.urlopen", return_value=init_cm), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000000), \
         patch.object(auth, "credentials", return_value=YouTubeCredentials("tok")):
        with pytest.raises(RuntimeError):
            uploader.upload("fake.mp4", "Title", "Desc", ["tag"])


# --- YouTube Trending Topics tests --------------------------------------------

def test_trending_topic_dataclass():
    t = TrendingTopic("Test Title", ["kw1", "kw2"], "28", 10000)
    assert t.title == "Test Title"
    assert t.keywords == ["kw1", "kw2"]
    assert t.category_id == "28"
    assert t.view_count == 10000


def test_fetch_trending_api_requires_key(monkeypatch):
    """Without YOUTUBE_API_KEY, API fetch returns empty list."""
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = fetch_trending_topics_api(max_results=5)
    assert result == []


def test_fetch_trending_api_mocked():
    """Mock a successful API response."""
    mock_json = json.dumps({
        "items": [{
            "snippet": {
                "title": "Hot AI Video",
                "tags": ["AI", "tech"],
                "categoryId": "28",
            },
            "statistics": {"viewCount": "123456"},
            "topicDetails": {"topicCategories": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]}
        }]
    })

    with patch("urllib.request.urlopen") as mock_open:
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda s: MagicMock(read=lambda: mock_json.encode())
        mock_cm.__exit__ = lambda s, *a: None
        mock_open.return_value = mock_cm

        with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test_key"}):
            result = fetch_trending_topics_api(max_results=5)

    assert len(result) == 1
    assert result[0].title == "Hot AI Video"
    assert "AI" in result[0].keywords
    assert "Artificial intelligence" in result[0].keywords
    assert result[0].view_count == 123456


def test_fetch_trending_scrape_fallback(monkeypatch):
    """Scrape returns empty on error (no network in tests)."""
    # Don't actually scrape in tests - just verify the function exists and handles missing data
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.side_effect = Exception("network error")
        result = fetch_trending_topics_scrape()
        assert result == []


def test_fetch_trending_prefers_api(monkeypatch):
    """When API key is set, fetch_trending_topics calls API path."""
    with patch("app.research.youtube_trending.fetch_trending_topics_api") as mock_api:
        mock_api.return_value = [TrendingTopic("API Result", ["kw"], "28", 100)]
        with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test_key"}):
            result = fetch_trending_topics(prefer_api=True)
        mock_api.assert_called_once()
        assert result[0].title == "API Result"
