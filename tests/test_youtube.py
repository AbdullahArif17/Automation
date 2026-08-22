"""Phase 10 tests: YouTube auth, uploader (mocked), publishing guard."""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.youtube.auth import YouTubeAuth, YouTubeCredentials
from app.youtube.uploader import YouTubeUploader, UploadResult


def test_auth_not_configured():
    auth = YouTubeAuth(client_id="", client_secret="", refresh_token="")
    assert not auth.is_configured()


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
    auth = YouTubeAuth(client_id="", client_secret="", refresh_token="")
    with pytest.raises(RuntimeError):
        auth.get_access_token()


def test_uploader_requires_auth():
    uploader = YouTubeUploader(auth=YouTubeAuth("", "", ""))
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
