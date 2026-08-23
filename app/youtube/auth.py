"""YouTube OAuth 2.0 authentication (refresh-token flow, zero new deps).

Uses the Google OAuth 2.0 token endpoint with urllib (stdlib). The initial
consent happens once (interactively) to obtain a refresh token, stored in
.env as YOUTUBE_REFRESH_TOKEN. This module only refreshes access tokens.

Security:
- Never logs tokens (access or refresh).
- Reads credentials from env only.
- Fails loud if credentials missing (no silent fallback to paid services).
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from app.config.settings import get_settings
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
SCOPES = "https://www.googleapis.com/auth/youtube.upload"


@dataclass
class YouTubeCredentials:
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"{self.token_type} {self.access_token}"}


class YouTubeAuth:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None,
                 refresh_token: str | None = None):
        s = get_settings()
        self.client_id = client_id or s.youtube_client_id
        self.client_secret = client_secret or s.youtube_client_secret
        self.refresh_token = refresh_token or s.youtube_refresh_token

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.refresh_token)

    def get_access_token(self) -> str:
        """Exchange refresh token for a fresh access token."""
        if not self.is_configured():
            raise RuntimeError(
                "YouTube OAuth not configured: set YOUTUBE_CLIENT_ID, "
                "YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN in .env"
            )

        body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }
        data = json.dumps(body).encode("utf-8")

        def _post():
            req = urllib.request.Request(
                TOKEN_URL, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            result = retry(_post, max_attempts=3,
                           retry_on=(urllib.error.URLError, TimeoutError))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            logger.error(f"token refresh failed: {detail}",
                         extra={"stage": "youtube_auth", "status": "error", "error": detail})
            raise RuntimeError(f"YouTube token refresh failed: {detail}")

        token = result.get("access_token")
        if not token:
            raise RuntimeError("No access_token in token response")
        logger.info("YouTube access token refreshed",
                    extra={"stage": "youtube_auth", "status": "ok"})
        return token

    def credentials(self) -> YouTubeCredentials:
        return YouTubeCredentials(access_token=self.get_access_token())

    @staticmethod
    def build_auth_url(client_id: str, redirect_uri: str = "http://localhost") -> str:
        """Build the consent URL for the one-time interactive OAuth flow."""
        from urllib.parse import urlencode
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPES,
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    @staticmethod
    def exchange_code_for_tokens(client_id: str, client_secret: str, code: str,
                                  redirect_uri: str = "http://localhost") -> dict:
        """Exchange authorization code for tokens (one-time setup step)."""
        from urllib.parse import urlencode
        body = urlencode({
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }).encode("utf-8")
        req = urllib.request.Request(
            TOKEN_URL, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
        # Caller must store refresh_token securely in .env (never log it).
