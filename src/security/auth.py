"""API authentication, session cookies, and origin checks for the Data_Talk web server.

Model
-----
* Every browser gets a signed, random ``dt_session`` cookie (``HttpOnly``, ``SameSite=Strict``).
  Cross-site pages cannot make the browser attach a ``SameSite=Strict`` cookie, which blocks
  CSRF / drive-by requests against the local API.
* Scripts can instead send ``Authorization: Bearer $DATA_TALK_API_TOKEN``.
* When ``DATA_TALK_API_TOKEN`` is set ("token mode", recommended for remote/Docker deployments)
  the browser must log in once with the token before a session cookie is issued.
* State-changing requests carrying a foreign ``Origin`` header are rejected (defence in depth).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field

from starlette.requests import Request

SESSION_COOKIE = "dt_session"
API_SESSION_ID = "api"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:8504",
    "http://localhost:8504",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)


def _split_csv(raw: str | None) -> list[str]:
    return [p.strip().rstrip("/") for p in (raw or "").split(",") if p.strip()]


@dataclass
class AuthConfig:
    """Runtime auth settings (read from environment by :meth:`from_env`)."""

    api_token: str | None = None
    allowed_origins: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ORIGINS))
    session_secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    frame_ancestors: str = "'self' http://localhost:* http://127.0.0.1:*"

    @classmethod
    def from_env(cls) -> AuthConfig:
        token = (os.environ.get("DATA_TALK_API_TOKEN") or "").strip() or None
        origins = _split_csv(os.environ.get("DATA_TALK_ALLOWED_ORIGINS")) or list(DEFAULT_ALLOWED_ORIGINS)
        secret_raw = (os.environ.get("DATA_TALK_SESSION_SECRET") or "").strip()
        secret = secret_raw.encode("utf-8") if secret_raw else secrets.token_bytes(32)
        frame = (os.environ.get("DATA_TALK_FRAME_ANCESTORS") or "").strip() or cls.frame_ancestors
        return cls(api_token=token, allowed_origins=origins, session_secret=secret, frame_ancestors=frame)

    @property
    def token_mode(self) -> bool:
        return bool(self.api_token)


class Authenticator:
    """Issues/verifies signed session cookies and bearer tokens."""

    def __init__(self, config: AuthConfig | None = None) -> None:
        self.config = config or AuthConfig.from_env()

    # -- session cookie ----------------------------------------------------
    def _sign(self, session_id: str) -> str:
        return hmac.new(self.config.session_secret, session_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]

    def new_session_cookie(self) -> str:
        sid = secrets.token_urlsafe(24)
        return f"{sid}.{self._sign(sid)}"

    def session_id_from_cookie(self, cookie_value: str | None) -> str | None:
        if not cookie_value or "." not in cookie_value:
            return None
        sid, sig = cookie_value.rsplit(".", 1)
        if not sid or not hmac.compare_digest(sig, self._sign(sid)):
            return None
        return sid

    # -- bearer token ------------------------------------------------------
    def check_token(self, candidate: str | None) -> bool:
        expected = self.config.api_token
        if not expected or not candidate:
            return False
        return hmac.compare_digest(candidate.strip().encode("utf-8"), expected.encode("utf-8"))

    def bearer_ok(self, request: Request) -> bool:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return False
        return self.check_token(header[7:])

    # -- request-level helpers --------------------------------------------
    def resolve_session_id(self, request: Request) -> str | None:
        """Return the session id for an authenticated request, or ``None`` if unauthenticated."""
        sid = self.session_id_from_cookie(request.cookies.get(SESSION_COOKIE))
        if sid:
            return sid
        if self.bearer_ok(request):
            return API_SESSION_ID
        return None

    def origin_allowed(self, request: Request) -> bool:
        """Reject state-changing requests whose ``Origin`` is neither same-origin nor allow-listed."""
        if request.method.upper() in _SAFE_METHODS:
            return True
        origin = request.headers.get("origin")
        if not origin or (origin == "null" and self.bearer_ok(request)):
            return True
        origin = origin.rstrip("/")
        own = f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"
        return origin == own or origin in self.config.allowed_origins

    @staticmethod
    def cookie_kwargs(request: Request) -> dict[str, object]:
        return {
            "httponly": True,
            "samesite": "strict",
            "secure": request.url.scheme == "https",
            "path": "/",
        }


LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" /><title>Data_Talk — Sign in</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
 body{font-family:Inter,system-ui,sans-serif;background:#090d16;color:#e2e8f0;display:grid;place-items:center;height:100vh;margin:0}
 form{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:28px;width:320px}
 h1{font-size:18px;margin:0 0 16px} input{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0}
 button{margin-top:14px;width:100%;padding:10px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-weight:600;cursor:pointer}
 .err{color:#f87171;font-size:13px;margin-top:10px}
</style></head><body>
<form method="post" action="/auth/login">
 <h1>Data_Talk access token</h1>
 <input type="password" name="token" placeholder="DATA_TALK_API_TOKEN" autofocus required />
 <button type="submit">Sign in</button>
 __ERROR__
</form></body></html>"""
