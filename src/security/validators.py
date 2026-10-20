"""Central input validation helpers (workspace ids, upload filenames, path containment, URLs).

These guards are shared by the API layer and the persistence layer so that untrusted
client input can never escape the workspace directory tree (path traversal) or write
arbitrary files on disk.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

_WS_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]")

ALLOWED_UPLOAD_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".pbix", ".pbit"}
)


class ValidationError(ValueError):
    """Raised when untrusted input fails validation (mapped to HTTP 400 by the API)."""


def normalize_workspace_id(raw: str) -> str:
    """Normalise a user-supplied workspace id and reject anything path-like.

    Accepts 1-64 chars of ``a-z``, ``0-9``, ``_`` or ``-`` (after lower-casing and
    replacing spaces with underscores). Rejects ``..``, slashes, drive letters, etc.
    """
    ws = (raw or "").strip().lower().replace(" ", "_")
    if not _WS_ID_RE.fullmatch(ws):
        raise ValidationError(
            "Workspace id must be 1-64 characters of a-z, 0-9, '_' or '-' and start with a letter or digit."
        )
    return ws


def safe_upload_name(raw: str | None) -> str:
    """Strip any directory components from an uploaded filename and validate its extension."""
    candidate = (raw or "").replace("\\", "/")
    name = PurePosixPath(candidate).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    if not name or name in {".", ".."}:
        name = "upload.csv"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValidationError(
            f"Unsupported file type '{suffix or '(none)'}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}"
        )
    return name[:180]


def ensure_within(base: Path, target: Path) -> Path:
    """Return the resolved ``target`` if it lies inside ``base``; raise otherwise."""
    base_r = base.resolve()
    tgt_r = target.resolve()
    if tgt_r != base_r and not tgt_r.is_relative_to(base_r):
        raise ValidationError("Path escapes the allowed directory.")
    return tgt_r


def validate_http_url(raw: str) -> str:
    """Validate that ``raw`` is an absolute http(s) URL (used for custom LLM api_base)."""
    url = (raw or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationError("api_base must be an absolute http:// or https:// URL.")
    return url
