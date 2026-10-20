"""Tests for API authentication, origin defense, upload validation, and security headers."""

from io import BytesIO
from fastapi.testclient import TestClient

from src.security.auth import SESSION_COOKIE


def test_open_mode_issues_session_cookie(test_client: TestClient):
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert SESSION_COOKIE in resp.cookies
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("referrer-policy") == "same-origin"


def test_token_mode_requires_authentication(authed_client: tuple[TestClient, str]):
    client, token = authed_client

    # 1. /api/health is always unauthenticated for monitoring
    health = client.get("/api/health")
    assert health.status_code == 200

    # 2. Protected routes return 401 without auth
    unauth = client.get("/api/state")
    assert unauth.status_code == 401

    # 3. Bearer token allows access
    bearer_res = client.get("/api/state", headers={"Authorization": f"Bearer {token}"})
    assert bearer_res.status_code == 200

    # 4. Web login form flow sets session cookie
    login_fail = client.post("/auth/login", data={"token": "wrong-token"}, follow_redirects=False)
    assert login_fail.status_code == 401

    login_ok = client.post("/auth/login", data={"token": token}, follow_redirects=False)
    assert login_ok.status_code == 303
    assert SESSION_COOKIE in login_ok.cookies

    # 5. Cookie allows state access
    cookie_res = client.get("/api/state", cookies={SESSION_COOKIE: login_ok.cookies[SESSION_COOKIE]})
    assert cookie_res.status_code == 200


def test_origin_defense_rejects_cross_site_mutations(test_client: TestClient):
    # State-changing POST from untrusted origin is rejected
    post_res = test_client.post(
        "/api/settings",
        json={"workspace_id": "demo_ws"},
        headers={"Origin": "https://malicious-site.com"},
    )
    assert post_res.status_code == 403
    assert "cross-origin" in post_res.text.lower()


def test_path_traversal_in_settings_rejected(test_client: TestClient):
    resp = test_client.post(
        "/api/settings",
        json={"workspace_id": "../../escape_workspace"},
    )
    assert resp.status_code == 400


def test_upload_extension_guard(test_client: TestClient):
    fake_sh = BytesIO(b"#!/bin/bash\nrm -rf /")
    resp = test_client.post(
        "/api/connect/upload",
        files={"files": ("malicious.sh", fake_sh, "application/x-sh")},
    )
    assert resp.status_code == 400
    assert "unsupported file type" in resp.text.lower()


def test_upload_size_limit(test_client: TestClient, monkeypatch):
    monkeypatch.setattr("src.api.routes.connect.MAX_UPLOAD_BYTES", 1024)  # 1 KB limit
    oversized = BytesIO(b"x" * 2048)
    resp = test_client.post(
        "/api/connect/upload",
        files={"files": ("large.csv", oversized, "text/csv")},
    )
    assert resp.status_code == 413
