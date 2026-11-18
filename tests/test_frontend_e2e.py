"""Frontend DOM contract & Playwright E2E tests (skipped automatically if Playwright is unavailable)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def test_frontend_html_contracts() -> None:
    index_html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    for required_id in (
        'id="i-chat"',
        'id="i-database"',
        'id="i-alert"',
        'id="rootCauseForm"',
        'id="whatIfDrawer"',
        'id="whatIfForm"',
        'id="whatIfMetricSelect"',
        'id="whatIfHorizonSelect"',
        'id="goalSeekBtn"',
        'id="scratchpadDialog"',
        'id="chatInputForm"',
    ):
        assert required_id in index_html, f"Missing {required_id} in frontend/index.html"

    studio_html = (FRONTEND_DIR / "thread_studio.html").read_text(encoding="utf-8")
    assert "/api/state" in studio_html
    assert "/api/system/state" not in studio_html
    assert "&#39;" in studio_html or "&#x27;" in studio_html


def test_frontend_app_js_contracts() -> None:
    app_js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    assert "let APP_STATE" in app_js
    assert "function renderChatView()" in app_js
    assert "function renderConnectView()" in app_js
    assert "function renderMeasuresView()" in app_js


@pytest.mark.e2e
def test_playwright_smoke_load(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    import threading
    import time
    import uvicorn
    from src.api import create_app
    from src.api.runtime import AppRuntime
    from src.security.auth import AuthConfig

    runtime = AppRuntime(base_dir=tmp_path)
    app = create_app(runtime=runtime, auth_config=AuthConfig(api_token=None))
    config = uvicorn.Config(app, host="127.0.0.1", port=18504, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(1.0)

    errors: list[str] = []
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.goto("http://127.0.0.1:18504/", wait_until="networkidle")
            assert page.title()
            assert not errors, f"Uncaught page errors: {errors}"
            browser.close()
    finally:
        server.should_exit = True
