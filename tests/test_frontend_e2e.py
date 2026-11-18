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
        'id="i-pie"',
        'id="i-schema"',
        'id="i-alert-circle"',
        'id="rootCauseForm"',
        'id="whatIfDrawer"',
        'id="whatIfForm"',
        'id="whatIfMetricSelect"',
        'id="whatIfHorizonSelect"',
        'id="goalSeekBtn"',
        'id="scratchpadDialog"',
        'id="scratchpadResultBox"',
        'id="chatInputForm"',
        'id="erModeSwitcher"',
        'id="lineage3dPipeline"',
        'id="btnToggleTopology"',
        'id="lineageTopologyPanel"',
        'id="clarificationInterruptDialog"',
        'id="clarificationInterruptForm"',
        'id="globalDiagnosticMount"',
        'id="toggleHitlInterruptBtn"',
        'id="toggleSupervisorBtn"',
        '/static/modules/dt3d_boot.js',
    ):
        assert required_id in index_html, f"Missing {required_id} in frontend/index.html"

    studio_html = (FRONTEND_DIR / "thread_studio.html").read_text(encoding="utf-8")
    assert "/api/state" in studio_html
    assert "/api/system/state" not in studio_html
    assert "&#39;" in studio_html or "&#x27;" in studio_html


def test_frontend_app_js_contracts() -> None:
    app_js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    for required_symbol in (
        "let APP_STATE",
        "window.APP_STATE = APP_STATE",
        "window.renderApp = renderAllViews",
        "function renderChatView()",
        "function renderConnectView()",
        "function renderMeasuresView()",
        "/api/chat/query/resume",
        "scatter3d",
        "surface3d",
    ):
        assert required_symbol in app_js, f"Missing {required_symbol} in frontend/app.js"

    modules_dir = FRONTEND_DIR / "modules"
    for mod_name in (
        "dt3d_boot.js",
        "spatial_tilt.js",
        "error_boundary.js",
        "er3d_engine.js",
        "langgraph_3d_trace.js",
        "plotly_3d_surfaces.js",
    ):
        mod_path = modules_dir / mod_name
        assert mod_path.exists(), f"Missing ES6 module {mod_path}"

    styles_css = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion: reduce" in styles_css
    assert ".dt-diag-card" in styles_css
    assert ".er3d-iso-plane" in styles_css

    er3d_js = (modules_dir / "er3d_engine.js").read_text(encoding="utf-8")
    assert "export function renderER" in er3d_js
    assert "export { renderER as render }" in er3d_js

    import shutil
    import subprocess

    node_exe = shutil.which("node")
    if node_exe:
        for js_file in [FRONTEND_DIR / "app.js", *sorted(modules_dir.glob("*.js"))]:
            proc = subprocess.run([node_exe, "--check", str(js_file)], capture_output=True, text=True)
            assert proc.returncode == 0, f"Syntax error in {js_file.name}: {proc.stderr}"


def test_frontend_modules_and_topology_served(test_client) -> None:
    for url_path in (
        "/",
        "/static/modules/dt3d_boot.js",
        "/static/modules/spatial_tilt.js",
        "/static/modules/error_boundary.js",
        "/static/modules/er3d_engine.js",
        "/static/modules/langgraph_3d_trace.js",
        "/static/modules/plotly_3d_surfaces.js",
        "/api/chat/graph-topology",
        "/api/state",
    ):
        r = test_client.get(url_path)
        assert r.status_code == 200, f"Expected 200 on {url_path}, got {r.status_code}"


@pytest.mark.e2e
def test_playwright_smoke_load(tmp_path: Path, monkeypatch) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    import threading
    import time
    import uvicorn
    import server
    from src.api import create_app
    from src.api.runtime import AppRuntime
    from src.security.auth import AuthConfig, Authenticator

    runtime = AppRuntime(base_dir=tmp_path)
    auth_cfg = AuthConfig(api_token=None)
    monkeypatch.setattr(server, "runtime", runtime)
    monkeypatch.setattr(server, "authenticator", Authenticator(auth_cfg))
    app = create_app(runtime=runtime, auth_config=auth_cfg)
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
