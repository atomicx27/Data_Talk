"""FastAPI Application factory for Data_Talk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import src
from src.api.deps import set_global_authenticator, set_global_runtime
from src.api.routes import chat, connect, dashboard, export, interview, joins, measures, schema, system
from src.api.runtime import AppRuntime
from src.security.auth import (
    LOGIN_PAGE_HTML,
    SESSION_COOKIE,
    AuthConfig,
    Authenticator,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def create_app(
    runtime: AppRuntime | None = None,
    auth_config: AuthConfig | None = None,
) -> FastAPI:
    """Create and configure a production FastAPI application instance."""
    app_runtime = runtime or AppRuntime()
    authenticator = Authenticator(auth_config)

    set_global_runtime(app_runtime)
    set_global_authenticator(authenticator)

    app = FastAPI(
        title="Data_Talk Enterprise BI API",
        description="Two-Tier OKF + Hybrid RAG Conversational Business Intelligence Platform",
        version=getattr(src, "__version__", "2.4.0"),
    )

    # Attach to app state for introspectability
    app.state.runtime = app_runtime
    app.state.authenticator = authenticator

    # CORS configuration with strict origins by default
    app.add_middleware(
        CORSMiddleware,
        allow_origins=authenticator.config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )

    # Security & Session Middleware
    @app.middleware("http")
    async def security_and_session_middleware(request: Request, call_next: Any) -> Response:
        path = request.url.path

        # Origin check for mutating requests (CSRF / drive-by defense)
        if not authenticator.origin_allowed(request):
            return JSONResponse(
                {"detail": "Forbidden: cross-origin state-changing request rejected."},
                status_code=status.HTTP_403_FORBIDDEN,
            )

        # In token mode, protect all /api/* routes except /api/health
        if authenticator.config.token_mode:
            if path.startswith("/api/") and path != "/api/health":
                sid = authenticator.resolve_session_id(request)
                if not sid:
                    return JSONResponse(
                        {"detail": "Unauthorized: valid session cookie or Bearer token required."},
                        status_code=status.HTTP_401_UNAUTHORIZED,
                    )

        response: Response = await call_next(request)

        # Inject session cookie if user visits GET / or /api/* in open mode and has no cookie yet
        existing_sid = authenticator.session_id_from_cookie(request.cookies.get(SESSION_COOKIE))
        if not existing_sid and request.method == "GET" and (path == "/" or path.startswith("/api/")):
            if not authenticator.config.token_mode:
                cookie_val = authenticator.new_session_cookie()
                response.set_cookie(
                    key=SESSION_COOKIE,
                    value=cookie_val,
                    **authenticator.cookie_kwargs(request),
                )

        # Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        if authenticator.config.frame_ancestors:
            response.headers["Content-Security-Policy"] = f"frame-ancestors {authenticator.config.frame_ancestors};"

        # Cache headers for static / root
        if path.startswith("/static") or path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response

    # Mount API routers
    app.include_router(system.router)
    app.include_router(connect.router)
    app.include_router(schema.router)
    app.include_router(joins.router)
    app.include_router(interview.router)
    app.include_router(measures.router)
    app.include_router(chat.router)
    app.include_router(dashboard.router)
    app.include_router(export.router)

    # Auth routes
    @app.post("/auth/login")
    def login_with_token(request: Request, token: str = Form(...)) -> Response:
        if not authenticator.check_token(token):
            error_html = LOGIN_PAGE_HTML.replace(
                "__ERROR__", '<div class="err">Invalid access token. Please check your token and retry.</div>'
            )
            return HTMLResponse(error_html, status_code=status.HTTP_401_UNAUTHORIZED)
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=authenticator.new_session_cookie(),
            **authenticator.cookie_kwargs(request),
        )
        return response

    @app.post("/auth/logout")
    def logout(request: Request) -> Response:
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(key=SESSION_COOKIE, path="/")
        return response

    # Static assets
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    # Root route
    @app.get("/")
    def serve_index(request: Request) -> Response:
        if authenticator.config.token_mode:
            sid = authenticator.resolve_session_id(request)
            if not sid:
                return HTMLResponse(LOGIN_PAGE_HTML.replace("__ERROR__", ""))
        index_file = FRONTEND_DIR / "index.html"
        if not index_file.exists():
            return JSONResponse({"status": "healthy", "message": "Frontend not found"})
        return FileResponse(str(index_file))

    # Dedicated Full-Page Chart Thread Studio route (opens in new tab/window)
    @app.get("/thread-studio")
    def serve_thread_studio(request: Request, idx: int = 0) -> Response:
        if authenticator.config.token_mode:
            sid = authenticator.resolve_session_id(request)
            if not sid:
                return HTMLResponse(LOGIN_PAGE_HTML.replace("__ERROR__", ""))
        studio_file = FRONTEND_DIR / "thread_studio.html"
        if not studio_file.exists():
            return RedirectResponse(url=f"/?thread_idx={idx}")
        return FileResponse(str(studio_file))

    return app
