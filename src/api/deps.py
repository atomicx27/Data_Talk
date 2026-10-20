"""FastAPI dependencies for Data_Talk API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from src.api.runtime import AppRuntime, SessionState
from src.security.auth import Authenticator

_runtime: AppRuntime | None = None
_authenticator: Authenticator | None = None


def set_global_runtime(runtime: AppRuntime) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> AppRuntime:
    try:
        import server
        if getattr(server, "runtime", None) is not None:
            return server.runtime
    except (ImportError, AttributeError):
        pass
    if _runtime is not None:
        return _runtime
    raise RuntimeError("AppRuntime has not been initialized.")


def set_global_authenticator(authenticator: Authenticator) -> None:
    global _authenticator
    _authenticator = authenticator


def get_authenticator() -> Authenticator:
    try:
        import server
        if getattr(server, "authenticator", None) is not None:
            return server.authenticator
    except (ImportError, AttributeError):
        pass
    if _authenticator is not None:
        return _authenticator
    return Authenticator()


def get_session(
    request: Request,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    auth: Annotated[Authenticator, Depends(get_authenticator)],
) -> SessionState:
    """Resolve session state from session cookie or API token."""
    sid = auth.resolve_session_id(request)
    if not sid:
        if not auth.config.token_mode:
            sid = "default_session"
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Provide a valid dt_session cookie or Bearer token.",
            )
    return runtime.get_session(sid)
