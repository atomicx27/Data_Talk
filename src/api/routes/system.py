"""System and configuration API routes."""

from __future__ import annotations

import os
from typing import Annotated, Any

import duckdb
from fastapi import APIRouter, Depends, HTTPException

import src
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import SettingsUpdateRequest
from src.security.privacy_firewall import PrivacyMode
from src.security.validators import ValidationError, normalize_workspace_id, validate_http_url

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/health")
def health_check(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    ws = runtime.get_workspace(runtime.default_workspace_id)
    return {
        "status": "healthy",
        "product": "Data_Talk Enterprise Semantic BI",
        "version": getattr(src, "__version__", "2.4.0"),
        "duckdb_version": duckdb.__version__,
        "active_workspace": ws.workspace_id,
        "active_llm_model": os.environ.get("DATATALK_DEFAULT_MODEL", "groq/openai/gpt-oss-120b"),
        "privacy_mode": "strict_schema_only",
        "providers_ready": {
            "groq": bool(os.environ.get("GROQ_API_KEY")),
            "nvidia_nim": bool(os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")),
        },
    }


@router.get("/state")
def get_state(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    return runtime.serialize_state(session)


@router.post("/settings")
def update_settings(
    req: SettingsUpdateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    if req.workspace_id and req.workspace_id.strip():
        try:
            clean_id = normalize_workspace_id(req.workspace_id)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.active_workspace_id = clean_id
        runtime.get_workspace(clean_id)

    if req.privacy_mode:
        try:
            session.llm_router.firewall.privacy_mode = PrivacyMode(req.privacy_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid privacy mode: {req.privacy_mode}") from exc

    if req.llm_model:
        session.llm_router.model = req.llm_model

    if req.api_key is not None and req.api_key.strip():
        session.llm_router.api_key = req.api_key.strip()

    if req.api_base is not None:
        raw_base = req.api_base.strip()
        if raw_base:
            try:
                session.llm_router.api_base = validate_http_url(raw_base)
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            session.llm_router.api_base = None

    return runtime.serialize_state(session)
