"""Data source connection and onboarding routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.agents.measure_compiler import MeasureCompiler
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import PowerBIConnectRequest, SQLConnectRequest
from src.connectors.duckdb_connector import DuckDBConnector
from src.connectors.powerbi_connector import PowerBIConnector
from src.connectors.sql_connector import SQLConnector
from src.security.validators import ValidationError, safe_upload_name

router = APIRouter(prefix="/api/connect", tags=["Connectors"])

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024


@router.post("/demo")
def connect_demo_dataset(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        ws_dir = runtime.okf_store.workspace_dir(ws.workspace_id)
        duck_conn = DuckDBConnector(ws_dir)
        duck_conn.load_builtin_demo_dataset()
        runtime.set_connector(ws.workspace_id, duck_conn)

        ws.source_type = "duckdb"
        ws.dialect = "duckdb"
        runtime.run_onboarding(ws, duck_conn, llm_router=session.llm_router)

        # Pre-verify 2 starter DAX measures so the Measure Studio is immediately populated
        rag = runtime.get_rag_store(ws.workspace_id)
        for tmpl in MeasureCompiler.STARTER_MEASURE_TEMPLATES[:2]:
            MeasureCompiler.create_and_verify_measure(
                workspace=ws,
                rag_store=rag,
                connector=duck_conn,
                name=tmpl["name"],
                table_name=tmpl["table_name"],
                user_formula=tmpl["dax_expression"],
                input_type="dax",
                description=tmpl["description"],
                llm_router=session.llm_router,
            )
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.post("/upload")
async def connect_uploaded_files(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        ws_dir = runtime.okf_store.workspace_dir(ws.workspace_id)
        upload_dir = ws_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        duck_conn = DuckDBConnector(ws_dir)
        for uf in files:
            raw_fname = uf.filename or "uploaded.csv"
            try:
                fname = safe_upload_name(raw_fname)
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            dest = upload_dir / fname
            total_size = 0
            try:
                with dest.open("wb") as buffer:
                    while chunk := await uf.read(1024 * 64):
                        total_size += len(chunk)
                        if total_size > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"File exceeds maximum upload size of {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
                            )
                        buffer.write(chunk)
            except Exception:
                if dest.exists():
                    dest.unlink(missing_ok=True)
                raise

            suffix = Path(fname).suffix.lower()
            if suffix in (".pbix", ".pbit"):
                pbi_conn = PowerBIConnector(mode="powerbi_pbix", workspace_dir=ws_dir, pbix_path=dest)
                runtime.set_connector(ws.workspace_id, pbi_conn)
                ws.source_type = "powerbi_pbix"
                ws.dialect = "duckdb"
                runtime.run_onboarding(ws, pbi_conn, llm_router=session.llm_router)
                return runtime.serialize_state(session)
            else:
                duck_conn.ingest_file(dest, original_filename=fname)

        runtime.set_connector(ws.workspace_id, duck_conn)
        ws.source_type = "duckdb"
        ws.dialect = "duckdb"
        runtime.run_onboarding(ws, duck_conn, llm_router=session.llm_router)
    return runtime.serialize_state(session)


@router.post("/sql")
def connect_sql_database(
    req: SQLConnectRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    if not req.connection_url.strip():
        raise HTTPException(status_code=400, detail="Connection URL is required.")
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        try:
            ws = runtime.get_workspace(ws_id)
            runtime.vault.set_secret(req.credential_name, req.connection_url.strip())
            sql_conn = SQLConnector(req.connection_url.strip(), dialect_hint=req.dialect)
            runtime.set_connector(ws.workspace_id, sql_conn)
            ws.source_type = "sql"
            ws.dialect = sql_conn.dialect
            ws.credential_ref = req.credential_name
            runtime.run_onboarding(ws, sql_conn, llm_router=session.llm_router)
            return runtime.serialize_state(session)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"SQL Connection Failed: {exc}") from exc


@router.post("/powerbi")
def connect_powerbi(
    req: PowerBIConnectRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        try:
            if req.mode == "powerbi_desktop":
                pbi_conn = PowerBIConnector(mode="powerbi_desktop", localhost_port=req.localhost_port)
                ws.source_type = "powerbi_desktop"
                ws.dialect = "dax"
            else:
                runtime.vault.set_secret("pbi_cloud_token", req.access_token.strip())
                pbi_conn = PowerBIConnector(
                    mode="powerbi_cloud",
                    dataset_id=req.dataset_id.strip(),
                    access_token=req.access_token.strip(),
                )
                ws.source_type = "powerbi_cloud"
                ws.dialect = "dax"
                ws.credential_ref = "pbi_cloud_token"

            runtime.set_connector(ws.workspace_id, pbi_conn)
            runtime.run_onboarding(ws, pbi_conn, llm_router=session.llm_router)
            return runtime.serialize_state(session)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
