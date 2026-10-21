"""Data source connection and onboarding routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.agents.interviewer_agent import InterviewerAgent
from src.agents.measure_compiler import MeasureCompiler
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import PowerBIConnectRequest, RemoveSourceRequest, SQLConnectRequest
from src.connectors.duckdb_connector import DuckDBConnector
from src.connectors.federated_connector import FederatedConnector
from src.connectors.powerbi_connector import PowerBIConnector
from src.connectors.profiler import StatisticalProfiler
from src.connectors.sql_connector import SQLConnector
from src.knowledge.join_inference import JoinInferenceEngine
from src.knowledge.okf_models import DataSourceConfig
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

        demo_src = DataSourceConfig(
            id="enterprise_demo",
            name="Enterprise E-Commerce Demo",
            source_type="duckdb",
            dialect="duckdb",
            table_names=[t.name for t in ws.tables],
        )
        ws.data_sources = [demo_src]
        for tbl in ws.tables:
            tbl.source_id = "enterprise_demo"
            tbl.source_name = "Enterprise Demo"
        for q in ws.interview_questions:
            q.source_id = "enterprise_demo"

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

        upload_src = DataSourceConfig(
            id="uploaded_files",
            name="Uploaded Datasets",
            source_type="duckdb",
            dialect="duckdb",
            table_names=[t.name for t in ws.tables if t.source_id in ("uploaded_files", "default", "")],
        )
        ws.data_sources = [s for s in ws.data_sources if s.id != "uploaded_files"] + [upload_src]
        for tbl in ws.tables:
            if not tbl.source_id or tbl.source_id == "default":
                tbl.source_id = "uploaded_files"
                tbl.source_name = "Uploaded Files"
        for q in ws.interview_questions:
            if not q.source_id or q.source_id == "default":
                q.source_id = "uploaded_files"
        runtime.okf_store.save(ws)

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
            ws_dir = runtime.okf_store.workspace_dir(ws.workspace_id)
            source_id = (req.source_id or req.credential_name).strip()
            source_name = (req.source_name or source_id.replace("_", " ").title()).strip()

            runtime.vault.set_secret(req.credential_name, req.connection_url.strip())

            # Retrieve existing connector or create FederatedConnector
            current_conn = runtime.get_connector(ws.workspace_id)
            if isinstance(current_conn, FederatedConnector):
                fed_conn = current_conn
            else:
                fed_conn = FederatedConnector(ws_dir)

            sql_conn = fed_conn.register_sql_connector(
                source_id=source_id,
                connection_url=req.connection_url.strip(),
                dialect_hint=req.dialect,
            )

            # Profile source tables directly on remote database push-down
            blocked_t = ws.blocked_table_names()
            blocked_c = ws.blocked_column_names()
            raw_tables, explicit_rels, imported_measures = sql_conn.extract_schema(
                blocked_tables=blocked_t,
                blocked_columns=blocked_c,
            )
            profiled_tables = StatisticalProfiler.profile_tables(
                sql_conn,
                raw_tables,
                blocked_tables=blocked_t,
                blocked_columns=blocked_c,
            )
            for tbl in profiled_tables:
                tbl.source_id = source_id
                tbl.source_name = source_name

            # Mirror/attach into DuckDB warehouse
            fed_conn.sync_sql_source_to_duckdb(source_id, [t.name for t in profiled_tables])

            # Update workspace data_sources
            src_cfg = DataSourceConfig(
                id=source_id,
                name=source_name,
                source_type="sql",
                dialect=sql_conn.dialect,
                credential_ref=req.credential_name,
                table_names=[t.name for t in profiled_tables],
            )

            if req.attach_mode:
                existing_sources = [s for s in ws.data_sources if s.id != source_id]
                ws.data_sources = existing_sources + [src_cfg]
                # Keep other sources' tables, update this source's tables
                other_tables = [t for t in ws.tables if t.source_id != source_id]
                ws.tables = other_tables + profiled_tables
            else:
                ws.data_sources = [src_cfg]
                ws.tables = profiled_tables

            # Infer relationships across all active tables (including cross-database!)
            ws.relationships = JoinInferenceEngine.infer_relationships(
                ws.tables,
                existing_relationships=ws.relationships + explicit_rels,
                connector=fed_conn,
            )
            if imported_measures:
                existing_ids = {m.id for m in ws.measures}
                for im in imported_measures:
                    if im.id not in existing_ids:
                        ws.measures.append(im)

            runtime.okf_store.classify_table_roles(ws)
            rag = runtime.get_rag_store(ws.workspace_id)
            InterviewerAgent.sync_schema_to_rag(ws, rag)
            InterviewerAgent.generate_prioritized_questions(ws, llm_router=session.llm_router)

            ws.source_type = "federated"
            ws.dialect = "duckdb"
            runtime.set_connector(ws.workspace_id, fed_conn)
            runtime.okf_store.save(ws)

            return runtime.serialize_state(session)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"SQL Connection Failed: {exc}") from exc


@router.post("/remove-source")
def remove_source(
    req: RemoveSourceRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        source_id = req.source_id.strip()
        matched = ws.get_source(source_id)
        if not matched:
            raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")

        # Clean up connector
        conn = runtime.get_connector(ws.workspace_id)
        if isinstance(conn, FederatedConnector):
            conn.remove_source(source_id, matched.table_names)

        # Remove source and its tables, rels, questions
        ws.data_sources = [s for s in ws.data_sources if s.id != source_id]
        ws.tables = [t for t in ws.tables if t.source_id != source_id]
        ws.relationships = [
            r
            for r in ws.relationships
            if r.source_source_id != source_id
            and r.target_source_id != source_id
            and ws.get_table(r.source_table) is not None
            and ws.get_table(r.target_table) is not None
        ]
        ws.interview_questions = [q for q in ws.interview_questions if q.source_id != source_id]

        runtime.okf_store.classify_table_roles(ws)
        rag = runtime.get_rag_store(ws.workspace_id)
        InterviewerAgent.sync_schema_to_rag(ws, rag)
        runtime.okf_store.save(ws)

        return runtime.serialize_state(session)


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
