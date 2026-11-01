"""Verified Query Memory API routes for Data_Talk.

Enables saving, listing, testing, editing, and deleting human-verified canonical
queries ('Golden Queries') in Workspace memory.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import (
    VerifiedQueryCreateRequest,
    VerifiedQueryTestRequest,
    VerifiedQueryUpdateRequest,
)
from src.knowledge.okf_models import VerifiedQuery

router = APIRouter(prefix="/api/verified-queries", tags=["Verified Queries"])


@router.get("")
def list_verified_queries(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> list[dict[str, Any]]:
    """Return all verified golden queries saved in the active workspace."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    return [vq.model_dump(mode="json") for vq in ws.verified_queries]


@router.post("")
def create_verified_query(
    req: VerifiedQueryCreateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Save a verified query into workspace memory."""
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)
        conn = runtime.get_connector(ws.workspace_id)

        # Quick validation of query execution
        try:
            conn.execute_query(
                req.executed_query.strip(),
                max_rows=1,
                blocked_tables=ws.blocked_table_names(),
                blocked_columns=ws.blocked_column_names(),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query validation failed: {exc}",
            ) from exc

        query_id = f"vq_{len(ws.verified_queries) + 1}_{int(time.time())}"
        now_iso = datetime.now(timezone.utc).isoformat()
        vq = VerifiedQuery(
            id=query_id,
            question_pattern=req.question_pattern.strip(),
            executed_query=req.executed_query.strip(),
            dax_equivalent=req.dax_equivalent.strip() if req.dax_equivalent else "",
            query_dialect=req.query_dialect or ws.dialect,
            chart_type=req.chart_type or "bar",
            x_col=req.x_col,
            y_cols=req.y_cols or [],
            color_col=req.color_col,
            created_at=now_iso,
            updated_at=now_iso,
            execution_count=1,
        )

        ws.verified_queries.append(vq)
        runtime.okf_store.save(ws)

        # Index into RAGStore few_shot_queries for hybrid retrieval reinforcement
        try:
            rag.upsert(
                collection="few_shot_queries",
                doc_id=f"golden_{query_id}",
                content=f"User Question: {vq.question_pattern}\nVerified Query ({vq.query_dialect}):\n{vq.executed_query}\nDAX Equivalent:\n{vq.dax_equivalent}",
                metadata={"question": vq.question_pattern, "dialect": vq.query_dialect, "type": "verified_golden"},
            )
        except Exception:
            pass

    state = runtime.serialize_state(session)
    state["saved_query"] = vq.model_dump(mode="json")
    return state


@router.put("/{query_id}")
def update_verified_query(
    query_id: str,
    req: VerifiedQueryUpdateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Edit an existing verified query's question pattern, SQL, DAX, or chart type."""
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)
        conn = runtime.get_connector(ws.workspace_id)

        target = next((q for q in ws.verified_queries if q.id == query_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Verified query '{query_id}' not found.")

        # If executed_query changed, validate it
        if req.executed_query is not None and req.executed_query.strip() != target.executed_query:
            try:
                conn.execute_query(
                    req.executed_query.strip(),
                    max_rows=1,
                    blocked_tables=ws.blocked_table_names(),
                    blocked_columns=ws.blocked_column_names(),
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Query validation failed: {exc}",
                ) from exc
            target.executed_query = req.executed_query.strip()

        if req.question_pattern is not None:
            target.question_pattern = req.question_pattern.strip()
        if req.dax_equivalent is not None:
            target.dax_equivalent = req.dax_equivalent.strip()
        if req.chart_type is not None:
            target.chart_type = req.chart_type
        if req.query_dialect is not None:
            target.query_dialect = req.query_dialect
        if req.x_col is not None:
            target.x_col = req.x_col
        if req.y_cols is not None:
            target.y_cols = req.y_cols
        if req.color_col is not None:
            target.color_col = req.color_col

        target.updated_at = datetime.now(timezone.utc).isoformat()
        runtime.okf_store.save(ws)

        # Update in RAGStore
        try:
            rag.upsert(
                collection="few_shot_queries",
                doc_id=f"golden_{query_id}",
                content=f"User Question: {target.question_pattern}\nVerified Query ({target.query_dialect}):\n{target.executed_query}\nDAX Equivalent:\n{target.dax_equivalent}",
                metadata={"question": target.question_pattern, "dialect": target.query_dialect, "type": "verified_golden"},
            )
        except Exception:
            pass

    state = runtime.serialize_state(session)
    state["updated_query"] = target.model_dump(mode="json")
    return state


@router.delete("/{query_id}")
def delete_verified_query(
    query_id: str,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Delete a verified query from workspace memory."""
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)

        before_count = len(ws.verified_queries)
        ws.verified_queries = [q for q in ws.verified_queries if q.id != query_id]
        if len(ws.verified_queries) == before_count:
            raise HTTPException(status_code=404, detail=f"Verified query '{query_id}' not found.")

        runtime.okf_store.save(ws)

        # Remove from RAGStore
        try:
            rag.delete("few_shot_queries", f"golden_{query_id}")
        except Exception:
            pass

    return runtime.serialize_state(session)


@router.post("/{query_id}/test")
def test_verified_query(
    query_id: str,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Execute a saved verified query and return fresh live sample rows."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)

    target = next((q for q in ws.verified_queries if q.id == query_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Verified query '{query_id}' not found.")

    try:
        t0 = time.perf_counter()
        df = conn.execute_query(
            target.executed_query,
            max_rows=100,
            blocked_tables=ws.blocked_table_names(),
            blocked_columns=ws.blocked_column_names(),
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "success": True,
            "query_id": target.id,
            "elapsed_ms": elapsed_ms,
            "row_count": len(df),
            "columns": [str(c) for c in df.columns],
            "rows": df.head(50).to_dict(orient="records"),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "query_id": target.id,
            "elapsed_ms": 0.0,
            "row_count": 0,
            "columns": [],
            "rows": [],
            "error": str(exc),
        }


@router.post("/test-raw")
def test_raw_query(
    req: VerifiedQueryTestRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Test execute an arbitrary SQL query on the active database connector before saving."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)
    try:
        t0 = time.perf_counter()
        df = conn.execute_query(
            req.query.strip(),
            max_rows=100,
            blocked_tables=ws.blocked_table_names(),
            blocked_columns=ws.blocked_column_names(),
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "success": True,
            "elapsed_ms": elapsed_ms,
            "row_count": len(df),
            "columns": [str(c) for c in df.columns],
            "rows": df.head(50).to_dict(orient="records"),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "elapsed_ms": 0.0,
            "row_count": 0,
            "columns": [],
            "rows": [],
            "error": str(exc),
        }
