"""Schema management and column configuration API routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from src.agents.interviewer_agent import InterviewerAgent
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import ColumnConfigUpdateRequest

router = APIRouter(prefix="/api/schema", tags=["Schema"])


@router.post("/column-config")
def update_column_config(
    req: ColumnConfigUpdateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        tbl = ws.get_table(req.table_name)
        if not tbl:
            for t in ws.tables:
                if t.name.lower() == req.table_name.lower():
                    tbl = t
                    break
        if not tbl:
            raise HTTPException(status_code=404, detail=f"Table '{req.table_name}' not found.")

        col = tbl.get_column(req.column_name)
        if not col:
            raise HTTPException(status_code=404, detail=f"Column '{req.column_name}' not found.")

        if req.is_blocked is not None:
            col.is_blocked = req.is_blocked
        if req.description is not None:
            col.description = req.description.strip()

        runtime.okf_store.save(ws)
        InterviewerAgent.sync_schema_to_rag(ws, runtime.get_rag_store(ws.workspace_id))
    return runtime.serialize_state(session)
