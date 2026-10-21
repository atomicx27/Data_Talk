"""Join inference and relationship governance API routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import CustomJoinRequest, JoinBatchUpdateRequest
from src.knowledge.okf_models import JoinCardinality, JoinStatus, RelationshipEdge

router = APIRouter(prefix="/api/joins", tags=["Joins"])


@router.post("/approve-all")
def approve_all_joins(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        for rel in ws.relationships:
            rel.status = JoinStatus.APPROVED
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.post("/update")
def update_joins(
    req: JoinBatchUpdateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        by_id = {r.id: r for r in ws.relationships}
        for item in req.relationships:
            if item.id in by_id:
                r = by_id[item.id]
                r.source_table = item.source_table
                r.source_column = item.source_column
                r.join_type = item.join_type
                r.target_table = item.target_table
                r.target_column = item.target_column
                r.cardinality = JoinCardinality(item.cardinality)
                r.status = JoinStatus(item.status)
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.post("/add")
def add_custom_join(
    req: CustomJoinRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        src_tbl = ws.get_table(req.source_table)
        tgt_tbl = ws.get_table(req.target_table)
        src_src_id = src_tbl.source_id if src_tbl else ""
        tgt_src_id = tgt_tbl.source_id if tgt_tbl else ""
        is_cross = bool(src_src_id and tgt_src_id and src_src_id != tgt_src_id)

        new_id = f"user_rel_{req.source_table}_{req.source_column}_to_{req.target_table}_{req.target_column}"
        ws.relationships.append(
            RelationshipEdge(
                id=new_id,
                source_table=req.source_table,
                source_column=req.source_column,
                target_table=req.target_table,
                target_column=req.target_column,
                join_type=req.join_type,
                cardinality=JoinCardinality(req.cardinality),
                confidence=1.0,
                discovery_method="cross_database_link" if is_cross else "user_defined",
                status=JoinStatus.APPROVED,
                overlap_ratio=1.0,
                is_cross_source=is_cross,
                source_source_id=src_src_id,
                target_source_id=tgt_src_id,
            )
        )
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)
