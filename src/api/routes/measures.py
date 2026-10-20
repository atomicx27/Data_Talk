"""Semantic measure and calculated column studio routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from src.agents.measure_compiler import MeasureCompiler
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import CalculatedColumnCreateRequest, MeasureCreateRequest

router = APIRouter(prefix="/api/measures", tags=["Measures"])


@router.post("/create")
def create_measure(
    req: MeasureCreateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)
        conn = runtime.get_connector(ws.workspace_id)
        meas = MeasureCompiler.create_and_verify_measure(
            workspace=ws,
            rag_store=rag,
            connector=conn,
            name=req.name.strip(),
            table_name=req.table_name.strip(),
            user_formula=req.user_formula.strip(),
            input_type=req.input_type,
            description=req.description.strip(),
            llm_router=session.llm_router,
        )
        runtime.okf_store.save(ws)
    state = runtime.serialize_state(session)
    state["latest_measure"] = meas.model_dump(mode="json")
    return state


@router.post("/calculated-column")
def create_calculated_column(
    req: CalculatedColumnCreateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)
        conn = runtime.get_connector(ws.workspace_id)
        cc = MeasureCompiler.create_and_verify_calculated_column(
            workspace=ws,
            rag_store=rag,
            connector=conn,
            table_name=req.table_name.strip(),
            column_name=req.column_name.strip(),
            user_formula=req.user_formula.strip(),
            input_type=req.input_type,
            llm_router=session.llm_router,
        )
        runtime.okf_store.save(ws)
    state = runtime.serialize_state(session)
    state["latest_calculated_column"] = cc.model_dump(mode="json")
    return state
