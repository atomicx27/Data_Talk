"""Conversational BI, Chart Generation, and AI Teaching routes."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends

from src.agents.query_agent import QueryAgent
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import ChartBuildRequest, ChatQueryRequest, TeachRuleRequest
from src.api.utils import fig_to_json
from src.knowledge.okf_models import BusinessRule
from src.viz.chart_recommender import ChartRecommender

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/query")
def run_chat_query(
    req: ChatQueryRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    rag = runtime.get_rag_store(ws_id)
    conn = runtime.get_connector(ws_id)

    overrides = {
        "Revenue Multiplier %": req.revenue_shock_pct / 100.0,
        "Cost Multiplier %": req.cost_shock_pct / 100.0,
    }
    session.what_if_overrides[ws_id] = overrides

    resp, df = QueryAgent.answer_question(
        question=req.question.strip(),
        workspace=ws,
        okf_store=runtime.okf_store,
        rag_store=rag,
        connector=conn,
        llm_router=session.llm_router,
        what_if_overrides=overrides,
    )
    rec_chart = ChartRecommender.recommend_chart_type(df, req.question)
    resp.recommended_chart_type = rec_chart

    fig = ChartRecommender.build_figure(df, chart_type=rec_chart, title=req.question)
    rows = df.head(250).to_dict(orient="records") if not df.empty else []
    columns = [str(c) for c in df.columns]

    history = session.chat_history.setdefault(ws_id, [])
    chat_entry = {
        "id": f"msg_{len(history) + 1}",
        "response": resp.model_dump(mode="json"),
        "rows": rows,
        "columns": columns,
        "row_count": len(df),
        "chart_type": rec_chart,
        "plotly_spec": fig_to_json(fig),
    }
    history.append(chat_entry)
    return {
        "entry": chat_entry,
        "state": runtime.serialize_state(session),
    }


@router.post("/chart")
def build_chart_for_rows(req: ChartBuildRequest) -> dict[str, Any]:
    df = pd.DataFrame(req.rows)
    fig = ChartRecommender.build_figure(df, chart_type=req.chart_type, title=req.title)
    return {
        "chart_type": req.chart_type,
        "plotly_spec": fig_to_json(fig),
    }


@router.post("/teach")
def teach_rule_from_chat(
    req: TeachRuleRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws_id)
        rid = f"chat_rule_{len(ws.business_rules) + 1}"
        sql_hint = "orders.ord_stat <> 'X'" if "ord_stat" in req.rule_text or "'X'" in req.rule_text else ""
        rule = BusinessRule(
            id=rid,
            term=f"Correction on: {req.question[:40]}",
            definition=req.rule_text.strip(),
            sql_filter_hint=sql_hint,
            source="chat_correction",
        )
        ws.business_rules.append(rule)
        rag.upsert("business_rules", rid, req.rule_text.strip(), metadata={"sql_filter_hint": sql_hint})
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.post("/clear")
def clear_chat_history(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    session.chat_history[session.active_workspace_id] = []
    return runtime.serialize_state(session)
