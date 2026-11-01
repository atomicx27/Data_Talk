"""Conversational BI, Chart Generation, and AI Teaching routes."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends

from src.agents.query_agent import QueryAgent
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import (
    ChartBuildRequest,
    ChartThreadQueryRequest,
    ChatQueryRequest,
    ImprovePromptRequest,
    TeachRuleRequest,
)
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

    history = session.chat_history.setdefault(ws_id, [])

    resp, df = QueryAgent.answer_question(
        question=req.question.strip(),
        workspace=ws,
        okf_store=runtime.okf_store,
        rag_store=rag,
        connector=conn,
        llm_router=session.llm_router,
        what_if_overrides=overrides,
        conversation_history=history,
    )
    effective_q = resp.resolved_question or req.question
    show_chart = QueryAgent.should_render_chart(
        question=req.question,
        df=df,
        response_mode=req.response_mode,
        is_thread=False,
    )
    if not show_chart and req.response_mode == "text_only":
        rec_chart = "none"
        resp.recommended_chart_type = "none"
    elif resp.recommended_chart_type and resp.recommended_chart_type in ("table", "kpi", "none"):
        rec_chart = resp.recommended_chart_type
    else:
        rec_chart = ChartRecommender.recommend_chart_type(df, effective_q)
        resp.recommended_chart_type = rec_chart

    fig = None if rec_chart == "none" else ChartRecommender.build_figure(df, chart_type=rec_chart, title=effective_q)
    rows = df.head(250).to_dict(orient="records") if not df.empty else []
    columns = [str(c) for c in df.columns]

    chat_entry = {
        "id": f"msg_{len(history) + 1}",
        "response": resp.model_dump(mode="json"),
        "rows": rows,
        "columns": columns,
        "row_count": len(df),
        "chart_type": rec_chart,
        "plotly_spec": fig_to_json(fig) if fig else None,
        "response_mode": req.response_mode,
        "hide_chart": not show_chart,
        "thread": [],
    }
    history.append(chat_entry)
    return {
        "entry": chat_entry,
        "state": runtime.serialize_state(session),
    }


@router.post("/thread")
def run_chart_thread_query(
    req: ChartThreadQueryRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    rag = runtime.get_rag_store(ws_id)
    conn = runtime.get_connector(ws_id)
    overrides = session.what_if_overrides.get(ws_id, {})
    history = session.chat_history.setdefault(ws_id, [])

    target_entry: dict[str, Any] | None = None
    resolved_idx = req.entry_idx if req.entry_idx is not None else req.entry_index
    if resolved_idx is not None and 0 <= resolved_idx < len(history):
        target_entry = history[resolved_idx]
    elif req.entry_id:
        for item in history:
            if item.get("id") == req.entry_id:
                target_entry = item
                break

    parent_resp = (target_entry or {}).get("response", {})
    parent_question = req.parent_question or parent_resp.get("resolved_question") or parent_resp.get("user_question", "")
    parent_sql = req.parent_sql or parent_resp.get("executed_query", "")
    parent_summary = req.parent_summary or parent_resp.get("executive_summary", "")
    parent_rows = req.parent_rows if req.parent_rows else (target_entry or {}).get("rows", [])
    parent_columns = req.parent_columns if req.parent_columns else (target_entry or {}).get("columns", [])
    existing_thread = (target_entry or {}).get("thread", req.thread_history or [])

    thread_msg = QueryAgent.answer_chart_thread_question(
        question=req.question,
        parent_question=parent_question,
        parent_sql=parent_sql,
        parent_summary=parent_summary,
        parent_rows=parent_rows,
        parent_columns=parent_columns,
        thread_history=existing_thread,
        response_mode=req.response_mode,
        workspace=ws,
        okf_store=runtime.okf_store,
        rag_store=rag,
        connector=conn,
        llm_router=session.llm_router,
        what_if_overrides=overrides,
    )

    if target_entry is not None:
        thread_list = target_entry.setdefault("thread", [])
        thread_msg["id"] = f"thr_{len(thread_list) + 1}"
        thread_list.append(thread_msg)

    return {
        "turn": thread_msg,
        "thread_message": thread_msg,
        "entry": target_entry,
        "state": runtime.serialize_state(session),
    }


@router.post("/improve-prompt")
def improve_chat_prompt(
    req: ImprovePromptRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)
    history = session.chat_history.get(ws_id, [])
    return QueryAgent.improve_user_prompt(
        question=req.question,
        workspace=ws,
        connector=conn,
        conversation_history=history,
        llm_router=session.llm_router,
    )


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
