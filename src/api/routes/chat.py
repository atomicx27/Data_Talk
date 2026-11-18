"""Conversational BI, LangGraph Streaming/HITL, Chart Generation, and AI Teaching routes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.agents.compilers.text_matching import normalize_what_if_overrides
from src.agents.graph.query_graph import (
    get_query_graph_topology,
    resume_query_graph,
    stream_query_graph,
)
from src.agents.graph.supervisor_graph import (
    get_supervisor_graph_topology,
    run_supervisor_graph,
)
from src.agents.query_agent import QueryAgent, QueryResponse
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import (
    ChartBuildRequest,
    ChartThreadQueryRequest,
    ChatQueryRequest,
    ChatQueryResumeRequest,
    ImprovePromptRequest,
    TeachRuleRequest,
)
from src.api.utils import fig_to_json
from src.knowledge.okf_models import BusinessRule
from src.viz.chart_recommender import ChartRecommender

router = APIRouter(prefix="/api/chat", tags=["Chat"])


def _normalize_chat_what_if(revenue_shock_pct: float, cost_shock_pct: float) -> dict[str, float]:
    """Build non-zero normalized decimal What-If overrides from percentage slider inputs."""
    raw = {
        "Revenue Multiplier %": float(revenue_shock_pct or 0.0) / 100.0,
        "Cost Multiplier %": float(cost_shock_pct or 0.0) / 100.0,
    }
    return normalize_what_if_overrides(raw)


def _format_sse(event_name: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event_name}\ndata: {payload}\n\n"


def _build_chat_entry_payload(
    *,
    question: str,
    resp: QueryResponse,
    df: pd.DataFrame,
    response_mode: str,
    history: list[dict[str, Any]],
    thread_id: str | None = None,
    interrupted: bool = False,
    interrupt_payload: dict[str, Any] | None = None,
    resumed_from_thread_id: str | None = None,
) -> dict[str, Any]:
    """Assemble standardized chat entry dictionary with optional Plotly chart spec."""
    effective_q = resp.resolved_question or question
    show_chart = QueryAgent.should_render_chart(
        question=question,
        df=df,
        response_mode=response_mode,
        is_thread=False,
    )
    if interrupted or (not show_chart and response_mode == "text_only"):
        rec_chart = "none"
        resp.recommended_chart_type = "none"
    elif resp.recommended_chart_type and resp.recommended_chart_type in ("table", "kpi", "none"):
        rec_chart = resp.recommended_chart_type
    else:
        rec_chart = ChartRecommender.recommend_chart_type(df, effective_q)
        resp.recommended_chart_type = rec_chart

    fig = (
        None
        if (rec_chart == "none" or df.empty)
        else ChartRecommender.build_figure(df, chart_type=rec_chart, title=effective_q)
    )
    rows = df.head(250).to_dict(orient="records") if not df.empty else []
    columns = [str(c) for c in df.columns]

    chat_entry: dict[str, Any] = {
        "id": f"msg_{len(history) + 1}",
        "response": resp.model_dump(mode="json"),
        "rows": rows,
        "columns": columns,
        "row_count": len(df),
        "chart_type": rec_chart,
        "plotly_spec": fig_to_json(fig) if fig else None,
        "response_mode": response_mode,
        "hide_chart": not show_chart,
        "thread": [],
        "thread_id": thread_id,
        "interrupted": interrupted,
        "interrupt_payload": interrupt_payload,
    }
    if resumed_from_thread_id:
        chat_entry["resumed_from_thread_id"] = resumed_from_thread_id
    return chat_entry


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

    overrides = _normalize_chat_what_if(req.revenue_shock_pct, req.cost_shock_pct)
    session.what_if_overrides[ws_id] = overrides

    history = session.chat_history.setdefault(ws_id, [])
    tid = runtime.allocate_thread_id(session, ws_id, scope="chat", explicit_thread_id=req.thread_id)

    if req.use_supervisor:
        sup_res = run_supervisor_graph(
            question=req.question.strip(),
            workspace=ws,
            okf_store=runtime.okf_store,
            rag_store=rag,
            connector=conn,
            llm_router=session.llm_router,
            what_if_overrides=overrides or None,
            conversation_history=history,
            force_recompile=req.force_recompile,
            allow_interrupts=req.allow_interrupts,
            force_heuristic=req.force_heuristic,
            thread_id=tid,
            checkpointer=runtime.checkpointer,
        )
        payload = dict(sup_res.get("response_dict") or {})
        payload.setdefault("user_question", req.question.strip())
        payload.setdefault("success", False)
        payload["interrupted"] = bool(sup_res.get("interrupted", False))
        payload["interrupt_payload"] = sup_res.get("interrupt_payload")
        payload["thread_id"] = sup_res.get("thread_id", tid)
        resp = QueryResponse.model_validate(payload)
        df = sup_res.get("df", pd.DataFrame())
    else:
        resp, df = QueryAgent.answer_question(
            question=req.question.strip(),
            workspace=ws,
            okf_store=runtime.okf_store,
            rag_store=rag,
            connector=conn,
            llm_router=session.llm_router,
            what_if_overrides=overrides or None,
            conversation_history=history,
            force_recompile=req.force_recompile,
            allow_interrupts=req.allow_interrupts,
            force_heuristic=req.force_heuristic,
            thread_id=tid,
            checkpointer=runtime.checkpointer,
        )

    int_payload = (
        resp.interrupt_payload.model_dump(mode="json")
        if resp.interrupt_payload is not None
        else None
    )
    chat_entry = _build_chat_entry_payload(
        question=req.question.strip(),
        resp=resp,
        df=df,
        response_mode=req.response_mode,
        history=history,
        thread_id=resp.thread_id or tid,
        interrupted=bool(resp.interrupted),
        interrupt_payload=int_payload,
    )
    if resp.interrupted:
        session.pending_interrupts[ws_id] = {
            "thread_id": chat_entry["thread_id"],
            "question": req.question.strip(),
            "interrupt_payload": int_payload,
        }
    else:
        session.pending_interrupts.pop(ws_id, None)
        history.append(chat_entry)

    return {
        "entry": chat_entry,
        "state": runtime.serialize_state(session),
    }


@router.post("/query/stream")
def stream_chat_query(
    req: ChatQueryRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> StreamingResponse:
    """Stream real-time LangGraph node stages, telemetry pills, HITL interrupts, and completion via SSE."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    rag = runtime.get_rag_store(ws_id)
    conn = runtime.get_connector(ws_id)

    overrides = _normalize_chat_what_if(req.revenue_shock_pct, req.cost_shock_pct)
    session.what_if_overrides[ws_id] = overrides

    history = session.chat_history.setdefault(ws_id, [])
    tid = runtime.allocate_thread_id(session, ws_id, scope="chat", explicit_thread_id=req.thread_id)

    def _event_stream() -> Iterator[str]:
        if req.use_supervisor:
            yield _format_sse("stage", {"type": "stage", "stage": "supervisor", "label": "Classifying analytical intent..."})
            sup_res = run_supervisor_graph(
                question=req.question.strip(),
                workspace=ws,
                okf_store=runtime.okf_store,
                rag_store=rag,
                connector=conn,
                llm_router=session.llm_router,
                what_if_overrides=overrides or None,
                conversation_history=history,
                force_recompile=req.force_recompile,
                allow_interrupts=req.allow_interrupts,
                force_heuristic=req.force_heuristic,
                thread_id=tid,
                checkpointer=runtime.checkpointer,
            )
            for rec in sup_res.get("node_telemetry") or []:
                yield _format_sse("node_update", rec)
            finalized = sup_res
        else:
            finalized = None
            for evt in stream_query_graph(
                question=req.question.strip(),
                workspace=ws,
                okf_store=runtime.okf_store,
                rag_store=rag,
                connector=conn,
                llm_router=session.llm_router,
                what_if_overrides=overrides or None,
                conversation_history=history,
                force_recompile=req.force_recompile,
                allow_interrupts=req.allow_interrupts,
                force_heuristic=req.force_heuristic,
                thread_id=tid,
                checkpointer=runtime.checkpointer,
            ):
                evt_name = evt["event"]
                evt_data = evt["data"]
                if evt_name == "interrupt":
                    payload = dict(evt_data.get("response_dict") or {})
                    payload.setdefault("user_question", req.question.strip())
                    payload.setdefault("success", False)
                    payload["interrupted"] = True
                    payload["interrupt_payload"] = evt_data.get("interrupt_payload")
                    payload["thread_id"] = evt_data.get("thread_id", tid)
                    session.pending_interrupts[ws_id] = {
                        "thread_id": payload["thread_id"],
                        "question": req.question.strip(),
                        "interrupt_payload": payload["interrupt_payload"],
                    }
                    yield _format_sse(
                        "interrupt",
                        {
                            "thread_id": payload["thread_id"],
                            "interrupt_payload": payload["interrupt_payload"],
                            "state": runtime.serialize_state(session),
                        },
                    )
                    return
                elif evt_name == "complete":
                    finalized = evt_data
                else:
                    yield _format_sse(evt_name, evt_data)

        if not finalized:
            return

        if finalized.get("interrupted"):
            int_payload = finalized.get("interrupt_payload")
            session.pending_interrupts[ws_id] = {
                "thread_id": finalized.get("thread_id", tid),
                "question": req.question.strip(),
                "interrupt_payload": int_payload,
            }
            yield _format_sse(
                "interrupt",
                {
                    "thread_id": finalized.get("thread_id", tid),
                    "interrupt_payload": int_payload,
                    "state": runtime.serialize_state(session),
                },
            )
            return

        payload = dict(finalized.get("response_dict") or {})
        payload.setdefault("user_question", req.question.strip())
        payload.setdefault("success", False)
        payload["interrupted"] = False
        payload["interrupt_payload"] = None
        payload["thread_id"] = finalized.get("thread_id", tid)
        resp = QueryResponse.model_validate(payload)
        df = finalized.get("df", pd.DataFrame())
        chat_entry = _build_chat_entry_payload(
            question=req.question.strip(),
            resp=resp,
            df=df,
            response_mode=req.response_mode,
            history=history,
            thread_id=payload["thread_id"],
        )
        session.pending_interrupts.pop(ws_id, None)
        history.append(chat_entry)
        yield _format_sse(
            "complete",
            {
                "entry": chat_entry,
                "state": runtime.serialize_state(session),
            },
        )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/resume")
def resume_chat_query(
    req: ChatQueryResumeRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Resume a paused LangGraph blocking_modal clarification interrupt via Command(resume=...)."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    rag = runtime.get_rag_store(ws_id)
    conn = runtime.get_connector(ws_id)
    history = session.chat_history.setdefault(ws_id, [])

    try:
        resumed = resume_query_graph(
            thread_id=req.thread_id,
            resume_payload={
                "answer": req.answer,
                "save_rule": req.save_rule,
                "question_id": req.question_id,
            },
            workspace=ws,
            okf_store=runtime.okf_store,
            rag_store=rag,
            connector=conn,
            llm_router=session.llm_router,
            conversation_history=history,
            checkpointer=runtime.checkpointer,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload = dict(resumed.get("response_dict") or {})
    orig_q = str(payload.get("user_question") or session.pending_interrupts.get(ws_id, {}).get("question") or "")
    payload.setdefault("user_question", orig_q)
    payload.setdefault("success", False)
    payload["interrupted"] = bool(resumed.get("interrupted", False))
    payload["interrupt_payload"] = resumed.get("interrupt_payload")
    payload["thread_id"] = req.thread_id
    resp = QueryResponse.model_validate(payload)
    df = resumed.get("df", pd.DataFrame())

    chat_entry = _build_chat_entry_payload(
        question=orig_q,
        resp=resp,
        df=df,
        response_mode="auto",
        history=history,
        thread_id=req.thread_id,
        interrupted=bool(resp.interrupted),
        interrupt_payload=payload["interrupt_payload"],
        resumed_from_thread_id=req.thread_id,
    )
    session.pending_interrupts.pop(ws_id, None)
    history.append(chat_entry)

    return {
        "entry": chat_entry,
        "state": runtime.serialize_state(session),
    }


@router.get("/graph-topology")
def get_chat_graph_topology(
    graph_type: Literal["query", "supervisor"] = Query(default="query"),
) -> dict[str, Any]:
    """Introspect compiled LangGraph node topology and Mermaid definition."""
    if graph_type == "supervisor":
        return get_supervisor_graph_topology()
    return get_query_graph_topology()


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
    overrides = normalize_what_if_overrides(session.what_if_overrides.get(ws_id, {}))
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
        what_if_overrides=overrides or None,
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
    ws_id = session.active_workspace_id
    session.chat_history[ws_id] = []
    session.pending_interrupts.pop(ws_id, None)
    for k in [k for k in session.langgraph_threads if k.startswith(f"{ws_id}:")]:
        session.langgraph_threads.pop(k, None)
    return runtime.serialize_state(session)
