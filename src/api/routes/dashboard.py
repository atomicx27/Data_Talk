"""Dashboard, Root Cause Analysis, What-If Simulation, and SQL Scratchpad routes."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.agents.query_agent import QueryResponse
from src.agents.root_cause_agent import RootCauseAgent
from src.analytics.simulation_engine import SimulationEngine
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import (
    PinWidgetRequest,
    RootCauseRequest,
    ScratchpadQueryRequest,
    WhatIfSimulateRequest,
)
from src.api.utils import fig_to_json
from src.knowledge.okf_models import PinnedWidget
from src.viz.chart_recommender import ChartRecommender

router = APIRouter(tags=["Dashboard"])


@router.get("/api/root-cause/options")
def get_root_cause_options(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)
    return RootCauseAgent.discover_options(workspace=ws, connector=conn)


@router.post("/api/root-cause")
def run_root_cause_analysis(
    req: RootCauseRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)
    res = RootCauseAgent.analyze_variance(
        workspace=ws,
        connector=conn,
        metric_col=req.metric_col,
        dimension=req.dimension,
        algorithm=req.algorithm,
        window_type=req.window_type,
        date_col=req.date_col,
        t0_start=req.t0_start,
        t0_end=req.t0_end,
        t1_start=req.t1_start,
        t1_end=req.t1_end,
    )
    rc_df: pd.DataFrame = res["df"]
    chart_type = res.get("chart_type", "waterfall")
    dim_title = req.dimension if req.dimension else "All Dimensions"
    title = f"Diagnostic Decomposition: {req.metric_col} by {dim_title}"

    fig = None
    if not rc_df.empty:
        try:
            fig = ChartRecommender.build_figure(
                rc_df,
                chart_type=chart_type,
                title=title,
                x_col="segment",
            )
        except Exception:
            fig = None

    audit = res.get("audit_receipt", {})
    executed_sql = audit.get("executed_sql", "")
    formula = audit.get("formula_latex", "")

    # Build structured QueryResponse for Conversational BI chat stream
    user_q = f"Explain variance in {req.metric_col} by {dim_title}"
    if req.window_type == "custom" and req.t0_start and req.t1_end:
        user_q += f" ({req.t0_start}..{req.t0_end} vs {req.t1_start}..{req.t1_end})"
    else:
        user_q += f" ({req.window_type.replace('_', ' ').title()})"

    query_resp = QueryResponse(
        user_question=user_q,
        resolved_question=title,
        success=True,
        executed_query=executed_sql,
        compiled_dax_equivalent=formula,
        compiler_mode="deterministic",
        compiler_detail=audit.get("algorithm", "Diagnostic Decomposition Engine"),
        executive_summary=res["summary"],
        recommended_chart_type=chart_type,
        suggested_follow_ups=res.get("suggested_followups", []),
        involved_tables=list(audit.get("tables_used", [])),
        drill_down_dimensions=[],
    )

    history = session.chat_history.setdefault(ws_id, [])
    chat_entry = {
        "id": f"rc_{len(history) + 1}",
        "response": query_resp.model_dump(mode="json"),
        "rows": rc_df.to_dict(orient="records") if not rc_df.empty else [],
        "columns": [str(c) for c in rc_df.columns],
        "row_count": len(rc_df),
        "chart_type": chart_type,
        "plotly_spec": fig_to_json(fig) if fig else None,
        "thread": [],
    }
    chat_entry["response"]["is_diagnostic"] = True
    chat_entry["response"]["kpi_ribbon"] = res.get("kpi_ribbon", {})
    chat_entry["response"]["audit_receipt"] = audit
    history.append(chat_entry)

    return {
        "entry": chat_entry,
        "state": runtime.serialize_state(session),
        "summary": res["summary"],
        "rows": chat_entry["rows"],
        "columns": chat_entry["columns"],
        "chart_type": chart_type,
        "plotly_spec": chat_entry["plotly_spec"],
        "kpi_ribbon": res.get("kpi_ribbon", {}),
        "audit_receipt": audit,
        "suggested_followups": res.get("suggested_followups", []),
    }


@router.post("/api/dashboard/pin")
def pin_widget(
    req: PinWidgetRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        widget = PinnedWidget(
            id=f"pin_{len(ws.pinned_widgets) + 1}",
            title=req.title[:70],
            user_question=req.user_question,
            executed_query=req.executed_query,
            query_dialect=req.query_dialect,
            chart_type=req.chart_type,
        )
        ws.pinned_widgets.append(widget)
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.delete("/api/dashboard/pin/{widget_id}")
def delete_pinned_widget(
    widget_id: str,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        ws.pinned_widgets = [w for w in ws.pinned_widgets if w.id != widget_id]
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.post("/api/dashboard/simulate")
def simulate_dashboard(
    req: WhatIfSimulateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)

    levers = dict(req.levers)
    if req.revenue_shock_pct != 0.0:
        levers["Revenue Multiplier %"] = req.revenue_shock_pct
    if req.cost_shock_pct != 0.0:
        levers["Cost Multiplier %"] = req.cost_shock_pct

    session.what_if_overrides[ws_id] = {
        "Revenue Multiplier %": req.revenue_shock_pct / 100.0,
        "Cost Multiplier %": req.cost_shock_pct / 100.0,
    }

    sim_res = SimulationEngine.simulate_workspace_kpis(
        workspace=ws,
        connector=conn,
        revenue_shock_pct=req.revenue_shock_pct,
        cost_shock_pct=req.cost_shock_pct,
        levers=levers,
    )
    kpis = sim_res["kpis"]

    rendered_widgets: list[dict[str, Any]] = []
    for pw in ws.pinned_widgets:
        try:
            w_df = conn.execute_query(pw.executed_query, max_rows=300)
            fig = ChartRecommender.build_figure(w_df, chart_type=pw.chart_type, title=pw.title)
            rendered_widgets.append(
                {
                    "widget": pw.model_dump(mode="json"),
                    "rows": w_df.head(50).to_dict(orient="records"),
                    "plotly_spec": fig_to_json(fig),
                    "error": None,
                }
            )
        except Exception as exc:
            rendered_widgets.append(
                {
                    "widget": pw.model_dump(mode="json"),
                    "rows": [],
                    "plotly_spec": None,
                    "error": str(exc),
                }
            )

    return {
        "kpis": kpis,
        "widgets": rendered_widgets,
    }


@router.post("/api/sql/execute")
def execute_scratchpad_query(
    req: ScratchpadQueryRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Execute a live read-only SQL or DAX query in the Studio Scratchpad (guarded by PrivacyFirewall AST)."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)
    try:
        df = conn.execute_query(
            req.query.strip(),
            max_rows=200,
            blocked_tables=ws.blocked_table_names(),
            blocked_columns=ws.blocked_column_names(),
        )
        return {
            "success": True,
            "row_count": len(df),
            "columns": [str(c) for c in df.columns],
            "rows": df.head(100).to_dict(orient="records"),
            "error": None,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
