"""Dashboard, Root Cause Analysis, What-If Simulation, and SQL Scratchpad routes."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.agents.root_cause_agent import RootCauseAgent
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
    )
    rc_df: pd.DataFrame = res["df"]
    fig_wf = ChartRecommender.build_figure(
        rc_df,
        chart_type="waterfall",
        title=f"Period-over-Period Variance of {req.metric_col} by {req.dimension}",
        x_col="segment",
    )
    return {
        "summary": res["summary"],
        "rows": rc_df.to_dict(orient="records") if not rc_df.empty else [],
        "columns": [str(c) for c in rc_df.columns],
        "plotly_spec": fig_to_json(fig_wf),
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
    rm = 1.0 + (req.revenue_shock_pct / 100.0)
    cm = 1.0 + (req.cost_shock_pct / 100.0)
    session.what_if_overrides[ws_id] = {
        "Revenue Multiplier %": req.revenue_shock_pct / 100.0,
        "Cost Multiplier %": req.cost_shock_pct / 100.0,
    }

    kpis: dict[str, Any] = {
        "sim_revenue": 0.0,
        "sim_profit": 0.0,
        "sim_margin_pct": 0.0,
        "base_revenue": 0.0,
        "base_profit": 0.0,
    }
    target_tbl = ws.get_table("orders") or ws.get_table("fact_orders") or ws.get_table("fee_invoices")
    if not target_tbl:
        for t in ws.active_tables():
            col_names = [c.name.lower() for c in t.columns]
            if any(any(k in c for k in ("rev", "sales", "price", "amount", "fee", "tuition", "salary", "budget")) for c in col_names):
                target_tbl = t
                break

    if target_tbl:
        try:
            col_map = {c.name.lower(): c.name for c in target_tbl.columns if not c.is_blocked}
            rev_col = (
                col_map.get("rev_net_usd")
                or col_map.get("net_revenue_usd")
                or col_map.get("gross_revenue_usd")
                or col_map.get("revenue")
                or col_map.get("sales")
                or col_map.get("total_fee_due")
                or col_map.get("amount_paid")
                or col_map.get("tuition_fee")
                or col_map.get("salary")
                or col_map.get("annual_budget")
            )
            if not rev_col:
                for cl, cname in col_map.items():
                    if any(k in cl for k in ("rev", "sales", "fee", "amount", "salary", "budget")) and not cl.endswith("_id"):
                        rev_col = cname
                        break
            cogs_col = (
                col_map.get("cogs_usd")
                or col_map.get("cogs")
                or col_map.get("cost")
                or col_map.get("unit_cost_usd")
                or col_map.get("scholarship_discount")
                or col_map.get("balance_due")
            )
            stat_col = col_map.get("ord_stat") or col_map.get("order_status") or col_map.get("status")
            where_clause = ""
            if stat_col:
                where_clause = f'WHERE "{stat_col}" NOT IN (\'X\', \'Cancelled\')'

            if rev_col:
                cogs_expr = f'"{cogs_col}"' if cogs_col else "0"
                sim_df = conn.execute_query(
                    f"SELECT "
                    f'ROUND(SUM("{rev_col}"), 2) AS base_rev, '
                    f'ROUND(SUM("{rev_col}" - {cogs_expr}), 2) AS base_prof, '
                    f'ROUND(SUM("{rev_col}" * {rm}), 2) AS sim_rev, '
                    f'ROUND(SUM(("{rev_col}" * {rm}) - ({cogs_expr} * {cm})), 2) AS sim_prof '
                    f'FROM "{target_tbl.name}" {where_clause}',
                    max_rows=1,
                )
                if not sim_df.empty:
                    r0 = sim_df.iloc[0]
                    sim_rev = float(r0["sim_rev"] or 0.0)
                    sim_prof = float(r0["sim_prof"] or 0.0)
                    kpis = {
                        "base_revenue": float(r0["base_rev"] or 0.0),
                        "base_profit": float(r0["base_prof"] or 0.0),
                        "sim_revenue": sim_rev,
                        "sim_profit": sim_prof,
                        "sim_margin_pct": round((sim_prof / sim_rev) * 100.0, 2) if sim_rev else 0.0,
                    }
        except Exception:
            pass

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
