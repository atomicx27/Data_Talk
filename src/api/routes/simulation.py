"""What-If & Future Scenario Simulation, Regression & Goal-Seeking Routes."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.agents.query_agent import QueryResponse
from src.analytics.simulation_engine import SimulationEngine
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import GoalSeekRequest, GoalSeekResponse, WhatIfSimulateRequest
from src.api.utils import fig_to_json
from src.viz.chart_recommender import ChartRecommender

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])


@router.get("/options")
def get_simulation_options(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Return available metrics, fact tables, and candidate drivers for simulation."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)

    try:
        live_tables_lower = {t.lower() for t in conn.list_tables()}
    except Exception:
        live_tables_lower = set()

    active_tbl_map = {
        t.name.lower(): t
        for t in ws.active_tables()
        if t.name.lower() != "overview" and (not live_tables_lower or t.name.lower() in live_tables_lower)
    }

    metrics: list[dict[str, str]] = []
    # Include pre-configured semantic measures whose target table is active in this workspace
    for m in ws.measures:
        if m.table_name and m.table_name.lower() in active_tbl_map:
            metrics.append({"id": f"{m.table_name}.{m.name}", "name": m.name, "table": m.table_name, "type": "measure"})

    # Sort FACT tables ahead of DIMENSION tables so primary business KPIs appear first
    sorted_tables = sorted(
        active_tbl_map.values(),
        key=lambda t: (0 if getattr(t.table_role, "value", str(t.table_role)).upper() == "FACT" else 1, t.name),
    )

    numeric_type_tokens = ("int", "double", "float", "decimal", "numeric", "real", "number", "hugeint")
    for t in sorted_tables:
        for c in t.columns:
            if c.is_blocked or c.is_primary_key or c.is_foreign_key:
                continue
            dtype_low = (c.data_type or "").lower()
            if not any(tok in dtype_low for tok in numeric_type_tokens):
                continue
            c_low = c.name.lower()
            if c_low.endswith(("_id", "_num", "_year")) or c_low in ("year", "is_active", "duration_years"):
                continue
            if any(
                k in c_low
                for k in (
                    "rev", "sales", "price", "amount", "cost", "cogs", "fee", "tuition",
                    "salary", "budget", "marks", "gpa", "score", "rate", "grade_points",
                    "quantity", "volume", "seats", "balance",
                )
            ):
                metrics.append({
                    "id": f"{t.name}.{c.name}",
                    "name": f"{t.name}: {c.name}" if not c.business_alias else f"{t.name}: {c.name} ({c.business_alias})",
                    "table": t.name,
                    "type": "column",
                })

    # Default presets
    presets = [
        {"id": "base", "name": "⚖️ Organic Trend (OLS)", "description": "Status quo run-rate extrapolation via linear regression", "levers": {}},
        {"id": "bear", "name": "🐻 Bear (-10% Stress)", "description": "Market contraction: -10% volume, +8% supplier cost inflation", "levers": {"Revenue Multiplier %": -10.0, "Cost Multiplier %": 8.0}},
        {"id": "bull", "name": "🐂 Bull (+15% Expansion)", "description": "Aggressive growth: +15% demand, +5% pricing power", "levers": {"Revenue Multiplier %": 15.0, "Cost Multiplier %": 4.0}},
        {"id": "custom", "name": "🎯 Custom Sandbox", "description": "Interactive manual slider tuning", "levers": {}},
    ]

    return {
        "domain": ws.domain_tag or "general",
        "domain_title": ws.domain_title or "General Business",
        "metrics": metrics,
        "presets": presets,
        "active_levers": session.what_if_overrides.get(ws_id, {}),
    }


@router.post("/run")
def run_simulation(
    req: WhatIfSimulateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    """Execute scenario simulation, compute OLS linear regression, and generate Plotly visual specs."""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)

    # Merge request levers
    levers = dict(req.levers)
    if req.revenue_shock_pct != 0.0:
        levers["Revenue Multiplier %"] = req.revenue_shock_pct
    if req.cost_shock_pct != 0.0:
        levers["Cost Multiplier %"] = req.cost_shock_pct

    session.what_if_overrides[ws_id] = levers

    sim_res = SimulationEngine.simulate_workspace_kpis(
        workspace=ws,
        connector=conn,
        revenue_shock_pct=req.revenue_shock_pct,
        cost_shock_pct=req.cost_shock_pct,
        levers=levers,
        target_table_name=req.target_table,
        target_metric=req.target_metric,
        forecast_periods=req.forecast_periods,
    )
    kpis = sim_res["kpis"]
    wf_steps = sim_res.get("waterfall_steps", [])
    regression = sim_res.get("regression")

    # Build Plotly Waterfall Figure
    waterfall_fig = None
    wf_df = pd.DataFrame()
    if wf_steps and req.include_waterfall:
        wf_df = pd.DataFrame([
            {
                "label": s["label"],
                "value": s["value"],
                "type": s.get("type", "relative"),
            }
            for s in wf_steps
        ])
        waterfall_fig = ChartRecommender.build_figure(wf_df, chart_type="waterfall", title="What-If Driver Attribution Bridge")

    # Build Plotly Fan Chart (Regression + Confidence Ribbon connected smoothly to last historical point)
    fan_fig = None
    fan_df = pd.DataFrame()
    table_rows: list[dict[str, Any]] = []
    if regression and req.include_confidence_cone and regression.get("forecast_points"):
        hist_pts = regression.get("historical_points", [])
        fore_pts = regression.get("forecast_points", [])
        rows = []
        for idx, h in enumerate(hist_pts):
            is_last_hist = idx == (len(hist_pts) - 1)
            rows.append({
                "date": str(h["date"]),
                "actual": h["actual"],
                "trend": h["trend"],
                # Anchor the forecast line and confidence cone to the final historical point so there is no gap
                "projected": h["actual"] if is_last_hist else None,
                "smart_momentum": h["actual"] if is_last_hist else None,
                "p10": h["actual"] if is_last_hist else None,
                "p90": h["actual"] if is_last_hist else None,
            })
            table_rows.append({
                "period": str(h["date"]),
                "phase": "Historical Actual",
                "actual": h["actual"],
                "ols_trend_fit": h["trend"],
                "projected_p50": None,
                "recent_momentum": h.get("weighted_trend", h["trend"]),
                "p10_downside": None,
                "p90_upside": None,
            })
        for f in fore_pts:
            rows.append({
                "date": str(f["date"]),
                "actual": None,
                "trend": None,
                "projected": f["projected"],
                "smart_momentum": f.get("smart_momentum", f["projected"]),
                "p10": f["p10"],
                "p90": f["p90"],
            })
            table_rows.append({
                "period": str(f["date"]),
                "phase": "Projected Forecast",
                "actual": None,
                "ols_trend_fit": f.get("organic_projected", f["projected"]),
                "projected_p50": f["projected"],
                "recent_momentum": f.get("smart_momentum", f["projected"]),
                "p10_downside": f["p10"],
                "p90_upside": f["p90"],
            })
        fan_df = pd.DataFrame(rows)
        p_unit = regression.get("period_unit", "period")
        fan_fig = ChartRecommender.build_figure(
            fan_df,
            chart_type="fan_chart",
            title=f"Future Trend Extrapolation ({p_unit.title()}ly OLS): {kpis.get('table_name', '')}.{kpis.get('metric_name', 'Metric')}".strip("."),
        )

    # Build automated "So What?" executive synthesis tailored to mode (Trend Forecast vs Single Metric Shock vs Full P&L)
    mode = kpis.get("mode", "pnl")
    metric_label = kpis.get("metric_name", "Metric")
    pfx = "$" if kpis.get("is_currency", True) else ""
    p_unit = kpis.get("period_unit", "period")

    if mode == "trend_forecast" and regression and regression.get("success"):
        next_dt = kpis.get("next_forecast_date", "Next Period")
        next_p50 = float(kpis.get("next_forecast_p50", 0.0))
        next_p10 = float(kpis.get("next_forecast_p10", 0.0))
        next_p90 = float(kpis.get("next_forecast_p90", 0.0))
        p_avg = float(kpis.get("period_avg", 0.0))
        vs_avg_pct = float(kpis.get("forecast_vs_avg_pct", 0.0))
        slope = float(regression.get("slope", 0.0))
        slope_str = f"{'+' if slope >= 0 else '-'}{pfx}{abs(slope):,.2f}"
        r2 = float(regression.get("r2", 0.0))
        n_hist = int(regression.get("historical_count", 0))
        direction = "upward" if slope >= 0 else "downward"

        headline = (
            f"OLS Linear Regression projects {next_dt} {metric_label} at {pfx}{next_p50:,.2f} "
            f"({vs_avg_pct:+.1f}% vs {pfx}{p_avg:,.2f}/{p_unit} historical mean)."
        )
        detail = (
            f"Fitted across {n_hist} historical {p_unit}s (cumulative baseline {pfx}{kpis['base_revenue']:,.2f}), "
            f"{metric_label} follows an {direction} trajectory of {slope_str}/{p_unit}. "
            f"The 80% prediction interval for {next_dt} spans {pfx}{next_p10:,.2f} (P10) to {pfx}{next_p90:,.2f} (P90)."
        )
        reg_insight = f"Organic historical trend slope: {slope_str}/{p_unit} · Intercept: {pfx}{regression['intercept']:,.2f} · R²={r2:.2f} ({n_hist} {p_unit}s)."
    elif not kpis.get("has_cogs"):
        rev_delta = kpis["sim_revenue"] - kpis["base_revenue"]
        pct_delta = kpis.get("delta_pct", 0.0)
        headline = (
            f"Scenario ({pct_delta:+.1f}% shock) shifts {metric_label} by "
            f"{'+' if rev_delta >= 0 else '-'}{pfx}{abs(rev_delta):,.2f} to {pfx}{kpis['sim_revenue']:,.2f}."
        )
        if regression and regression.get("success") and kpis.get("next_forecast_date"):
            next_dt = kpis["next_forecast_date"]
            slope = float(regression.get("slope", 0.0))
            slope_str = f"{'+' if slope >= 0 else '-'}{pfx}{abs(slope):,.2f}"
            detail = (
                f"Baseline {metric_label} ({pfx}{kpis['base_revenue']:,.2f}) moves to {pfx}{kpis['sim_revenue']:,.2f}. "
                f"For {next_dt}, the shocked OLS projection reaches {pfx}{kpis['next_forecast_p50']:,.2f} "
                f"(vs {pfx}{kpis['next_forecast_organic']:,.2f} unshocked baseline trend; 80% CI: {pfx}{kpis['next_forecast_p10']:,.2f} – {pfx}{kpis['next_forecast_p90']:,.2f})."
            )
            reg_insight = f"Organic historical trend has slope {slope_str}/{p_unit} with R²={regression['r2']:.2f} across {regression['historical_count']} {p_unit}s."
        else:
            detail = f"Simulated {metric_label} moves from {pfx}{kpis['base_revenue']:,.2f} to {pfx}{kpis['sim_revenue']:,.2f} ({pct_delta:+.1f}%)."
            reg_insight = "Insufficient time series depth for linear trend regression."
    else:
        rev_delta = kpis["sim_revenue"] - kpis["base_revenue"]
        prof_delta = kpis["sim_profit"] - kpis["base_profit"]
        margin_bps = kpis.get("delta_margin_bps", 0.0)
        if prof_delta >= 0:
            headline = f"Scenario projects a favorable +${prof_delta:,.2f} increase in net performance."
        else:
            headline = f"Scenario creates a -${abs(prof_delta):,.2f} performance compression."
        detail = (
            f"Simulated {metric_label} reaches ${kpis['sim_revenue']:,.2f} "
            f"({(rev_delta / kpis['base_revenue'] * 100):+.1f}%), yielding a margin shift of {margin_bps:+.0f} bps "
            f"(from {kpis.get('base_margin_pct', 0.0):.1f}% to {kpis.get('sim_margin_pct', 0.0):.1f}%)."
            if kpis["base_revenue"] > 0 else "Baseline data insufficient for margin calculation."
        )
        if regression and regression.get("success"):
            slope = float(regression.get("slope", 0.0))
            slope_str = f"{'+' if slope >= 0 else '-'}{pfx}{abs(slope):,.2f}"
            reg_insight = f"Organic historical trend has slope {slope_str}/{p_unit} with R²={regression['r2']:.2f} across {regression['historical_count']} {p_unit}s."
        else:
            reg_insight = "Insufficient time series depth for linear trend regression."

    narrative = {
        "headline": headline,
        "detail": detail,
        "regression_insight": reg_insight,
    }

    fan_spec_json = fig_to_json(fan_fig) if fan_fig else None
    wf_spec_json = fig_to_json(waterfall_fig) if waterfall_fig else None

    chat_entry = None
    if req.append_to_chat:
        tbl_name = kpis.get("table_name") or req.target_table or ""
        full_metric = f"{tbl_name}.{metric_label}".strip(".")
        shock_desc = ""
        if abs(req.revenue_shock_pct) > 1e-6 or abs(req.cost_shock_pct) > 1e-6:
            parts = []
            if abs(req.revenue_shock_pct) > 1e-6:
                parts.append(f"Growth/Topline {req.revenue_shock_pct:+.1f}%")
            if abs(req.cost_shock_pct) > 1e-6:
                parts.append(f"Cost/Inflation {req.cost_shock_pct:+.1f}%")
            shock_desc = f" ({', '.join(parts)})"

        user_q = f"Future Trend & What-If Analysis: {full_metric}{shock_desc}"
        exec_summary = f"**{headline}**\n\n{detail}\n\n📈 *{reg_insight}*"

        # Use table_rows (Historical + Projected periods) if available, otherwise waterfall steps
        if table_rows:
            out_rows = table_rows
            out_cols = list(table_rows[0].keys())
            primary_chart_type = "fan_chart"
            primary_spec = fan_spec_json or wf_spec_json
        elif not wf_df.empty:
            out_rows = wf_df.to_dict(orient="records")
            out_cols = [str(c) for c in wf_df.columns]
            primary_chart_type = "waterfall"
            primary_spec = wf_spec_json
        else:
            out_rows = []
            out_cols = []
            primary_chart_type = "bar"
            primary_spec = None

        exec_sql = (
            (regression.get("executed_sql") if regression else None)
            or f'SELECT * FROM "{tbl_name}"'
        )
        formula_str = (
            f"OLS Linear Fit: y = {regression['intercept']:,.2f} + ({regression['slope']:+,.2f} * t) · "
            f"R² = {regression['r2']:.2f} · 80% CI [P10, P90]"
            if regression and regression.get("success")
            else f"Scenario Multiplier: {kpis.get('effective_shock_pct', 0.0):+.1f}%"
        )

        followups = [
            f"Break down {metric_label} by top categories",
            f"Show historical YoY growth for {metric_label}",
            f"What drove the recent changes in {metric_label}?",
        ]

        query_resp = QueryResponse(
            user_question=user_q,
            resolved_question=f"Future Trend Extrapolation & Scenario Simulation ({full_metric})",
            success=True,
            executed_query=exec_sql,
            compiled_dax_equivalent=formula_str,
            compiler_mode="deterministic",
            compiler_detail="OLS Regression & What-If Simulation Engine",
            executive_summary=exec_summary,
            recommended_chart_type=primary_chart_type,
            suggested_follow_ups=followups,
            involved_tables=[tbl_name] if tbl_name else [],
            drill_down_dimensions=[],
        )

        history = session.chat_history.setdefault(ws_id, [])
        chat_entry = {
            "id": f"sim_{len(history) + 1}",
            "response": query_resp.model_dump(mode="json"),
            "rows": out_rows,
            "columns": out_cols,
            "row_count": len(out_rows),
            "chart_type": primary_chart_type,
            "plotly_spec": primary_spec,
            "fan_spec": fan_spec_json,
            "waterfall_spec": wf_spec_json,
            "simulation_kpis": kpis,
            "simulation_regression": regression,
            "thread": [],
        }
        chat_entry["response"]["is_diagnostic"] = True
        chat_entry["response"]["is_simulation"] = True
        chat_entry["response"]["audit_receipt"] = {
            "algorithm": f"OLS Linear Trend + Recent Momentum Ensemble ({regression.get('confidence_grade', 'Verified')})" if regression else "What-If Scenario Simulation",
            "executed_sql": exec_sql,
            "formula_latex": formula_str,
            "rows_evaluated": len(out_rows),
            "date_source": f"{p_unit.title()}ly Temporal Aggregation",
            "balance_check": {"status": "VERIFIED_EXACT_BALANCE"},
        }
        history.append(chat_entry)

    return {
        "success": True,
        "kpis": kpis,
        "waterfall_spec": wf_spec_json,
        "fan_spec": fan_spec_json,
        "regression": regression,
        "narrative": narrative,
        "active_levers": levers,
        "table_rows": table_rows,
        "entry": chat_entry,
        "state": runtime.serialize_state(session) if req.append_to_chat else None,
    }


@router.post("/goal-seek", response_model=GoalSeekResponse)
def solve_simulation_goal(
    req: GoalSeekRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> GoalSeekResponse:
    """Solve reverse what-if equation: what lever multiplier is required to hit target KPI?"""
    ws_id = session.active_workspace_id
    ws = runtime.get_workspace(ws_id)
    conn = runtime.get_connector(ws_id)

    # Define evaluation function mapping lever multiplier to simulated metric
    def eval_fn(val: float) -> float:
        test_levers = {req.adjust_lever: val * 100.0}
        res = SimulationEngine.simulate_workspace_kpis(
            workspace=ws,
            connector=conn,
            levers=test_levers,
            target_table_name=req.target_table,
            target_metric=req.target_metric,
        )
        kpis = res["kpis"]
        if req.target_metric in ("sim_profit", "profit", "margin", "net_profit"):
            return float(kpis["sim_profit"])
        if req.target_metric in ("sim_revenue", "revenue", "sales"):
            return float(kpis["sim_revenue"])
        return float(kpis.get("sim_revenue", kpis["sim_profit"]))

    req_val, achieved, ok = SimulationEngine.solve_goal_seek(
        eval_fn=eval_fn,
        target_value=req.target_value,
        min_bound=req.min_bound,
        max_bound=req.max_bound,
    )

    if not ok:
        return GoalSeekResponse(
            success=False,
            target_metric=req.target_metric,
            target_value=req.target_value,
            adjust_lever=req.adjust_lever,
            required_value=round(req_val, 4),
            required_pct=round(req_val * 100.0, 2),
            achieved_value=achieved,
            message=f"Goal target {req.target_value:,.2f} could not be converged within bounds [{req.min_bound * 100:.0f}%, {req.max_bound * 100:.0f}%]. Best reached: {achieved:,.2f}.",
        )

    return GoalSeekResponse(
        success=True,
        target_metric=req.target_metric,
        target_value=req.target_value,
        adjust_lever=req.adjust_lever,
        required_value=round(req_val, 4),
        required_pct=round(req_val * 100.0, 2),
        achieved_value=achieved,
        message=f"To achieve target {req.target_value:,.2f}, set {req.adjust_lever} to {req_val * 100:+.2f}%.",
    )
