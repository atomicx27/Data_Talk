"""What-If & Future Scenario Simulation, Regression & Goal-Seeking Routes."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

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

    metrics: list[dict[str, str]] = []
    # Include pre-configured semantic measures
    for m in ws.measures:
        metrics.append({"id": m.name, "name": m.name, "table": m.table_name, "type": "measure"})

    # Include numeric columns from active fact/dimension tables
    for t in ws.active_tables():
        for c in t.columns:
            if c.is_blocked:
                continue
            c_low = c.name.lower()
            if any(k in c_low for k in ("rev", "sales", "price", "amount", "cost", "cogs", "fee", "tuition", "salary", "budget", "marks", "gpa", "quantity", "volume", "seats")):
                metrics.append({
                    "id": f"{t.name}.{c.name}",
                    "name": c.business_alias or f"{t.business_name or t.name}: {c.name}",
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
    )
    kpis = sim_res["kpis"]
    wf_steps = sim_res.get("waterfall_steps", [])
    regression = sim_res.get("regression")

    # Build Plotly Waterfall Figure
    waterfall_fig = None
    if wf_steps and req.include_waterfall:
        wf_df = pd.DataFrame([
            {"label": s["label"], "value": s["value"]}
            for s in wf_steps
        ])
        waterfall_fig = ChartRecommender.build_figure(wf_df, chart_type="waterfall", title="What-If Driver Attribution Bridge")

    # Build Plotly Fan Chart (Regression + Confidence Ribbon)
    fan_fig = None
    if regression and req.include_confidence_cone and regression.get("forecast_points"):
        hist_pts = regression.get("historical_points", [])
        fore_pts = regression.get("forecast_points", [])
        rows = []
        for h in hist_pts:
            rows.append({
                "date": h["date"],
                "actual": h["actual"],
                "trend": h["trend"],
                "projected": None,
                "p10": None,
                "p90": None,
            })
        for f in fore_pts:
            rows.append({
                "date": f["date"],
                "actual": None,
                "trend": None,
                "projected": f["projected"],
                "p10": f["p10"],
                "p90": f["p90"],
            })
        fan_df = pd.DataFrame(rows)
        fan_fig = ChartRecommender.build_figure(fan_df, chart_type="fan_chart", title=f"Future Trend Extrapolation: {kpis.get('metric_name', 'Metric')}")

    # Build automated "So What?" executive synthesis
    rev_delta = kpis["sim_revenue"] - kpis["base_revenue"]
    prof_delta = kpis["sim_profit"] - kpis["base_profit"]
    margin_bps = kpis.get("delta_margin_bps", 0.0)

    if prof_delta >= 0:
        verdict = f"Scenario projects a favorable +${prof_delta:,.2f} increase in net performance."
    else:
        verdict = f"Scenario creates a -${abs(prof_delta):,.2f} performance compression."

    narrative = {
        "headline": verdict,
        "detail": (
            f"Simulated {kpis.get('metric_name', 'Topline')} reaches ${kpis['sim_revenue']:,.2f} "
            f"({(rev_delta / kpis['base_revenue'] * 100):+.1f}%), yielding a margin shift of {margin_bps:+.0f} bps "
            f"(from {kpis.get('base_margin_pct', 0.0):.1f}% to {kpis.get('sim_margin_pct', 0.0):.1f}%)."
            if kpis["base_revenue"] > 0 else "Baseline data insufficient for margin calculation."
        ),
        "regression_insight": (
            f"Organic historical trend has slope {regression['slope']:+,.2f}/period with R²={regression['r2']:.2f}."
            if regression and regression.get("success") else "Insufficient time series depth for linear trend regression."
        ),
    }

    return {
        "success": True,
        "kpis": kpis,
        "waterfall_spec": fig_to_json(waterfall_fig) if waterfall_fig else None,
        "fan_spec": fig_to_json(fan_fig) if fan_fig else None,
        "regression": regression,
        "narrative": narrative,
        "active_levers": levers,
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
        )
        kpis = res["kpis"]
        if req.target_metric in ("sim_profit", "profit", "margin", "net_profit"):
            return float(kpis["sim_profit"])
        if req.target_metric in ("sim_revenue", "revenue", "sales"):
            return float(kpis["sim_revenue"])
        return float(kpis.get(req.target_metric, kpis["sim_profit"]))

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
