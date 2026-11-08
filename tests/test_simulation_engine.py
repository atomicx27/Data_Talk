"""Automated unit & integration tests for SimulationEngine, OLS Regression, and What-If routes."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import app
from src.agents.query_agent import QueryAgent
from src.analytics.simulation_engine import SimulationEngine


def test_linear_trend_regression_perfect_line():
    """Verify OLS regression finds exact slope and R2=1.0 for a linear series."""
    df = pd.DataFrame({
        "month": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"],
        "revenue": [100.0, 120.0, 140.0, 160.0, 180.0],
    })
    res = SimulationEngine.fit_linear_trend_regression(df, "month", "revenue", forecast_periods=3)
    assert res["success"] is True
    assert res["slope"] == pytest.approx(20.0, abs=1e-2)
    assert res["intercept"] == pytest.approx(100.0, abs=1e-2)
    assert res["r2"] == pytest.approx(1.0, abs=1e-3)
    assert len(res["forecast_points"]) == 3
    # Check extrapolated dates
    assert res["forecast_points"][0]["date"] == "2025-06"
    assert res["forecast_points"][0]["projected"] == pytest.approx(200.0, abs=1e-2)
    assert res["forecast_points"][1]["date"] == "2025-07"
    assert res["forecast_points"][1]["projected"] == pytest.approx(220.0, abs=1e-2)


def test_linear_trend_regression_noisy_series():
    """Verify regression works with realistic noisy data and constructs widening fan intervals."""
    df = pd.DataFrame({
        "date": [f"2025-{i:02d}" for i in range(1, 13)],
        "sales": [105, 118, 112, 130, 125, 142, 138, 155, 150, 168, 162, 180],
    })
    res = SimulationEngine.fit_linear_trend_regression(df, "date", "sales", forecast_periods=4)
    assert res["success"] is True
    assert res["slope"] > 0
    assert res["r2"] > 0.90
    forecasts = res["forecast_points"]
    assert len(forecasts) == 4
    # P10 <= P50 <= P90 invariant
    for f in forecasts:
        assert f["p10"] <= f["p50"] <= f["p90"]
    # Widening trumpet shape: future interval width expands over time
    width_1 = forecasts[0]["p90"] - forecasts[0]["p10"]
    width_4 = forecasts[3]["p90"] - forecasts[3]["p10"]
    assert width_4 >= width_1


def test_goal_seek_solver():
    """Verify Brent's root-finder converges to required lever percentage."""
    base_val = 200000.0
    target_val = 250000.0  # +25% increase needed
    lever, achieved, ok = SimulationEngine.solve_goal_seek(
        eval_fn=lambda m: base_val * (1.0 + m),
        target_value=target_val,
        min_bound=-0.50,
        max_bound=1.00,
    )
    assert ok is True
    assert lever == pytest.approx(0.25, abs=1e-3)
    assert achieved == pytest.approx(250000.0, abs=1.0)


def test_monte_carlo_distribution():
    """Verify Monte Carlo returns sensible quantiles."""
    mc = SimulationEngine.run_monte_carlo(base_value=1000.0, lever_delta=0.05, elasticity=-1.2)
    assert "P10" in mc and "P50" in mc and "P90" in mc
    assert mc["P10"] <= mc["P50"] <= mc["P90"]


def test_waterfall_steps_decomposition():
    """Verify waterfall attribution steps sum up properly."""
    steps = SimulationEngine.compute_waterfall_steps(
        base_profit=1000.0,
        sim_profit=1200.0,
        price_effect=300.0,
        cost_effect=-100.0,
    )
    assert len(steps) >= 4
    assert steps[0]["label"] == "Baseline Target"
    assert steps[0]["value"] == 1000.0
    assert steps[-1]["label"] == "Simulated Outcome"
    assert steps[-1]["value"] == 1200.0


def test_conversational_what_if_extraction():
    """Verify QueryAgent parses conversational multipliers from user natural language prompts."""
    q1 = "What if revenue increases by 15% and costs drop 5% next quarter?"
    overrides1 = QueryAgent._extract_what_if_from_prompt(q1)
    assert overrides1.get("Revenue Multiplier %") == pytest.approx(0.15, abs=1e-3)
    assert overrides1.get("Cost Multiplier %") == pytest.approx(-0.05, abs=1e-3)

    q2 = "Simulate 8% cost inflation"
    overrides2 = QueryAgent._extract_what_if_from_prompt(q2)
    assert overrides2.get("Cost Multiplier %") == pytest.approx(0.08, abs=1e-3)

    q3 = "Show total sales by category"
    overrides3 = QueryAgent._extract_what_if_from_prompt(q3)
    assert len(overrides3) == 0


def test_api_simulation_endpoints():
    """Verify FastAPI simulation REST endpoints with authenticated test client."""
    client = TestClient(app)

    # 1. Load demo dataset first
    connect_resp = client.post("/api/connect/demo")
    assert connect_resp.status_code == 200

    # 2. Test /api/simulation/options
    opt_resp = client.get("/api/simulation/options")
    assert opt_resp.status_code == 200
    opt_data = opt_resp.json()
    assert "metrics" in opt_data
    assert "presets" in opt_data

    # 3. Test /api/simulation/run
    run_resp = client.post(
        "/api/simulation/run",
        json={
            "revenue_shock_pct": 10.0,
            "cost_shock_pct": -5.0,
            "include_waterfall": True,
            "include_confidence_cone": True,
        },
    )
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["success"] is True
    assert "kpis" in run_data
    assert run_data["kpis"]["sim_revenue"] > 0
    assert "narrative" in run_data

    # 4. Test /api/simulation/goal-seek
    base_prof = run_data["kpis"]["base_profit"]
    target_goal = base_prof * 1.10
    goal_resp = client.post(
        "/api/simulation/goal-seek",
        json={
            "target_metric": "sim_profit",
            "target_value": target_goal,
            "adjust_lever": "Revenue Multiplier %",
        },
    )
    assert goal_resp.status_code == 200
    goal_data = goal_resp.json()
    assert goal_data["success"] is True
    assert goal_data["required_pct"] > 0


def test_fan_chart_chronological_order_and_single_metric_forecast():
    """Verify fan_chart places historical dates first and forecast dates on the right, and handles single-metric OLS."""
    from pathlib import Path
    from src.api.routes.simulation import get_simulation_options, run_simulation
    from src.api.runtime import AppRuntime, SessionState
    from src.api.schemas import WhatIfSimulateRequest

    rt = AppRuntime(Path("."))
    if "univercity" not in rt.list_workspaces():
        return

    s = SessionState(session_id="test_univ", active_workspace_id="univercity")
    opts = get_simulation_options(rt, s)
    metric_ids = [m["id"] for m in opts["metrics"]]
    # Ensure non-numeric 'remarks' and unattached tables are never included
    assert "enrollments_marks.remarks" not in metric_ids
    assert "faculty.salary" in metric_ids

    res = run_simulation(
        WhatIfSimulateRequest(
            revenue_shock_pct=0.0,
            cost_shock_pct=0.0,
            target_metric="faculty.salary",
            target_table="faculty",
        ),
        rt,
        s,
    )
    kpis = res["kpis"]
    reg = res["regression"]
    assert kpis["mode"] == "trend_forecast"
    assert kpis["has_cogs"] is False
    assert kpis["sim_margin_pct"] == 0.0
    assert reg["historical_count"] == 19
    assert reg["slope"] == pytest.approx(14649.2658, abs=1.0)
    assert kpis["next_forecast_date"] == "2024"
    assert kpis["next_forecast_p50"] == pytest.approx(2290993.61, abs=1.0)

    # Verify fan chart X-axis is strictly chronological (2005..2023 followed by 2024..2027)
    cat_arr = res["fan_spec"]["layout"]["xaxis"]["categoryarray"]
    assert cat_arr[0] == "2005"
    assert cat_arr[18] == "2023"
    assert cat_arr[19:] == ["2024", "2025", "2026", "2027"]
    # Verify Historical Actuals is the first trace
    assert res["fan_spec"]["data"][0]["name"] == "Historical Actuals"

    # Verify Waterfall Bridge measure array grounds the first bar as 'absolute' and final bar as 'total' (preventing 2x double-stacking)
    wf_trace = res["waterfall_spec"]["data"][0]
    assert list(wf_trace["measure"]) == ["absolute", "relative", "relative", "total"]
    # Verify hoverlabel has explicit high-contrast font color
    assert res["fan_spec"]["layout"]["hoverlabel"]["font"]["color"] == "#f8fafc"

    # Verify Recent-Weighted Momentum and Predictability Confidence Grade
    assert "weighted_slope" in reg
    assert "confidence_grade" in reg
    assert kpis["next_forecast_smart"] > 0
    assert any(t.get("name") == "Recent Momentum (Weighted)" for t in res["fan_spec"]["data"])

    # Verify append_to_chat=True appends full interactive chat card + historical/projected table rows to chat stream
    res_chat = run_simulation(
        WhatIfSimulateRequest(
            revenue_shock_pct=0.0,
            cost_shock_pct=0.0,
            target_metric="faculty.salary",
            target_table="faculty",
            forecast_periods=6,
            append_to_chat=True,
        ),
        rt,
        s,
    )
    assert "entry" in res_chat
    assert len(s.chat_history["univercity"]) == 1
    chat_entry = res_chat["entry"]
    assert chat_entry["chart_type"] == "fan_chart"
    # 19 historical years + 6 projected years = 25 rows in table
    assert len(chat_entry["rows"]) == 25
    assert chat_entry["rows"][-1]["period"] == "2029"
    assert chat_entry["rows"][-1]["phase"] == "Projected Forecast"
    assert chat_entry["rows"][-1]["projected_p50"] is not None
    assert chat_entry["rows"][-1]["recent_momentum"] is not None

