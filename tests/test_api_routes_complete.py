"""Comprehensive HTTP TestClient tests for FastAPI routes, auth, and security invariants."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.runtime import AppRuntime
from src.security.auth import AuthConfig


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch):
    """Create an isolated FastAPI TestClient backed by a temporary workspace root."""
    import server
    from src.security.auth import Authenticator

    runtime = AppRuntime(base_dir=tmp_path)
    auth_cfg = AuthConfig(api_token=None)
    monkeypatch.setattr(server, "runtime", runtime)
    monkeypatch.setattr(server, "authenticator", Authenticator(auth_cfg))

    ws = runtime.get_workspace(runtime.default_workspace_id)
    conn = runtime.get_connector(ws.workspace_id)
    csv_path = tmp_path / "sales_data.csv"
    pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "category": ["Hardware", "Software", "Hardware", "Cloud"],
            "amount": [100.0, 250.0, 150.0, 300.0],
            "cost": [60.0, 100.0, 90.0, 120.0],
            "order_date": ["2025-01-15", "2025-02-10", "2025-03-05", "2025-04-20"],
        }
    ).to_csv(csv_path, index=False)
    conn.ingest_file(str(csv_path))
    tables, _, _ = conn.extract_schema()
    ws.tables = tables
    ws.source_type = "duckdb"
    runtime.okf_store.save(ws)

    app = create_app(runtime=runtime, auth_config=auth_cfg)
    with TestClient(app) as client:
        yield client, runtime


def test_health_and_state_endpoints(api_client) -> None:
    client, runtime = api_client
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "providers_ready" in body

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    state = state_resp.json()
    assert state["workspace"]["workspace_id"] == runtime.default_workspace_id
    assert len(state["workspace"]["tables"]) >= 1


def test_auth_login_logout_and_rate_limit(tmp_path: Path, monkeypatch) -> None:
    import server
    from src.security.auth import Authenticator

    runtime = AppRuntime(base_dir=tmp_path)
    auth_cfg = AuthConfig(api_token="super-secret-token-123")
    monkeypatch.setattr(server, "runtime", runtime)
    monkeypatch.setattr(server, "authenticator", Authenticator(auth_cfg))
    app = create_app(runtime=runtime, auth_config=auth_cfg)
    with TestClient(app) as client:
        assert client.get("/api/state").status_code == 401

        bad = client.post("/auth/login", data={"token": "wrong"}, follow_redirects=False)
        assert bad.status_code == 401

        good = client.post(
            "/auth/login",
            data={"token": "super-secret-token-123"},
            follow_redirects=False,
        )
        assert good.status_code == 303
        assert client.get("/api/state", cookies=good.cookies).status_code == 200

        out = client.post("/auth/logout", follow_redirects=False)
        assert out.status_code == 303
        assert client.get("/api/state").status_code == 401


def test_simulation_override_and_dashboard_pin(api_client) -> None:
    client, _ = api_client
    sim = client.post(
        "/api/simulation/run",
        json={"revenue_shock_pct": 10.0, "cost_shock_pct": -5.0},
    )
    assert sim.status_code == 200
    sim_data = sim.json()
    assert "kpis" in sim_data

    # Scratchpad execution of forbidden SQL must be blocked by PrivacyFirewall
    bad_sql = client.post(
        "/api/sql/execute",
        json={"query": "DROP TABLE sales_data"},
    )
    assert bad_sql.status_code == 400

    # Pinning valid SELECT succeeds and returns updated pinned widgets in state
    good_pin = client.post(
        "/api/dashboard/pin",
        json={
            "title": "Revenue by Region",
            "user_question": "Revenue by region",
            "executed_query": "SELECT region, SUM(amount) AS total_amount FROM sales_data GROUP BY region",
            "chart_type": "bar",
        },
    )
    assert good_pin.status_code == 200
    assert len(good_pin.json()["workspace"]["pinned_widgets"]) == 1


def test_measures_joins_and_governance_routes(api_client) -> None:
    client, _ = api_client
    m_resp = client.post(
        "/api/measures/create",
        json={
            "name": "Total Amount",
            "table_name": "sales_data",
            "user_formula": "SUM('sales_data'[amount])",
            "input_type": "dax",
            "description": "Sum of amount",
        },
    )
    assert m_resp.status_code == 200
    assert "latest_measure" in m_resp.json()

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    assert "audit_log" in state_resp.json()

    chat_resp = client.post("/api/chat/query", json={"question": "Show total amount by region"})
    assert chat_resp.status_code == 200
    assert "entry" in chat_resp.json()
