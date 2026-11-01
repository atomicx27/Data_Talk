"""Comprehensive Unit & Mathematical Verification Tests for RootCauseAgent."""

import duckdb
import pandas as pd
import pytest

from src.agents.root_cause_agent import RootCauseAgent
from src.connectors.duckdb_connector import DuckDBConnector
from src.knowledge.okf_models import (
    ColumnProfile,
    OKFWorkspace,
    RelationshipEdge,
    TableRole,
    TableSchema,
)


@pytest.fixture
def mock_duckdb_retail(tmp_path):
    """Create in-memory DuckDB warehouse with retail orders, products, customers."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE products (
            product_id VARCHAR PRIMARY KEY,
            category VARCHAR
        );
        INSERT INTO products VALUES
            ('P1', 'Electronics'),
            ('P2', 'Apparel'),
            ('P3', 'Home Goods');

        CREATE TABLE customers (
            customer_id VARCHAR PRIMARY KEY,
            region_code VARCHAR,
            cust_tier VARCHAR
        );
        INSERT INTO customers VALUES
            ('C1', 'North', 'Tier 1'),
            ('C2', 'South', 'Tier 2'),
            ('C3', 'West', 'Tier 1');

        CREATE TABLE orders (
            order_id VARCHAR PRIMARY KEY,
            product_id VARCHAR,
            customer_id VARCHAR,
            sales_channel VARCHAR,
            order_date DATE,
            rev_net_usd DOUBLE,
            profit_usd DOUBLE,
            ord_stat VARCHAR
        );
        INSERT INTO orders VALUES
            -- 2024 Orders
            ('O1', 'P1', 'C1', 'Online', '2024-05-10', 1000.0, 300.0, 'C'),
            ('O2', 'P2', 'C2', 'Store',  '2024-06-15', 500.0,  100.0, 'C'),
            ('O3', 'P3', 'C3', 'Online', '2024-07-20', 800.0,  200.0, 'C'),
            ('O4', 'P1', 'C2', 'Store',  '2024-08-01', 1200.0, 400.0, 'C'),
            -- 2025 Orders (Electronics grew, Home Goods dropped)
            ('O5', 'P1', 'C1', 'Online', '2025-05-10', 2500.0, 800.0, 'C'),
            ('O6', 'P2', 'C2', 'Store',  '2025-06-15', 600.0,  120.0, 'C'),
            ('O7', 'P3', 'C3', 'Online', '2025-07-20', 200.0,  40.0,  'C'),
            ('O8', 'P1', 'C2', 'Store',  '2025-08-01', 1500.0, 500.0, 'C'),
            -- Cancelled order (should be excluded)
            ('O9', 'P1', 'C1', 'Online', '2025-09-01', 9999.0, 9999.0, 'X');
    """)

    workspace = OKFWorkspace(
        workspace_id="test_retail",
        workspace_name="Test Retail Workspace",
        tables=[
            TableSchema(
                name="orders",
                table_role=TableRole.FACT,
                row_count=9,
                columns=[
                    ColumnProfile(name="order_id", data_type="VARCHAR", is_primary_key=True),
                    ColumnProfile(name="product_id", data_type="VARCHAR", is_foreign_key=True),
                    ColumnProfile(name="customer_id", data_type="VARCHAR", is_foreign_key=True),
                    ColumnProfile(name="sales_channel", data_type="VARCHAR"),
                    ColumnProfile(name="order_date", data_type="DATE"),
                    ColumnProfile(name="rev_net_usd", data_type="DOUBLE"),
                    ColumnProfile(name="profit_usd", data_type="DOUBLE"),
                    ColumnProfile(name="ord_stat", data_type="VARCHAR"),
                ],
            ),
            TableSchema(
                name="products",
                table_role=TableRole.DIMENSION,
                row_count=3,
                columns=[
                    ColumnProfile(name="product_id", data_type="VARCHAR", is_primary_key=True),
                    ColumnProfile(name="category", data_type="VARCHAR"),
                ],
            ),
            TableSchema(
                name="customers",
                table_role=TableRole.DIMENSION,
                row_count=3,
                columns=[
                    ColumnProfile(name="customer_id", data_type="VARCHAR", is_primary_key=True),
                    ColumnProfile(name="region_code", data_type="VARCHAR"),
                    ColumnProfile(name="cust_tier", data_type="VARCHAR"),
                ],
            ),
        ],
        relationships=[
            RelationshipEdge(
                id="rel_orders_products",
                source_table="orders",
                source_column="product_id",
                target_table="products",
                target_column="product_id",
            ),
            RelationshipEdge(
                id="rel_orders_customers",
                source_table="orders",
                source_column="customer_id",
                target_table="customers",
                target_column="customer_id",
            ),
        ],
    )

    connector = DuckDBConnector(workspace_dir=tmp_path)
    connector._conn = conn
    return workspace, connector


def test_discover_options(mock_duckdb_retail):
    workspace, connector = mock_duckdb_retail
    options = RootCauseAgent.discover_options(workspace, connector)

    assert options["fact_table"] == "orders"
    assert options["has_temporal"] is True
    metric_cols = [m["column"] for m in options["metrics"]]
    assert "rev_net_usd" in metric_cols
    assert "profit_usd" in metric_cols

    dim_ids = [d["id"] for d in options["dimensions"]]
    assert "products.category" in dim_ids or "orders.sales_channel" in dim_ids
    assert len(options["algorithms"]) == 3


def test_additive_variance_exact_reconciliation(mock_duckdb_retail):
    workspace, connector = mock_duckdb_retail
    res = RootCauseAgent.analyze_variance(
        workspace=workspace,
        connector=connector,
        metric_col="rev_net_usd",
        dimension="products.category",
        algorithm="additive",
        window_type="yoy",
    )

    assert "summary" in res
    assert res["total_prior"] == 3500.0  # 1000 + 500 + 800 + 1200
    assert res["total_curr"] == 4800.0   # 2500 + 600 + 200 + 1500 (O9 cancelled excluded)
    assert res["total_delta"] == 1300.0

    df = res["full_df"]
    assert not df.empty
    # Sum of variance deltas must equal total_delta
    sum_deltas = round(float(df["variance_delta"].sum()), 2)
    assert sum_deltas == res["total_delta"]

    # Verify audit receipt
    receipt = res["audit_receipt"]
    assert receipt["balance_check"]["status"] == "VERIFIED_EXACT_BALANCE"
    assert receipt["balance_check"]["unreconciled_residual"] == 0.0
    assert "executed_sql" in receipt


def test_simpsons_paradox_rate_mix_decomposition(tmp_path):
    """Verify Rate vs. Mix decomposition and Simpson's Paradox detection."""
    conn = duckdb.connect(":memory:")
    # Synthetic dataset where conversion in Mobile and Desktop both improved,
    # but overall rate dropped due to traffic surge in lower-converting mobile
    conn.execute("""
        CREATE TABLE sessions (
            session_id VARCHAR PRIMARY KEY,
            device VARCHAR,
            session_date DATE,
            conversions INT,
            visitors INT
        );
        INSERT INTO sessions VALUES
            -- 2024: 50 Desktop (4 conv = 8%), 50 Mobile (1 conv = 2%). Total: 5/100 = 5.0%
            ('S1', 'Desktop', '2024-05-01', 4, 50),
            ('S2', 'Mobile',  '2024-05-01', 1, 50),
            -- 2025: Desktop improved to 9% (9/100), Mobile improved to 3% (9/300).
            -- Total: 18 / 400 = 4.5% (Overall dropped from 5.0% to 4.5% despite both rates improving!)
            ('S3', 'Desktop', '2025-05-01', 9, 100),
            ('S4', 'Mobile',  '2025-05-01', 9, 300);
    """)

    workspace = OKFWorkspace(
        workspace_id="test_simpson",
        workspace_name="Simpson Test",
        tables=[
            TableSchema(
                name="sessions",
                table_role=TableRole.FACT,
                row_count=4,
                columns=[
                    ColumnProfile(name="session_id", data_type="VARCHAR", is_primary_key=True),
                    ColumnProfile(name="device", data_type="VARCHAR"),
                    ColumnProfile(name="session_date", data_type="DATE"),
                    ColumnProfile(name="conversions", data_type="INT"),
                    ColumnProfile(name="visitors", data_type="INT"),
                ],
            )
        ],
    )

    connector = DuckDBConnector(workspace_dir=tmp_path)
    connector._conn = conn

    res = RootCauseAgent.analyze_variance(
        workspace=workspace,
        connector=connector,
        metric_col="conversions",
        dimension="sessions.device",
        algorithm="rate_mix",
        window_type="yoy",
    )

    assert "summary" in res
    receipt = res["audit_receipt"]
    # Das Gupta decomposition must close with zero residual
    assert receipt["balance_check"]["status"] == "EXACT_RECONCILIATION_VERIFIED"
    assert receipt["balance_check"]["unexplained_residual"] <= 0.001
    # Simpson's Paradox must be detected!
    assert receipt["simpsons_paradox_flag"] is True
    assert "Simpson's Paradox Detected" in res["summary"]


def test_auto_scan_dimensions(mock_duckdb_retail):
    workspace, connector = mock_duckdb_retail
    res = RootCauseAgent.analyze_variance(
        workspace=workspace,
        connector=connector,
        metric_col="rev_net_usd",
        algorithm="auto_scan",
        window_type="yoy",
    )

    assert "summary" in res
    assert res["chart_type"] == "dimension_divergence"
    df = res["df"]
    assert not df.empty
    assert "explanatory_score" in df.columns
    assert "peak_driver" in df.columns


def test_non_temporal_cohort_fallback(tmp_path):
    """Verify graceful fallback when dataset has zero date columns."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE faculty (
            faculty_id VARCHAR PRIMARY KEY,
            department VARCHAR,
            salary DOUBLE
        );
        INSERT INTO faculty VALUES
            ('F1', 'Engineering', 95000.0),
            ('F2', 'Engineering', 105000.0),
            ('F3', 'Arts', 65000.0),
            ('F4', 'Arts', 70000.0);
    """)

    workspace = OKFWorkspace(
        workspace_id="test_faculty",
        workspace_name="Faculty Roster",
        tables=[
            TableSchema(
                name="faculty",
                table_role=TableRole.FACT,
                row_count=4,
                columns=[
                    ColumnProfile(name="faculty_id", data_type="VARCHAR", is_primary_key=True),
                    ColumnProfile(name="department", data_type="VARCHAR"),
                    ColumnProfile(name="salary", data_type="DOUBLE"),
                ],
            )
        ],
    )

    connector = DuckDBConnector(workspace_dir=tmp_path)
    connector._conn = conn

    options = RootCauseAgent.discover_options(workspace, connector)
    assert options["has_temporal"] is False
    assert options["temporal_options"][0]["id"] == "cohort"

    res = RootCauseAgent.analyze_variance(
        workspace=workspace,
        connector=connector,
        metric_col="salary",
        dimension="faculty.department",
        algorithm="additive",
    )
    assert not res["df"].empty


def test_custom_date_range_exact_variance(mock_duckdb_retail):
    """Test custom date ranges (t0_start/end vs t1_start/end) produce exact mathematical reconciliation."""
    workspace, connector = mock_duckdb_retail
    res = RootCauseAgent.analyze_variance(
        workspace=workspace,
        connector=connector,
        metric_col="rev_net_usd",
        dimension="products.category",
        algorithm="additive",
        window_type="custom",
        t0_start="2024-05-01",
        t0_end="2024-08-31",
        t1_start="2025-05-01",
        t1_end="2025-08-31",
    )
    assert not res["df"].empty
    audit = res["audit_receipt"]
    assert audit["balance_check"]["status"] == "VERIFIED_EXACT_BALANCE"
    assert audit["balance_check"]["unreconciled_residual"] == 0.0
    assert "2024-05-01 to 2024-08-31" in audit["window_t0"]
    assert "2025-05-01 to 2025-08-31" in audit["window_t1"]


def test_chart_switcher_compatibility_on_diagnostic_data(mock_duckdb_retail):
    """Test that ChartRecommender can dynamically switch diagnostic data to any visual type."""
    from src.viz.chart_recommender import ChartRecommender

    workspace, connector = mock_duckdb_retail
    res = RootCauseAgent.analyze_variance(
        workspace=workspace,
        connector=connector,
        metric_col="rev_net_usd",
        dimension="products.category",
        algorithm="additive",
    )
    df = res["df"]
    assert not df.empty

    for ct in ("waterfall", "bar", "horizontal_bar", "table"):
        fig = ChartRecommender.build_figure(df, chart_type=ct, title="Diagnostic Variance")
        if ct != "table":
            assert fig is not None, f"Chart type {ct} failed to build figure"


def test_api_root_cause_chat_stream_injection(mock_duckdb_retail, monkeypatch):
    """Test that POST /api/root-cause records the diagnostic turn into chat_history with full capabilities."""
    from src.api.routes.dashboard import run_root_cause_analysis
    from src.api.runtime import AppRuntime, SessionState
    from src.api.schemas import RootCauseRequest

    workspace, connector = mock_duckdb_retail
    runtime = AppRuntime()
    session = SessionState(session_id="test_sess")
    session.active_workspace_id = workspace.workspace_id

    # Mock runtime retrieval for test workspace
    monkeypatch.setattr(runtime, "get_workspace", lambda wid: workspace)
    monkeypatch.setattr(runtime, "get_connector", lambda wid: connector)

    req = RootCauseRequest(
        metric_col="rev_net_usd",
        dimension="products.category",
        algorithm="additive",
        window_type="yoy",
    )

    result = run_root_cause_analysis(req, runtime, session)
    assert "entry" in result
    assert "state" in result
    assert len(session.chat_history[workspace.workspace_id]) == 1

    entry = session.chat_history[workspace.workspace_id][0]
    assert entry["chart_type"] == "waterfall"
    assert entry["plotly_spec"] is not None
    assert entry["response"]["is_diagnostic"] is True
    assert entry["response"]["compiler_mode"] == "deterministic"
    assert entry["response"]["executed_query"] != ""
    assert entry["response"]["compiled_dax_equivalent"] != ""
    assert len(entry["response"]["suggested_follow_ups"]) > 0


def test_chart_thread_and_hybrid_response_modes(mock_duckdb_retail, monkeypatch):
    """Test chart-scoped Q&A threads (in-memory chart analysis, text-only vs with-chart, and warehouse SQL fallback)."""
    from src.api.routes.chat import run_chart_thread_query, run_chat_query
    from src.api.routes.dashboard import run_root_cause_analysis
    from src.api.runtime import AppRuntime, SessionState
    from src.api.schemas import ChartThreadQueryRequest, ChatQueryRequest, RootCauseRequest

    workspace, connector = mock_duckdb_retail
    runtime = AppRuntime()
    session = SessionState(session_id="test_thread_sess")
    session.active_workspace_id = workspace.workspace_id

    monkeypatch.setattr(runtime, "get_workspace", lambda wid: workspace)
    monkeypatch.setattr(runtime, "get_connector", lambda wid: connector)

    # 1. Main chat query in 'text_only' mode -> hides visual chart by default
    q_req = ChatQueryRequest(
        question="Show revenue by category",
        response_mode="text_only",
    )
    q_res = run_chat_query(q_req, runtime, session)
    entry0 = q_res["entry"]
    assert entry0["hide_chart"] is True
    assert entry0["chart_type"] == "none"
    assert entry0["plotly_spec"] is None
    assert len(entry0["rows"]) >= 2

    # 2. Ask a chart thread question in default 'text_only' mode (in-memory comparison)
    t_req1 = ChartThreadQueryRequest(
        entry_index=0,
        question="Compare the top 2 segments and their share of total",
        response_mode="text_only",
    )
    t_res1 = run_chart_thread_query(t_req1, runtime, session)
    turn1 = t_res1["turn"]
    assert turn1["source_mode"] == "chart_data"
    assert turn1["include_chart"] is False
    assert turn1["plotly_spec"] is None
    assert "Comparison:" in turn1["answer"] or "leading" in turn1["answer"]

    # 3. Ask an above-average question in 'with_chart' mode -> renders a sub-chart inside the thread
    t_req2 = ChartThreadQueryRequest(
        entry_index=0,
        question="Which segments are above average?",
        response_mode="with_chart",
    )
    t_res2 = run_chart_thread_query(t_req2, runtime, session)
    turn2 = t_res2["turn"]
    assert turn2["source_mode"] == "chart_data"
    assert turn2["include_chart"] is True
    assert turn2["plotly_spec"] is not None

    # 4. Ask a question requiring a warehouse SQL breakdown by another column ('region')
    t_req3 = ChartThreadQueryRequest(
        entry_index=0,
        question="Break this down by region in the warehouse",
        response_mode="auto",
    )
    t_res3 = run_chart_thread_query(t_req3, runtime, session)
    turn3 = t_res3["turn"]
    assert turn3["source_mode"] == "sql_query"
    assert turn3["executed_query"] != ""

    # 5. Also verify chart threads work seamlessly on Root-Cause Analysis cards
    rc_req = RootCauseRequest(
        metric_col="rev_net_usd",
        dimension="products.category",
        algorithm="additive",
        window_type="yoy",
    )
    run_root_cause_analysis(rc_req, runtime, session)
    rc_thread_req = ChartThreadQueryRequest(
        entry_index=1,
        question="Summarize the key takeaways from this chart",
        response_mode="text_only",
    )
    rc_thread_res = run_chart_thread_query(rc_thread_req, runtime, session)
    assert rc_thread_res["turn"]["source_mode"] == "chart_data"
    assert len(session.chat_history[workspace.workspace_id][1]["thread"]) == 1


