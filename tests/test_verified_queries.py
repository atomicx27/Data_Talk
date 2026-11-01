"""Automated test suite for Verified Query Memory (Golden Query Cache) in Data_Talk."""

from __future__ import annotations

import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.agents.query_agent import QueryAgent
from src.api import create_app
from src.api.runtime import AppRuntime
from src.connectors.duckdb_connector import DuckDBConnector
from src.knowledge.okf_models import OKFWorkspace, TableSchema, ColumnProfile, TableRole, VerifiedQuery
from src.knowledge.okf_store import OKFStore
from src.knowledge.rag_store import RAGStore


@pytest.fixture
def test_env(tmp_path):
    ws_dir = tmp_path / "workspaces"
    ws_dir.mkdir()
    okf = OKFStore(base_dir=ws_dir)
    ws_store_dir = okf.workspace_dir("test_ws")
    rag = RAGStore(workspace_dir=ws_store_dir)
    conn = DuckDBConnector(workspace_dir=ws_store_dir)
    conn.load_builtin_demo_dataset()

    raw_tables, explicit_rels, _ = conn.extract_schema()
    ws = OKFWorkspace(
        workspace_id="test_ws",
        workspace_name="Test Workspace",
        tables=raw_tables,
        relationships=explicit_rels,
        source_type="duckdb",
        dialect="duckdb",
    )
    okf.save(ws)

    return {
        "tmp_path": tmp_path,
        "okf": okf,
        "rag": rag,
        "conn": conn,
        "ws": ws,
    }


def test_verified_query_matching(test_env):
    ws = test_env["ws"]
    vq = VerifiedQuery(
        id="vq_1",
        question_pattern="Show total sales by product category",
        executed_query="SELECT category, SUM(rev_net_usd) AS total_sales FROM orders GROUP BY category",
        query_dialect="duckdb",
        chart_type="bar",
    )
    ws.verified_queries.append(vq)

    # Exact match (normalized)
    match = QueryAgent._find_matching_verified_query(
        "show total sales by product category", ws
    )
    assert match is not None
    assert match.id == "vq_1"

    # Close fuzzy match (punctuation and slight case change)
    match_fuzzy = QueryAgent._find_matching_verified_query(
        "Show total sales by product category?", ws
    )
    assert match_fuzzy is not None
    assert match_fuzzy.id == "vq_1"

    # Completely different question should not match
    match_diff = QueryAgent._find_matching_verified_query(
        "What is the average student age in college?", ws
    )
    assert match_diff is None


def test_verified_query_execution_hit(test_env):
    ws = test_env["ws"]
    okf = test_env["okf"]
    rag = test_env["rag"]
    conn = test_env["conn"]

    vq = VerifiedQuery(
        id="vq_1",
        question_pattern="Show total sales by order status",
        executed_query="SELECT ord_stat, SUM(rev_net_usd) AS total_sales FROM orders GROUP BY ord_stat",
        query_dialect="duckdb",
        chart_type="bar",
    )
    ws.verified_queries.append(vq)
    okf.save(ws)

    # 1. Answer question -> should hit verified memory
    resp, df = QueryAgent.answer_question(
        question="Show total sales by order status",
        workspace=ws,
        okf_store=okf,
        rag_store=rag,
        connector=conn,
    )

    assert resp.success is True
    assert resp.compiler_mode == "verified_memory"
    assert resp.from_verified_memory is True
    assert resp.verified_query_id == "vq_1"
    assert len(df) > 0
    assert "ord_stat" in df.columns
    assert vq.execution_count >= 1

    # 2. Force recompile -> should bypass memory
    resp_recompile, df_recompile = QueryAgent.answer_question(
        question="Show total sales by order status",
        workspace=ws,
        okf_store=okf,
        rag_store=rag,
        connector=conn,
        force_recompile=True,
    )
    assert resp_recompile.compiler_mode != "verified_memory"
    assert resp_recompile.from_verified_memory is False


def test_verified_queries_api_endpoints(test_env):
    import server
    from src.api.deps import get_runtime

    tmp_path = test_env["tmp_path"]
    runtime = AppRuntime(base_dir=tmp_path, default_workspace_id="test_ws")
    runtime.okf_store = test_env["okf"]
    runtime.connectors["test_ws"] = test_env["conn"]
    runtime.rag_stores["test_ws"] = test_env["rag"]

    server.runtime = runtime
    app = create_app(runtime=runtime)
    app.dependency_overrides[get_runtime] = lambda: runtime
    client = TestClient(app)

    # 1. List initially empty
    res = client.get("/api/verified-queries")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 2. Create verified query
    create_payload = {
        "question_pattern": "Show total sales by order status",
        "executed_query": "SELECT ord_stat, SUM(rev_net_usd) AS total_sales FROM orders GROUP BY ord_stat",
        "dax_equivalent": "EVALUATE SUMMARIZECOLUMNS(orders[ord_stat], \"total_sales\", SUM(orders[rev_net_usd]))",
        "chart_type": "bar",
    }
    create_res = client.post("/api/verified-queries", json=create_payload)
    assert create_res.status_code == 200
    created_vq = create_res.json()["saved_query"]
    assert created_vq["question_pattern"] == create_payload["question_pattern"]
    vq_id = created_vq["id"]

    # 3. Test execution endpoint
    test_res = client.post(f"/api/verified-queries/{vq_id}/test")
    assert test_res.status_code == 200
    test_data = test_res.json()
    assert test_data["success"] is True
    assert test_data["row_count"] > 0

    # 4. Update verified query
    update_res = client.put(
        f"/api/verified-queries/{vq_id}",
        json={"question_pattern": "Show all order status sales", "chart_type": "pie"},
    )
    assert update_res.status_code == 200
    assert update_res.json()["updated_query"]["question_pattern"] == "Show all order status sales"
    assert update_res.json()["updated_query"]["chart_type"] == "pie"

    # 5. Delete verified query
    del_res = client.delete(f"/api/verified-queries/{vq_id}")
    assert del_res.status_code == 200

    list_res = client.get("/api/verified-queries")
    assert not any(q["id"] == vq_id for q in list_res.json())
