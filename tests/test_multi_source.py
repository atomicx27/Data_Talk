"""Test suite for Multi-Source Connections and Cross-Database Relationships."""

from pathlib import Path
import sqlite3
import tempfile
import pytest

from src.agents.interviewer_agent import InterviewerAgent
from src.agents.query_agent import QueryAgent
from src.connectors.federated_connector import FederatedConnector
from src.connectors.profiler import StatisticalProfiler
from src.knowledge.join_inference import JoinInferenceEngine
from src.knowledge.okf_models import DataSourceConfig, JoinStatus, OKFWorkspace
from src.knowledge.okf_store import OKFStore
from src.knowledge.rag_store import RAGStore


@pytest.fixture
def multi_source_env():
    """Create two independent SQLite databases and an isolated test workspace."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        db1_path = tmp_path / "sales.sqlite"
        db2_path = tmp_path / "crm.sqlite"

        # Database 1: Sales DB (orders table)
        c1 = sqlite3.connect(str(db1_path))
        c1.execute(
            "CREATE TABLE orders (order_id INT PRIMARY KEY, customer_id INT, amount FLOAT, ord_stat TEXT)"
        )
        c1.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            [
                (1, 101, 250.0, "C"),
                (2, 102, 450.0, "C"),
                (3, 101, 120.0, "P"),
                (4, 103, 300.0, "X"),
            ],
        )
        c1.commit()
        c1.close()

        # Database 2: CRM DB (customers table)
        c2 = sqlite3.connect(str(db2_path))
        c2.execute(
            "CREATE TABLE customers (customer_id INT PRIMARY KEY, company_name TEXT, cust_tier TEXT)"
        )
        c2.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [
                (101, "Acme Corp", "ENT"),
                (102, "Beta LLC", "MM"),
                (103, "Gamma Tech", "SMB"),
            ],
        )
        c2.commit()
        c2.close()

        ws_dir = tmp_path / "workspaces" / "test_multi_ws"
        ws_dir.mkdir(parents=True, exist_ok=True)
        okf_store = OKFStore(base_dir=tmp_path / "workspaces")
        rag_store = RAGStore(workspace_dir=ws_dir)

        fed_conn = FederatedConnector(workspace_dir=ws_dir)

        yield {
            "tmp_path": tmp_path,
            "db1_url": f"sqlite:///{db1_path}",
            "db2_url": f"sqlite:///{db2_path}",
            "fed_conn": fed_conn,
            "ws_dir": ws_dir,
            "okf_store": okf_store,
            "rag_store": rag_store,
        }


def test_multi_source_connection_and_profiling(multi_source_env):
    """Verifies that multiple SQL databases can be connected, profiled push-down, and federated into DuckDB."""
    fed_conn: FederatedConnector = multi_source_env["fed_conn"]
    db1_url = multi_source_env["db1_url"]
    db2_url = multi_source_env["db2_url"]

    # 1. Connect Source 1 (Sales DB)
    sql_conn1 = fed_conn.register_sql_connector("sales_db", db1_url, dialect_hint="sqlite")
    raw_t1, rels1, _ = sql_conn1.extract_schema()
    prof_t1 = StatisticalProfiler.profile_tables(sql_conn1, raw_t1)
    for t in prof_t1:
        t.source_id = "sales_db"
        t.source_name = "Sales Database"
    fed_conn.sync_sql_source_to_duckdb("sales_db", [t.name for t in prof_t1])

    # 2. Connect Source 2 (CRM DB)
    sql_conn2 = fed_conn.register_sql_connector("crm_db", db2_url, dialect_hint="sqlite")
    raw_t2, rels2, _ = sql_conn2.extract_schema()
    prof_t2 = StatisticalProfiler.profile_tables(sql_conn2, raw_t2)
    for t in prof_t2:
        t.source_id = "crm_db"
        t.source_name = "CRM Database"
    fed_conn.sync_sql_source_to_duckdb("crm_db", [t.name for t in prof_t2])

    workspace = OKFWorkspace(
        workspace_id="test_multi_ws",
        workspace_name="Multi Source Workspace",
        source_type="federated",
        dialect="duckdb",
        data_sources=[
            DataSourceConfig(id="sales_db", name="Sales Database", table_names=["orders"]),
            DataSourceConfig(id="crm_db", name="CRM Database", table_names=["customers"]),
        ],
        tables=prof_t1 + prof_t2,
    )

    assert len(workspace.data_sources) == 2
    assert len(workspace.tables) == 2
    assert workspace.tables[0].source_id == "sales_db"
    assert workspace.tables[1].source_id == "crm_db"

    # Verify both databases are queryable inside federated DuckDB
    df_orders = fed_conn.execute_query("SELECT * FROM sales_db.orders")
    assert len(df_orders) == 4
    df_customers = fed_conn.execute_query("SELECT * FROM crm_db.customers")
    assert len(df_customers) == 3


def test_cross_database_join_inference_and_query(multi_source_env):
    """Verifies that 3-signal join inference discovers the common link between two SQL databases and executes joins."""
    fed_conn: FederatedConnector = multi_source_env["fed_conn"]
    db1_url = multi_source_env["db1_url"]
    db2_url = multi_source_env["db2_url"]
    okf_store: OKFStore = multi_source_env["okf_store"]
    rag_store: RAGStore = multi_source_env["rag_store"]

    # Register both sources and sync
    s1 = fed_conn.register_sql_connector("sales_db", db1_url, dialect_hint="sqlite")
    t1, _, _ = s1.extract_schema()
    pt1 = StatisticalProfiler.profile_tables(s1, t1)
    for t in pt1:
        t.source_id = "sales_db"
        t.source_name = "Sales DB"
    fed_conn.sync_sql_source_to_duckdb("sales_db", [t.name for t in pt1])

    s2 = fed_conn.register_sql_connector("crm_db", db2_url, dialect_hint="sqlite")
    t2, _, _ = s2.extract_schema()
    pt2 = StatisticalProfiler.profile_tables(s2, t2)
    for t in pt2:
        t.source_id = "crm_db"
        t.source_name = "CRM DB"
    fed_conn.sync_sql_source_to_duckdb("crm_db", [t.name for t in pt2])

    workspace = OKFWorkspace(
        workspace_id="test_multi_ws",
        workspace_name="Multi Source Workspace",
        source_type="federated",
        dialect="duckdb",
        data_sources=[
            DataSourceConfig(id="sales_db", name="Sales DB", table_names=["orders"]),
            DataSourceConfig(id="crm_db", name="CRM DB", table_names=["customers"]),
        ],
        tables=pt1 + pt2,
    )

    # Run Join Inference across all tables
    relationships = JoinInferenceEngine.infer_relationships(
        workspace.tables,
        connector=fed_conn,
    )
    workspace.relationships = relationships

    # There should be an inferred cross-database relationship: orders.customer_id -> customers.customer_id
    cross_rels = [r for r in relationships if r.is_cross_source]
    assert len(cross_rels) >= 1
    rel = cross_rels[0]
    assert rel.source_table == "orders"
    assert rel.source_column == "customer_id"
    assert rel.target_table == "customers"
    assert rel.target_column == "customer_id"
    assert rel.source_source_id == "sales_db"
    assert rel.target_source_id == "crm_db"

    # Pre-questions per database:
    questions = InterviewerAgent.generate_prioritized_questions(workspace)
    assert len(questions) > 0
    # Check that questions are tagged with source_id
    q_sources = {q.source_id for q in questions}
    assert "sales_db" in q_sources or "crm_db" in q_sources or "cross_database" in q_sources

    # Check cross-database link question
    cross_q = next((q for q in questions if q.id.startswith("q_cross_link_")), None)
    if cross_q:
        # User confirms the cross-database link:
        InterviewerAgent.record_answer(workspace, rag_store, cross_q.id, "Yes, connect sales orders to crm customers")
        assert rel.status == JoinStatus.APPROVED

    # Now execute a cross-database question using QueryAgent
    rel.status = JoinStatus.APPROVED
    okf_store.classify_table_roles(workspace)
    okf_store.save(workspace)

    resp, df = QueryAgent.answer_question(
        question="Show customer company name and order amount",
        workspace=workspace,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=fed_conn,
    )

    assert resp.success is True
    assert not df.empty
    # The result has columns from both sales_db and crm_db!
    cols = [c.lower() for c in df.columns]
    assert any("company" in c or "name" in c for c in cols)
    assert any("amount" in c or "order" in c for c in cols)


def test_api_multi_source_lifecycle(multi_source_env):
    """Verifies end-to-end API lifecycle: connect multiple SQL DBs, approve joins, query, and remove a source."""
    from fastapi.testclient import TestClient
    from src.api import create_app
    from src.api.runtime import AppRuntime
    from src.security.auth import AuthConfig, Authenticator

    tmp_path = multi_source_env["tmp_path"]
    db1_url = multi_source_env["db1_url"]
    db2_url = multi_source_env["db2_url"]

    import server
    from src.api.deps import set_global_authenticator, set_global_runtime

    runtime = AppRuntime(base_dir=tmp_path)
    auth = Authenticator(AuthConfig(api_token=None))
    server.runtime = runtime
    server.authenticator = auth
    set_global_runtime(runtime)
    set_global_authenticator(auth)

    app = create_app(runtime=runtime, auth_config=auth.config)
    client = TestClient(app)

    # Switch to isolated workspace for this test
    client.post("/api/settings", json={"workspace_id": "api_multi_ws"})

    # 1. Connect first database (Sales DB)
    res1 = client.post(
        "/api/connect/sql",
        json={
            "dialect": "sqlite",
            "credential_name": "sales_db_vault",
            "connection_url": db1_url,
            "source_id": "sales_db",
            "source_name": "Sales DB",
            "attach_mode": True,
        },
    )
    assert res1.status_code == 200
    state1 = res1.json()
    assert len(state1["workspace"]["data_sources"]) == 1
    assert any(t["name"] == "orders" for t in state1["workspace"]["tables"])

    # 2. Connect second database (CRM DB) in multi-source attach mode
    res2 = client.post(
        "/api/connect/sql",
        json={
            "dialect": "sqlite",
            "credential_name": "crm_db_vault",
            "connection_url": db2_url,
            "source_id": "crm_db",
            "source_name": "CRM DB",
            "attach_mode": True,
        },
    )
    assert res2.status_code == 200
    state2 = res2.json()
    assert len(state2["workspace"]["data_sources"]) == 2
    table_names = {t["name"] for t in state2["workspace"]["tables"]}
    assert "orders" in table_names
    assert "customers" in table_names

    # Check cross-database relationship inferred
    rels = state2["workspace"]["relationships"]
    cross = [r for r in rels if r.get("is_cross_source")]
    assert len(cross) >= 1

    # 3. Approve all joins
    res_app = client.post("/api/joins/approve-all")
    assert res_app.status_code == 200

    # 4. Chat query across both databases
    res_chat = client.post(
        "/api/chat/query",
        json={"question": "Show company name and order amount", "revenue_shock_pct": 0.0, "cost_shock_pct": 0.0},
    )
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert chat_data["entry"]["response"]["success"] is True
    assert len(chat_data["entry"]["rows"]) > 0

    # 5. Remove CRM DB source
    res_rem = client.post("/api/connect/remove-source", json={"source_id": "crm_db"})
    assert res_rem.status_code == 200
    state_rem = res_rem.json()
    assert len(state_rem["workspace"]["data_sources"]) == 1
    rem_table_names = {t["name"] for t in state_rem["workspace"]["tables"]}
    assert "orders" in rem_table_names
    assert "customers" not in rem_table_names
