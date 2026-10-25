"""Tests for Conversational UX, Multi-Turn Context, Interactive Visuals, and Domain Knowledge Enrichment."""

from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from src.agents.knowledge_enricher import BusinessKnowledgeEnricher
from src.agents.query_agent import QueryAgent
from src.connectors.duckdb_connector import DuckDBConnector
from src.connectors.profiler import StatisticalProfiler
from src.knowledge.join_inference import JoinInferenceEngine
from src.knowledge.okf_models import OKFWorkspace, TableRole
from src.knowledge.okf_store import OKFStore
from src.knowledge.rag_store import RAGStore


@pytest.fixture
def enriched_demo_workspace():
    """Sets up an OKF workspace with demo dataset, profiles it, and runs business knowledge enricher."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_dir = Path(tmpdir)
        okf_store = OKFStore(base_dir=base_dir / "workspaces")
        ws_dir = okf_store.workspace_dir("enrich_test_ws")

        conn = DuckDBConnector(workspace_dir=ws_dir)
        conn.load_builtin_demo_dataset()

        raw_tables, explicit_rels, imported_measures = conn.extract_schema()
        profiled_tables = StatisticalProfiler.profile_tables(conn, raw_tables)

        ws = OKFWorkspace(
            workspace_id="enrich_test_ws",
            workspace_name="Enriched Test BI",
            source_type="duckdb",
            dialect="duckdb",
            tables=profiled_tables,
            relationships=explicit_rels,
            measures=imported_measures,
        )
        ws.relationships = JoinInferenceEngine.infer_relationships(
            profiled_tables,
            existing_relationships=explicit_rels,
            connector=conn,
        )
        okf_store.classify_table_roles(ws)
        rag_store = RAGStore(workspace_dir=ws_dir)

        # Run auto enricher
        report = BusinessKnowledgeEnricher.auto_enrich_workspace(ws, rag_store=rag_store)
        okf_store.save(ws)

        yield ws, okf_store, rag_store, conn, report


def test_domain_detection_and_enrichment(enriched_demo_workspace):
    ws, okf_store, rag_store, conn, report = enriched_demo_workspace

    # 1. Verify Domain Detection
    domain_info = BusinessKnowledgeEnricher.detect_domain(ws)
    assert domain_info["domain_key"] == "ecommerce"
    assert "Retail" in domain_info["domain_title"]
    assert domain_info["confidence"] >= 0.7

    # 2. Verify Table Business Semantics
    orders_tbl = ws.get_table("orders")
    assert orders_tbl is not None
    assert orders_tbl.business_name == "Customer Orders & Sales Transactions"
    assert "status ('C'=Completed, 'X'=Cancelled)" in orders_tbl.description

    # 3. Verify Column Business Aliases & Status Code Deciphering
    stat_col = orders_tbl.get_column("ord_stat")
    assert stat_col is not None
    assert stat_col.business_alias == "Order Status Code"
    assert stat_col.value_meanings.get("C") == "Completed"
    assert stat_col.value_meanings.get("X") == "Cancelled"

    rev_col = orders_tbl.get_column("rev_net_usd")
    assert rev_col is not None
    assert rev_col.business_alias == "Net Recognized Revenue (USD)"

    # 4. Verify Standard Measures & Rules Added
    rule_terms = [r.term for r in ws.business_rules]
    assert "Exclude Cancelled Orders" in rule_terms

    meas_names = [m.name for m in ws.measures]
    assert "Net Revenue" in meas_names
    assert "Average Order Value (AOV)" in meas_names

    # 5. Verify RAG Indexing
    rag_counts = rag_store.count_by_collection()
    assert rag_counts["schema_glossary"] >= 4
    assert rag_counts["business_rules"] >= 2


def test_multi_turn_conversational_follow_up(enriched_demo_workspace):
    ws, okf_store, rag_store, conn, _ = enriched_demo_workspace

    # Turn 1: Initial broad query
    q1 = "Compare net revenue by product category"
    resp1, df1 = QueryAgent.answer_question(
        question=q1,
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=conn,
    )
    assert resp1.success is True
    assert len(df1) >= 2
    assert resp1.is_follow_up is False
    assert len(resp1.suggested_follow_ups) == 3
    assert len(resp1.drill_down_dimensions) >= 1

    chat_history = [
        {
            "id": "msg_1",
            "response": resp1.model_dump(mode="json"),
            "rows": df1.head(10).to_dict(orient="records"),
            "columns": list(df1.columns),
        }
    ]

    # Turn 2: Follow-up refinement with filter
    q2 = "filter that to region US"
    resp2, df2 = QueryAgent.answer_question(
        question=q2,
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=conn,
        conversation_history=chat_history,
    )
    assert resp2.success is True
    assert resp2.is_follow_up is True
    assert "Refined" in str(resp2.follow_up_context)
    assert "customers.region_code = 'US'" in resp2.executed_query
    assert "category" in resp2.executed_query

    chat_history.append({
        "id": "msg_2",
        "response": resp2.model_dump(mode="json"),
        "rows": df2.head(10).to_dict(orient="records"),
        "columns": list(df2.columns),
    })

    # Turn 3: Follow-up refinement with top ranking
    q3 = "show top 2"
    resp3, df3 = QueryAgent.answer_question(
        question=q3,
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=conn,
        conversation_history=chat_history,
    )
    assert resp3.success is True
    assert resp3.is_follow_up is True
    assert "LIMIT 2" in resp3.executed_query
    assert len(df3) <= 2


def test_interactive_drill_down_query(enriched_demo_workspace):
    ws, okf_store, rag_store, conn, _ = enriched_demo_workspace

    # Explicit Drill-down into Cloud Suite
    q = "Drill down into Cloud Suite by region"
    resp, df = QueryAgent.answer_question(
        question=q,
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=conn,
    )
    assert resp.success is True
    assert len(df) >= 1
    assert "region" in resp.executed_query.lower()
    assert "cloud" in resp.executed_query.lower()
    # Check that both customers and products were joined
    assert "JOIN customers" in resp.executed_query
    assert "JOIN products" in resp.executed_query


def test_api_auto_enrich_and_domain_routes(test_client: TestClient):
    # 1. Load Demo
    demo_res = test_client.post("/api/connect/demo")
    assert demo_res.status_code == 200

    # 2. Check /api/interview/domain
    domain_res = test_client.get("/api/interview/domain")
    assert domain_res.status_code == 200
    d_data = domain_res.json()
    assert d_data["domain_key"] == "ecommerce"

    # 3. Call /api/interview/auto-enrich
    enrich_res = test_client.post("/api/interview/auto-enrich")
    assert enrich_res.status_code == 200
    payload = enrich_res.json()
    assert "report" in payload
    assert payload["report"]["domain"]["domain_key"] == "ecommerce"
    assert payload["state"]["workspace"]["domain_tag"] == "ecommerce"

    # 4. Run conversational query through API with follow-up
    q1_res = test_client.post(
        "/api/chat/query",
        json={"question": "What is our revenue by product category?"},
    )
    assert q1_res.status_code == 200
    entry1 = q1_res.json()["entry"]
    assert entry1["response"]["is_follow_up"] is False
    assert len(entry1["response"]["suggested_follow_ups"]) == 3

    # Send conversational follow-up
    q2_res = test_client.post(
        "/api/chat/query",
        json={"question": "filter that to region US"},
    )
    assert q2_res.status_code == 200
    entry2 = q2_res.json()["entry"]
    assert entry2["response"]["is_follow_up"] is True
    assert "region_code = 'NA'" in entry2["response"]["executed_query"] or "region_code = 'US'" in entry2["response"]["executed_query"]


def test_human_domain_override_and_rag_sync(test_client: TestClient):
    """Business owner can alter domain classification and taxonomy."""
    test_client.post("/api/connect/demo")

    # 1. Custom Domain Override
    override_res = test_client.post(
        "/api/interview/domain-override",
        json={
            "domain_title": "Enterprise Cloud & Cyber Security",
            "domain_tag": "cybersec",
            "industry": "Information Security & SaaS",
        },
    )
    assert override_res.status_code == 200
    state = override_res.json()
    assert state["workspace"]["domain_title"] == "Enterprise Cloud & Cyber Security"
    assert state["workspace"]["domain_tag"] == "cybersec"

    # 2. Verify domain route respects human override
    domain_res = test_client.get("/api/interview/domain")
    assert domain_res.status_code == 200
    d_data = domain_res.json()
    assert d_data["domain_title"] == "Enterprise Cloud & Cyber Security"
    assert d_data["domain_key"] == "cybersec"


def test_human_table_and_column_semantics_customization(test_client: TestClient):
    """Business owner can alter table names, descriptions, column aliases, and code decoders."""
    test_client.post("/api/connect/demo")

    # 1. Update Table Semantics
    t_res = test_client.post(
        "/api/schema/table-config",
        json={
            "table_name": "orders",
            "business_name": "Global Bookings & Invoicing Ledger",
            "description": "Master transaction record for all booked software contracts.",
            "table_role": "fact",
        },
    )
    assert t_res.status_code == 200
    state = t_res.json()
    orders_tbl = next((t for t in state["workspace"]["tables"] if t["name"] == "orders"), None)
    assert orders_tbl is not None
    assert orders_tbl["business_name"] == "Global Bookings & Invoicing Ledger"
    assert "software contracts" in orders_tbl["description"]

    # 2. Update Column Semantics (alias + status code decoders)
    c_res = test_client.post(
        "/api/schema/column-config",
        json={
            "table_name": "orders",
            "column_name": "ord_stat",
            "business_alias": "Fulfillment State",
            "description": "Lifecycle state of the booking",
            "value_meanings": {
                "C": "Closed Won",
                "X": "Closed Lost",
                "P": "Pending Review",
            },
        },
    )
    assert c_res.status_code == 200
    state2 = c_res.json()
    orders_tbl2 = next((t for t in state2["workspace"]["tables"] if t["name"] == "orders"), None)
    ord_stat_col = next((c for c in orders_tbl2["columns"] if c["name"] == "ord_stat"), None)
    assert ord_stat_col is not None
    assert ord_stat_col["business_alias"] == "Fulfillment State"
    assert ord_stat_col["value_meanings"]["C"] == "Closed Won"
    assert ord_stat_col["value_meanings"]["X"] == "Closed Lost"


def test_human_business_rule_edit_and_delete(test_client: TestClient):
    """Business owner can add, edit, and delete business governance rules."""
    test_client.post("/api/connect/demo")

    # 1. Add Custom Rule
    add_res = test_client.post(
        "/api/interview/add-rule",
        json={
            "term": "Enterprise High Value",
            "definition": "Transactions exceeding $50k in recognized net revenue",
            "sql_filter_hint": "orders.rev_net_usd > 50000",
        },
    )
    assert add_res.status_code == 200
    rules = add_res.json()["workspace"]["business_rules"]
    target_rule = next((r for r in rules if r["term"] == "Enterprise High Value"), None)
    assert target_rule is not None
    rule_id = target_rule["id"]

    # 2. Edit the Rule
    edit_res = test_client.put(
        f"/api/interview/rule/{rule_id}",
        json={
            "term": "Strategic Tier Transactions",
            "definition": "Key strategic accounts exceeding $75k",
            "sql_filter_hint": "orders.rev_net_usd > 75000",
        },
    )
    assert edit_res.status_code == 200
    updated_rules = edit_res.json()["workspace"]["business_rules"]
    updated_rule = next((r for r in updated_rules if r["id"] == rule_id), None)
    assert updated_rule is not None
    assert updated_rule["term"] == "Strategic Tier Transactions"
    assert "75000" in updated_rule["sql_filter_hint"]

    # 3. Delete the Rule
    del_res = test_client.delete(f"/api/interview/rule/{rule_id}")
    assert del_res.status_code == 200
    final_rules = del_res.json()["workspace"]["business_rules"]
    assert not any(r["id"] == rule_id for r in final_rules)
