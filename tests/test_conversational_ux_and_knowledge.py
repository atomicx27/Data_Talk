"""Tests for Conversational UX, Multi-Turn Context, Interactive Visuals, and Domain Knowledge Enrichment."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agents.knowledge_enricher import BusinessKnowledgeEnricher
from src.agents.query_agent import QueryAgent
from src.api.runtime import AppRuntime
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


def test_delete_entire_project_and_files(test_client: TestClient, test_runtime: AppRuntime):
    """Business owner can delete an entire project/workspace and all its files from disk."""
    ws_name = "test_to_be_deleted"

    # 1. Switch to new workspace and load demo data
    sw_res = test_client.post("/api/settings", json={"workspace_id": ws_name})
    assert sw_res.status_code == 200

    demo_res = test_client.post("/api/connect/demo")
    assert demo_res.status_code == 200

    # Verify files exist on disk in the runtime workspace directory
    ws_path = test_runtime.okf_store.workspace_dir(ws_name)
    assert ws_path.exists()
    assert (ws_path / "okf.yaml").exists()

    # 2. Call DELETE /api/workspace/{workspace_id}
    del_res = test_client.delete(f"/api/workspace/{ws_name}")
    assert del_res.status_code == 200
    state = del_res.json()

    # Workspace directory must be completely removed from disk
    assert not ws_path.exists()
    assert ws_name not in state["workspaces"]
    assert state["active_workspace_id"] != ws_name


def test_delete_project_post_body_and_traversal_defense(test_client: TestClient, test_runtime: AppRuntime):
    """Verify POST /api/workspace/delete and traversal rejection."""
    ws_name = "test_post_delete"
    test_client.post("/api/settings", json={"workspace_id": ws_name})
    ws_path = test_runtime.okf_store.workspace_dir(ws_name)
    assert ws_path.exists()

    # Delete via POST
    del_res = test_client.post("/api/workspace/delete", json={"workspace_id": ws_name})
    assert del_res.status_code == 200
    assert not ws_path.exists()

    # Reject path traversal
    bad_res = test_client.post("/api/workspace/delete", json={"workspace_id": "../bad"})
    assert bad_res.status_code == 400


def test_conversational_follow_up_missing_category_and_demographic_summary(enriched_demo_workspace):
    """Verify that critique follow-ups (e.g. 'you are not showing the other students') are resolved
    contextually and demographic executive summaries use neutral phrasing rather than 'Top Performer'.
    """
    from src.agents.query_agent import QueryAgent

    ws, okf_store, rag_store, conn, report = enriched_demo_workspace

    history = [
        {
            "response": {
                "success": True,
                "executed_query": "SELECT college_name, male_count, female_count, other_count FROM college_stats",
                "user_question": "show me gender ratio of students for each college",
                "involved_tables": ["college_stats"],
            }
        }
    ]

    # 1. Follow-up resolution
    resolved_q, is_follow_up, ctx, tables = QueryAgent._resolve_follow_up_context(
        question="you are not showing the other students",
        conversation_history=history,
        workspace=ws,
        llm_router=None,
    )
    assert is_follow_up is True
    assert "gender ratio of students for each college" in resolved_q
    assert "college_stats" in tables

    # 2. Executive summary tone for demographic / student query
    df_demo = pd.DataFrame({
        "College Name": ["College A", "College B", "College C"],
        "Male Count": [400, 250, 100],
    })
    summary = QueryAgent._build_executive_summary(
        question="show me gender ratio of students for each college",
        df=df_demo,
        matched_measures=[],
        applied_rules=[],
        llm_router=None,
        workspace=ws,
    )
    assert "Top Performer" not in summary
    assert "Highest Representation" in summary or "Highest Volume" in summary


def test_now_show_conversational_pivot_and_gender_wise_query(enriched_demo_workspace):
    """Verify that 'now show me the rio of total grades gender wise' pivots to the grades subject
    rather than being trapped into filtering the prior colleges query, and groups by gender with ratio.
    """
    from src.knowledge.okf_models import TableSchema, ColumnProfile, RelationshipEdge
    from src.knowledge.okf_store import JoinPlan

    ws, okf_store, rag_store, conn, report = enriched_demo_workspace

    history = [
        {
            "response": {
                "success": True,
                "executed_query": "SELECT college_name, male_count, female_count FROM college_stats",
                "user_question": "show students gender ritio for each college",
                "involved_tables": ["colleges", "students"],
            }
        }
    ]

    # 1. Follow-up resolution: should NOT prepend "show students gender ritio for each college filtered by..."
    resolved_q, is_follow_up, ctx, tables = QueryAgent._resolve_follow_up_context(
        question="now show me the rio of total grades gender wise",
        conversation_history=history,
        workspace=ws,
        llm_router=None,
    )
    # Must be recognized as an independent query pivot on grades & gender
    assert is_follow_up is False
    assert "college" not in resolved_q.lower()
    assert "grades" in resolved_q.lower()

    # 2. Add university-like tables (students, enrollments_marks) to workspace to test SQL compilation
    tbl_students = TableSchema(
        name="students",
        columns=[
            ColumnProfile(name="student_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="gender", data_type="VARCHAR"),
        ]
    )
    tbl_grades = TableSchema(
        name="enrollments_marks",
        columns=[
            ColumnProfile(name="enrollment_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="student_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="total_marks", data_type="DOUBLE"),
        ]
    )
    ws.tables.extend([tbl_students, tbl_grades])
    plan = JoinPlan(
        required_tables=["enrollments_marks", "students"],
        bridge_tables=[],
        join_edges=[
            RelationshipEdge(
                id="rel_1",
                source_table="enrollments_marks",
                source_column="student_id",
                target_table="students",
                target_column="student_id",
                join_type="LEFT",
                cardinality="N:1",
            )
        ],
        sql_join_clauses=["LEFT JOIN students ON enrollments_marks.student_id = students.student_id"],
        has_fan_trap=False,
        fact_tables_involved=["enrollments_marks"],
        shared_dimensions=[],
        fan_trap_guidance="",
    )

    sql = QueryAgent._deterministic_sql_compiler(
        question="now show me the rio of total grades gender wise",
        workspace=ws,
        join_plan=plan,
        matched_measures=[],
    )
    assert '"students"."gender"' in sql
    assert '"enrollments_marks"' in sql
    assert "ratio_pct" in sql


def test_do_same_for_faculty_entity_substitution_and_no_dummy_gender(enriched_demo_workspace):
    """Verify that 'do same for faculty' replaces students with faculty, drops students from inherited tables,
    recognizes faculty has no gender column, and resolves to academic rank breakdown without dummy 'Unknown' literals.
    """
    from src.agents.query_agent import QueryAgent
    from src.knowledge.okf_models import TableSchema, ColumnProfile, RelationshipEdge
    from src.knowledge.okf_store import JoinPlan

    ws, okf_store, rag_store, conn, report = enriched_demo_workspace

    tbl_colleges = TableSchema(
        name="colleges",
        columns=[
            ColumnProfile(name="college_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_name", data_type="VARCHAR"),
        ]
    )
    tbl_depts = TableSchema(
        name="departments",
        columns=[
            ColumnProfile(name="department_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="department_name", data_type="VARCHAR"),
        ]
    )
    tbl_faculty = TableSchema(
        name="faculty",
        columns=[
            ColumnProfile(name="faculty_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="department_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="first_name", data_type="VARCHAR"),
            ColumnProfile(name="last_name", data_type="VARCHAR"),
            ColumnProfile(name="academic_rank", data_type="VARCHAR"),
            ColumnProfile(name="salary", data_type="DOUBLE"),
        ]
    )
    tbl_students = TableSchema(
        name="students",
        columns=[
            ColumnProfile(name="student_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="gender", data_type="VARCHAR"),
        ]
    )
    ws.tables.extend([tbl_colleges, tbl_depts, tbl_faculty, tbl_students])

    history = [
        {
            "response": {
                "success": True,
                "executed_query": "SELECT college_name, gender, count(*) FROM students GROUP BY 1, 2",
                "user_question": "show me gender ratio of students for each college",
                "involved_tables": ["colleges", "students"],
            }
        }
    ]

    # 1. Follow-up resolution
    resolved_q, is_follow_up, ctx, inherited = QueryAgent._resolve_follow_up_context(
        question="do same for faculty",
        conversation_history=history,
        workspace=ws,
        llm_router=None,
    )

    assert is_follow_up is True
    # 'students' MUST NOT be in inherited tables when substituting faculty
    assert "students" not in inherited
    assert "colleges" in inherited
    # Because faculty does not have gender, resolved_q should target faculty distribution / academic rank
    assert "faculty" in resolved_q.lower()
    assert "gender" not in resolved_q.lower()

    # 2. Join plan and deterministic SQL compilation
    plan = JoinPlan(
        required_tables=["faculty", "departments", "colleges"],
        bridge_tables=[],
        join_edges=[
            RelationshipEdge(
                id="r1", source_table="faculty", source_column="department_id",
                target_table="departments", target_column="department_id", join_type="LEFT", cardinality="N:1"
            ),
            RelationshipEdge(
                id="r2", source_table="departments", source_column="college_id",
                target_table="colleges", target_column="college_id", join_type="LEFT", cardinality="N:1"
            ),
        ],
        sql_join_clauses=[
            "LEFT JOIN departments ON faculty.department_id = departments.department_id",
            "LEFT JOIN colleges ON departments.college_id = colleges.college_id",
        ],
        has_fan_trap=False,
        fact_tables_involved=["faculty"],
        shared_dimensions=[],
        fan_trap_guidance="",
    )

    sql = QueryAgent._deterministic_sql_compiler(
        question=resolved_q,
        workspace=ws,
        join_plan=plan,
        matched_measures=[],
    )

    # Must NOT contain dummy 'Unknown' or hallucinated columns
    assert "'Unknown'" not in sql
    assert '"colleges"."college_name"' in sql
    assert '"faculty"' in sql
    assert "departments" in sql
    # Sibling table 'students' must not be joined
    assert "students" not in sql.lower()



