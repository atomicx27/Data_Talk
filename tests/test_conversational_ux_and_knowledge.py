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
from src.knowledge.okf_models import OKFWorkspace, TableRole, TableSchema, ColumnProfile, RelationshipEdge
from src.knowledge.okf_store import OKFStore, JoinPlan
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


def test_out_of_domain_query_graceful_advisory(enriched_demo_workspace):
    """Verify that when the user asks for data completely absent from the workspace
    (e.g. 'Show patient heart rates and surgical procedures'), the agent does NOT
    force an arbitrary SQL query onto a random table, but returns a clear advisory notice.
    """
    from src.agents.query_agent import QueryAgent

    ws, okf_store, rag_store, conn, report = enriched_demo_workspace

    resp, df = QueryAgent.answer_question(
        question="Show patient hospital visits and surgical procedures",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=conn,
    )

    assert resp.success is False
    assert df.empty
    assert "Requested Data Not Found in Workspace" in resp.executive_summary
    assert "Available tables in this workspace" in resp.executive_summary
    assert resp.executed_query.startswith("-- No matching tables")


def test_multi_table_marks_by_gender_and_college_query(enriched_demo_workspace):
    """Verify that when the user asks for 'students marks ratio based on there gender for each college',
    the agent accurately selects students.gender (rather than completion_status) and enrollments_marks.total_marks,
    performing multi-hop joins across colleges, students, and enrollments_marks without bogus domain notices.
    """
    from src.agents.query_agent import QueryAgent
    from src.knowledge.okf_models import OKFWorkspace, TableSchema, ColumnProfile, TableRole, RelationshipEdge
    from src.knowledge.okf_store import OKFStore

    # Construct schema with colleges, students, and enrollments_marks
    colleges_tbl = TableSchema(
        name="colleges",
        table_role=TableRole.DIMENSION,
        columns=[
            ColumnProfile(name="college_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_code", data_type="VARCHAR"),
        ],
    )
    students_tbl = TableSchema(
        name="students",
        table_role=TableRole.DIMENSION,
        columns=[
            ColumnProfile(name="student_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="gender", data_type="VARCHAR"),
            ColumnProfile(name="academic_status", data_type="VARCHAR"),
        ],
    )
    enrollments_tbl = TableSchema(
        name="enrollments_marks",
        table_role=TableRole.FACT,
        columns=[
            ColumnProfile(name="enrollment_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="student_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="total_marks", data_type="DOUBLE"),
            ColumnProfile(name="completion_status", data_type="VARCHAR"),
        ],
    )

    ws = OKFWorkspace(
        workspace_id="test_univ_schema",
        workspace_name="University BI",
        source_type="duckdb",
        dialect="duckdb",
        tables=[colleges_tbl, students_tbl, enrollments_tbl],
        relationships=[
            RelationshipEdge(id="rel_stud_col", source_table="students", source_column="college_id", target_table="colleges", target_column="college_id"),
            RelationshipEdge(id="rel_enr_stud", source_table="enrollments_marks", source_column="student_id", target_table="students", target_column="student_id"),
        ],
    )

    q = "show me students marks ratio based on there gender for each college"
    okf_store_inst = OKFStore()
    join_plan = okf_store_inst.get_join_plan(ws, ["colleges", "enrollments_marks", "students"])

    sql = QueryAgent._deterministic_sql_compiler(
        question=q,
        workspace=ws,
        join_plan=join_plan,
        matched_measures=[],
    )

    assert sql is not None
    sql_low = sql.lower()

    # 1. Must group by college_code and gender
    assert "college_code" in sql_low
    assert "gender" in sql_low

    # 2. Must NOT mistakenly choose completion_status
    assert "completion_status" not in sql_low

    # 3. Must aggregate total_marks
    assert "total_marks" in sql_low
    assert "marks_ratio_pct" in sql_low

    # 4. Must join across all three tables
    assert "students" in sql_low
    assert "colleges" in sql_low
    assert "enrollments_marks" in sql_low


def test_drill_down_into_specific_college_and_department_breakdown():
    """Verify that clicking suggestion chip 'Drill down into College of Engineering & Computing'
    resolves context to filter to the college and break down by department,
    compiling clean SQL without duplicate table alias joins."""
    from src.knowledge.okf_models import OKFWorkspace, TableSchema, ColumnProfile, RelationshipEdge
    from src.knowledge.okf_store import OKFStore
    from src.agents.query_agent import QueryAgent

    colleges_tbl = TableSchema(
        name="colleges",
        columns=[
            ColumnProfile(name="college_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_name", data_type="VARCHAR"),
        ],
    )
    departments_tbl = TableSchema(
        name="departments",
        columns=[
            ColumnProfile(name="department_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="department_name", data_type="VARCHAR"),
            ColumnProfile(name="college_id", data_type="BIGINT", is_foreign_key=True),
        ],
    )
    students_tbl = TableSchema(
        name="students",
        columns=[
            ColumnProfile(name="student_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="gender", data_type="VARCHAR"),
            ColumnProfile(name="department_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="college_id", data_type="BIGINT", is_foreign_key=True),
        ],
    )
    enrollments_tbl = TableSchema(
        name="enrollments_marks",
        columns=[
            ColumnProfile(name="enrollment_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="student_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="total_score", data_type="DOUBLE"),
        ],
    )

    ws = OKFWorkspace(
        workspace_id="test_univ_drilldown",
        workspace_name="University BI Drilldown",
        source_type="duckdb",
        dialect="duckdb",
        tables=[colleges_tbl, departments_tbl, students_tbl, enrollments_tbl],
        relationships=[
            RelationshipEdge(id="r1", source_table="students", source_column="college_id", target_table="colleges", target_column="college_id"),
            RelationshipEdge(id="r2", source_table="students", source_column="department_id", target_table="departments", target_column="department_id"),
            RelationshipEdge(id="r3", source_table="departments", source_column="college_id", target_table="colleges", target_column="college_id"),
            RelationshipEdge(id="r4", source_table="enrollments_marks", source_column="student_id", target_table="students", target_column="student_id"),
        ],
    )

    history = [
        {
            "response": {
                "success": True,
                "executed_query": "SELECT college_name, COUNT(*) FROM students GROUP BY college_name",
                "user_question": "show student enrollments by college",
                "involved_tables": ["colleges", "students", "enrollments_marks"],
            }
        }
    ]

    # 1. Test conversational resolution
    chip_query = "Drill down into College of Engineering & Computing"
    resolved_q, is_follow_up, ctx, tables = QueryAgent._resolve_follow_up_context(
        question=chip_query,
        conversation_history=history,
        workspace=ws,
        llm_router=None,
    )

    assert is_follow_up is True
    assert "college of engineering & computing" in resolved_q.lower()
    assert "broken down by department" in resolved_q.lower()

    # 2. Test deterministic SQL compilation
    okf_store = OKFStore()
    join_plan = okf_store.get_join_plan(ws, ["colleges", "departments", "students", "enrollments_marks"])
    sql = QueryAgent._deterministic_sql_compiler(
        question=resolved_q,
        workspace=ws,
        join_plan=join_plan,
        matched_measures=[],
    )

    assert sql is not None
    # Must NOT contain duplicate JOIN clauses
    assert sql.count('LEFT JOIN "colleges"') <= 1
    # Must filter for the specific college
    assert "engineering" in sql.lower()
    # Must group by department_name
    assert "department_name" in sql.lower()


def test_drill_down_dropdown_break_that_down_by_department():
    """Verify that clicking the drilldown dropdown option 'Break that down by Department'
    resolves context and produces a valid multi-table breakdown query."""
    from src.knowledge.okf_models import OKFWorkspace, TableSchema, ColumnProfile, RelationshipEdge
    from src.knowledge.okf_store import OKFStore
    from src.agents.query_agent import QueryAgent

    colleges_tbl = TableSchema(
        name="colleges",
        columns=[
            ColumnProfile(name="college_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_name", data_type="VARCHAR"),
        ],
    )
    departments_tbl = TableSchema(
        name="departments",
        columns=[
            ColumnProfile(name="department_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="department_name", data_type="VARCHAR"),
            ColumnProfile(name="college_id", data_type="BIGINT", is_foreign_key=True),
        ],
    )
    students_tbl = TableSchema(
        name="students",
        columns=[
            ColumnProfile(name="student_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="gender", data_type="VARCHAR"),
            ColumnProfile(name="department_id", data_type="BIGINT", is_foreign_key=True),
        ],
    )

    ws = OKFWorkspace(
        workspace_id="test_univ_dd",
        workspace_name="University BI DD",
        source_type="duckdb",
        dialect="duckdb",
        tables=[colleges_tbl, departments_tbl, students_tbl],
        relationships=[
            RelationshipEdge(id="r1", source_table="students", source_column="department_id", target_table="departments", target_column="department_id"),
            RelationshipEdge(id="r2", source_table="departments", source_column="college_id", target_table="colleges", target_column="college_id"),
        ],
    )

    history = [
        {
            "response": {
                "success": True,
                "executed_query": "SELECT college_name, COUNT(*) FROM students GROUP BY college_name",
                "user_question": "show students by college",
                "involved_tables": ["colleges", "students"],
            }
        }
    ]

    dropdown_query = "Break that down by Department"
    resolved_q, is_follow_up, ctx, tables = QueryAgent._resolve_follow_up_context(
        question=dropdown_query,
        conversation_history=history,
        workspace=ws,
        llm_router=None,
    )

    assert is_follow_up is True
    assert "broken down by department" in resolved_q.lower()

    okf_store = OKFStore()
    join_plan = okf_store.get_join_plan(ws, ["colleges", "departments", "students"])
    sql = QueryAgent._deterministic_sql_compiler(
        question=resolved_q,
        workspace=ws,
        join_plan=join_plan,
        matched_measures=[],
    )

    assert sql is not None
    assert "department_name" in sql.lower()
    assert sql.count('LEFT JOIN "colleges"') <= 1


def test_drill_down_break_trend_by_college_no_irrelevant_table():
    """Verify that breaking down a trend query by college:
    1. Does not mistakenly pull in unmentioned tables like carbon_footprint_100_countries_2018_2025.
    2. Correctly selects faculty.salary as the metric (AVG) and hire_year as the time dimension.
    3. Produces executable, correctly-joined multi-table SQL without empty results."""
    from src.knowledge.okf_models import OKFWorkspace, TableSchema, ColumnProfile, RelationshipEdge
    from src.knowledge.okf_store import OKFStore
    from src.agents.query_agent import QueryAgent
    import duckdb

    colleges_tbl = TableSchema(
        name="colleges",
        columns=[
            ColumnProfile(name="college_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="college_name", data_type="VARCHAR"),
            ColumnProfile(name="established_year", data_type="BIGINT"),
        ],
    )
    departments_tbl = TableSchema(
        name="departments",
        columns=[
            ColumnProfile(name="department_id", data_type="BIGINT", is_primary_key=True),
            ColumnProfile(name="department_name", data_type="VARCHAR"),
            ColumnProfile(name="college_id", data_type="BIGINT", is_foreign_key=True),
        ],
    )
    faculty_tbl = TableSchema(
        name="faculty",
        columns=[
            ColumnProfile(name="faculty_id", data_type="VARCHAR", is_primary_key=True),
            ColumnProfile(name="department_id", data_type="BIGINT", is_foreign_key=True),
            ColumnProfile(name="hire_date", data_type="VARCHAR"),
            ColumnProfile(name="salary", data_type="DOUBLE"),
        ],
    )
    carbon_tbl = TableSchema(
        name="carbon_footprint_100_countries_2018_2025",
        columns=[
            ColumnProfile(name="Country", data_type="VARCHAR"),
            ColumnProfile(name="Year", data_type="BIGINT"),
            ColumnProfile(name="Total", data_type="DOUBLE"),
        ],
    )

    ws = OKFWorkspace(
        workspace_id="test_faculty_trend",
        workspace_name="Faculty Trend BI",
        source_type="duckdb",
        dialect="duckdb",
        tables=[colleges_tbl, departments_tbl, faculty_tbl, carbon_tbl],
        relationships=[
            RelationshipEdge(id="r1", source_table="faculty", source_column="department_id", target_table="departments", target_column="department_id"),
            RelationshipEdge(id="r2", source_table="departments", source_column="college_id", target_table="colleges", target_column="college_id"),
        ],
    )

    history = [
        {
            "response": {
                "success": True,
                "executed_query": "SELECT STRFTIME(TRY_CAST(hire_date AS DATE), '%Y') AS hire_year, AVG(salary) AS avg_salary FROM faculty GROUP BY hire_year",
                "user_question": "show trend of salary of faculty year on year",
                "involved_tables": ["faculty"],
            }
        }
    ]

    # 1. Resolve context for drilldown: "Break that down by College"
    resolved_q, is_fu, ctx, prior_tables = QueryAgent._resolve_follow_up_context(
        question="Break that down by College",
        conversation_history=history,
        workspace=ws,
        llm_router=None,
    )
    assert is_fu is True
    assert "broken down by college" in resolved_q.lower()

    # 2. Relevant table detection MUST NOT include carbon footprint
    detected = QueryAgent._detect_relevant_tables(resolved_q, ws, {})
    assert "carbon_footprint_100_countries_2018_2025" not in detected
    assert "faculty" in detected
    assert "colleges" in detected

    # 3. Join plan must only connect valid tables
    store = OKFStore()
    plan = store.get_join_plan(ws, detected)
    assert "carbon_footprint_100_countries_2018_2025" not in plan.required_tables
    assert set(plan.required_tables) == {"colleges", "departments", "faculty"}

    # 4. Deterministic compiler must aggregate salary and group by college_name + hire_year
    sql = QueryAgent._deterministic_sql_compiler(resolved_q, ws, plan, [])
    assert sql is not None
    sql_low = sql.lower()
    assert "carbon_footprint" not in sql_low
    assert "college_name" in sql_low
    assert "hire_year" in sql_low
    assert "avg_salary" in sql_low
    assert "established_year" not in sql_low
    assert "from \"faculty\"" in sql_low

    # 5. Verify SQL execution against an in-memory DuckDB database
    con = duckdb.connect()
    con.execute('''
        CREATE TABLE colleges (college_id BIGINT, college_name VARCHAR, established_year BIGINT);
        CREATE TABLE departments (department_id BIGINT, department_name VARCHAR, college_id BIGINT);
        CREATE TABLE faculty (faculty_id VARCHAR, department_id BIGINT, hire_date VARCHAR, salary DOUBLE);
        INSERT INTO colleges VALUES (1, 'College of Engineering & Computing', 1980);
        INSERT INTO departments VALUES (10, 'Computer Science', 1);
        INSERT INTO faculty VALUES ('f1', 10, '2005-08-15', 94000.0), ('f2', 10, '2006-08-15', 109000.0);
    ''')
    df = con.execute(sql).fetchdf()
    assert not df.empty
    assert len(df) == 2
    assert "college_name" in df.columns
    assert "avg_salary" in df.columns


def test_fee_trend_and_drill_down_executive_summary_and_follow_ups():
    """Verify Screenshot 1 & 2 fixes:
    1. No duplicate 'total_total_fee_due' alias prefix.
    2. Suggested follow-ups suggest College/Department on University workspace, not product category.
    3. Executive summary on 2D drill-down analyzes Total Fee Due, not Payment Year Count.
    """
    ws = OKFWorkspace(
        workspace_id="test_uni_fees",
        workspace_name="University Workspace",
        tables=[
            TableSchema(name="colleges", columns=[ColumnProfile(name="college_id", data_type="BIGINT", is_primary_key=True), ColumnProfile(name="college_name", data_type="VARCHAR")]),
            TableSchema(name="departments", columns=[ColumnProfile(name="department_id", data_type="BIGINT", is_primary_key=True), ColumnProfile(name="department_name", data_type="VARCHAR"), ColumnProfile(name="college_id", data_type="BIGINT")]),
            TableSchema(name="students", columns=[ColumnProfile(name="student_id", data_type="BIGINT", is_primary_key=True), ColumnProfile(name="department_id", data_type="BIGINT"), ColumnProfile(name="payment_year", data_type="BIGINT"), ColumnProfile(name="total_fee_due", data_type="DOUBLE")]),
        ]
    )

    # 1. Alias clean test: no 'total_total_fee_due'
    clean_tot = QueryAgent._clean_agg_alias("total_fee_due", "SUM")
    assert clean_tot == "total_fee_due"
    clean_avg = QueryAgent._clean_agg_alias("total_fee_due", "AVG")
    assert clean_avg == "avg_fee_due"

    # Single-table trend query SQL alias verification
    plan_students = JoinPlan(
        required_tables=["students"],
        bridge_tables=[],
        join_edges=[],
        sql_join_clauses=[],
        has_fan_trap=False,
        fact_tables_involved=["students"],
        shared_dimensions=[],
        fan_trap_guidance="",
    )
    trend_sql = QueryAgent._deterministic_sql_compiler("show me trend of total fees", ws, plan_students, [])
    assert "total_total_fee_due" not in trend_sql
    assert "total_fee_due" in trend_sql

    # 2. Suggested follow-ups on University workspace
    df_trend = pd.DataFrame({
        "payment_year": [2023, 2024, 2025],
        "total_fee_due": [78000000.0, 89000000.0, 98000000.0],
        "row_count": [15000, 17000, 19000],
    })
    follow_ups = QueryAgent._generate_suggested_follow_ups(df_trend, "show me trend of total fees", ws, plan_students)
    assert not any("product category" in f.lower() for f in follow_ups)
    assert any("college" in f.lower() or "department" in f.lower() for f in follow_ups)

    # 3. Executive summary on 2D drill-down
    df_drill = pd.DataFrame({
        "college_name": ["College of Architecture & Built Environment", "College of Engineering & Technology"],
        "payment_year": [2025, 2025],
        "total_fee_due": [98240000.0, 85000000.0],
        "payment_year_count": [2483, 2100],
    })
    summary = QueryAgent._build_executive_summary("Break that down by College", df_drill, [], [], None, ws)
    assert "Total Fee Due" in summary
    assert "Payment Year Count" not in summary
    assert "College of Architecture & Built Environment" in summary
    assert "98,240,000.00" in summary or "98,240,000" in summary

    # 4. 2D breakdown limit must allow all colleges across years (LIMIT 500, not LIMIT 50)
    plan_2d = JoinPlan(
        required_tables=["students", "departments", "colleges"],
        bridge_tables=[],
        join_edges=[
            RelationshipEdge(id="r1", source_table="students", source_column="department_id", target_table="departments", target_column="department_id"),
            RelationshipEdge(id="r2", source_table="departments", source_column="college_id", target_table="colleges", target_column="college_id"),
        ],
        sql_join_clauses=[
            'LEFT JOIN "departments" ON "students"."department_id" = "departments"."department_id"',
            'LEFT JOIN "colleges" ON "departments"."college_id" = "colleges"."college_id"',
        ],
        has_fan_trap=False,
        fact_tables_involved=["students"],
        shared_dimensions=[],
        fan_trap_guidance="",
    )
    sql_2d = QueryAgent._deterministic_sql_compiler("show me fee trend of students year on year broken down by college", ws, plan_2d, [])
    # No artificial LIMIT should truncate grouped analytical results
    assert "LIMIT" not in sql_2d

    # But explicit user limit (e.g. top 3) must be honored
    sql_top = QueryAgent._deterministic_sql_compiler("show me top 3 fee trend of students year on year broken down by college", ws, plan_2d, [])
    assert "LIMIT 3" in sql_top


def test_multi_persona_audit_fixes_non_additive_summary_and_sql_compiler():
    """Verify fixes for the 4-subagent audit (Data Analyst, Business Analyst, Power BI Expert, Code Logic):
    1. Time-series executive summary reports Trend Trajectory, Peak/Lowest Period, and Period Benchmark (never '% of total' sum of averages).
    2. Categorical non-additive executive summary reports Highest/Lowest Average vs benchmark mean and Spread (never summing averages or showing Concentration Alert).
    3. 'Bottom 5' / 'Lowest 3' queries sort ASC instead of DESC.
    4. 'for each college' does not inject WHERE LOWER(...) LIKE '%each%' and selects colleges.college_name instead of departments.college_id.
    5. 1D 'Show faculty salary by college' uses AVG, not SUM.
    6. Single-table 'Show faculty count by academic rank' groups by academic_rank, not first_name.
    """
    ws = OKFWorkspace(
        workspace_id="test_audit_uni",
        workspace_name="University Audit Workspace",
        tables=[
            TableSchema(
                name="colleges",
                columns=[
                    ColumnProfile(name="college_id", data_type="BIGINT", is_primary_key=True),
                    ColumnProfile(name="college_name", data_type="VARCHAR"),
                ],
            ),
            TableSchema(
                name="departments",
                columns=[
                    ColumnProfile(name="department_id", data_type="BIGINT", is_primary_key=True),
                    ColumnProfile(name="department_name", data_type="VARCHAR"),
                    ColumnProfile(name="college_id", data_type="BIGINT"),
                ],
            ),
            TableSchema(
                name="faculty",
                columns=[
                    ColumnProfile(name="faculty_id", data_type="BIGINT", is_primary_key=True),
                    ColumnProfile(name="first_name", data_type="VARCHAR"),
                    ColumnProfile(name="academic_rank", data_type="VARCHAR"),
                    ColumnProfile(name="department_id", data_type="BIGINT"),
                    ColumnProfile(name="hire_year", data_type="BIGINT"),
                    ColumnProfile(name="salary", data_type="DOUBLE"),
                ],
            ),
        ],
    )

    # 1. Time-series executive summary on avg_salary
    df_ts = pd.DataFrame({
        "hire_year": [2020, 2021, 2022, 2023],
        "avg_salary": [80000.0, 85000.0, 90000.0, 100000.0],
        "row_count": [20, 25, 22, 30],
    })
    ts_sum = QueryAgent._build_executive_summary(
        "show me salary trend of faculty year on year", df_ts, [], [], None, ws
    )
    assert "Trend Trajectory" in ts_sum
    assert "Peak Period" in ts_sum
    assert "% of total" not in ts_sum
    assert "Concentration Alert" not in ts_sum

    # 2. Categorical non-additive executive summary on avg_salary
    df_cat_avg = pd.DataFrame({
        "college_name": ["College of Law", "College of Business", "College of Arts"],
        "avg_salary": [110000.0, 95000.0, 80000.0],
        "total_count": [30, 45, 50],
    })
    cat_sum = QueryAgent._build_executive_summary(
        "Show faculty salary by college", df_cat_avg, [], [], None, ws
    )
    assert "Highest Average / Rate" in cat_sum
    assert "benchmark mean" in cat_sum
    assert "% of total" not in cat_sum
    assert "Concentration Alert" not in cat_sum

    # 3, 4, 5. Multi-table 1D compiler checks: AVG(salary), colleges.college_name (not departments.college_id), no LIKE '%each%', and ASC for lowest/bottom
    plan_fac = JoinPlan(
        required_tables=["colleges", "departments", "faculty"],
        bridge_tables=["departments"],
        join_edges=[
            RelationshipEdge(id="e1", source_table="faculty", source_column="department_id", target_table="departments", target_column="department_id"),
            RelationshipEdge(id="e2", source_table="departments", source_column="college_id", target_table="colleges", target_column="college_id"),
        ],
        sql_join_clauses=[],
        has_fan_trap=False,
        fact_tables_involved=["faculty"],
        shared_dimensions=["colleges", "departments"],
        fan_trap_guidance="",
    )
    sql_each = QueryAgent._deterministic_sql_compiler(
        "Show faculty salary for each college", ws, plan_fac, []
    )
    assert 'AVG("faculty"."salary")' in sql_each
    assert '"colleges"."college_name"' in sql_each
    assert '"departments"."college_id" AS' not in sql_each
    assert "LIKE '%each%'" not in sql_each

    sql_bottom = QueryAgent._deterministic_sql_compiler(
        "Show bottom 3 faculty salary by college", ws, plan_fac, []
    )
    assert "ORDER BY avg_salary ASC" in sql_bottom
    assert "LIMIT 3" in sql_bottom

    # 6. Single-table query groups by requested dimension (academic_rank), not first_name
    plan_single = JoinPlan(
        required_tables=["faculty"],
        bridge_tables=[],
        join_edges=[],
        sql_join_clauses=[],
        has_fan_trap=False,
        fact_tables_involved=["faculty"],
        shared_dimensions=[],
        fan_trap_guidance="",
    )
    sql_single = QueryAgent._deterministic_sql_compiler(
        "Show faculty salary by academic_rank", ws, plan_single, []
    )
    assert '"academic_rank"' in sql_single
    assert '"first_name"' not in sql_single


def test_overview_table_roles_compound_fks_and_fuzzy_typo_compiler():
    """Verify table role healing, compound FK inference, dataset overview follow-ups, and fuzzy typo handling ('deparement')."""
    from src.knowledge.okf_models import OKFWorkspace, TableRole
    from src.knowledge.okf_store import OKFStore
    from src.knowledge.join_inference import JoinInferenceEngine

    ws = OKFWorkspace(
        workspace_id="test_uni_typo",
        workspace_name="University Test",
        tables=[
            TableSchema(
                name="colleges",
                row_count=8,
                table_role=TableRole.FACT,  # Stale misclassified role on disk
                columns=[
                    ColumnProfile(name="college_id", data_type="VARCHAR", is_primary_key=True, distinct_count=8),
                    ColumnProfile(name="college_name", data_type="VARCHAR", distinct_count=8),
                ],
            ),
            TableSchema(
                name="departments",
                row_count=32,
                table_role=TableRole.FACT,  # Stale misclassified role on disk
                columns=[
                    ColumnProfile(name="department_id", data_type="VARCHAR", is_primary_key=True, distinct_count=32),
                    ColumnProfile(name="college_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=8),
                    ColumnProfile(name="department_name", data_type="VARCHAR", distinct_count=32),
                    ColumnProfile(name="budget_allocated", data_type="BIGINT", distinct_count=32),
                ],
            ),
            TableSchema(
                name="degree_programs",
                row_count=64,
                table_role=TableRole.UNKNOWN,
                columns=[
                    ColumnProfile(name="program_id", data_type="VARCHAR", distinct_count=64),
                    ColumnProfile(name="department_id", data_type="VARCHAR", distinct_count=32),
                    ColumnProfile(name="program_name", data_type="VARCHAR", distinct_count=64),
                ],
            ),
            TableSchema(
                name="faculty",
                row_count=421,
                table_role=TableRole.FACT,  # Stale misclassified role on disk
                columns=[
                    ColumnProfile(name="faculty_id", data_type="VARCHAR", is_primary_key=True, distinct_count=421),
                    ColumnProfile(name="department_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=32),
                    ColumnProfile(name="hire_date", data_type="VARCHAR", distinct_count=350),
                    ColumnProfile(name="salary", data_type="BIGINT", distinct_count=400),
                ],
            ),
            TableSchema(
                name="students",
                row_count=12800,
                table_role=TableRole.FACT,  # Stale misclassified role on disk
                columns=[
                    ColumnProfile(name="student_id", data_type="VARCHAR", is_primary_key=True, distinct_count=12800),
                    ColumnProfile(name="college_id", data_type="VARCHAR", distinct_count=8),
                    ColumnProfile(name="program_id", data_type="VARCHAR", distinct_count=64),
                    ColumnProfile(name="date_of_birth", data_type="VARCHAR", distinct_count=3000),
                    ColumnProfile(name="current_semester_num", data_type="BIGINT", distinct_count=8),
                ],
            ),
            TableSchema(
                name="enrollments_marks",
                row_count=344054,
                table_role=TableRole.DIMENSION,  # Stale misclassified role on disk
                columns=[
                    ColumnProfile(name="enrollment_id", data_type="VARCHAR", is_primary_key=True, distinct_count=344054),
                    ColumnProfile(name="student_id", data_type="VARCHAR", distinct_count=12800),
                    ColumnProfile(name="total_marks", data_type="DOUBLE", distinct_count=100),
                ],
            ),
            TableSchema(
                name="fee_invoices",
                row_count=105260,
                table_role=TableRole.DIMENSION,  # Stale misclassified role on disk
                columns=[
                    ColumnProfile(name="invoice_id", data_type="VARCHAR", is_primary_key=True, distinct_count=105260),
                    ColumnProfile(name="student_id", data_type="VARCHAR", distinct_count=12800),
                    ColumnProfile(name="net_amount_due", data_type="BIGINT", distinct_count=500),
                ],
            ),
        ],
    )

    # 1. Verify compound FK inference (students.program_id -> degree_programs.program_id) and role classification
    ws.relationships = JoinInferenceEngine.infer_relationships(ws.tables, connector=None)
    OKFStore.classify_table_roles(ws)

    assert any(
        r.source_table == "students" and r.target_table == "degree_programs" and r.source_column == "program_id"
        for r in ws.relationships
    )
    assert ws.get_table("enrollments_marks").table_role == TableRole.FACT
    assert ws.get_table("fee_invoices").table_role == TableRole.FACT
    assert ws.get_table("colleges").table_role == TableRole.DIMENSION
    assert ws.get_table("departments").table_role == TableRole.DIMENSION
    assert ws.get_table("students").table_role == TableRole.DIMENSION
    assert ws.get_table("faculty").table_role == TableRole.DIMENSION

    # 2. Verify "what is the data about" overview does not suggest "Drill down into enrollments_marks"
    resp_ov, df_ov = QueryAgent._generate_dataset_overview("what is the data about", ws, None)
    assert "enrollments_marks" in resp_ov.executive_summary.split("Core Operational Facts & Metrics")[1]
    assert not any("Drill down into enrollments_marks" in s for s in resp_ov.suggested_follow_ups)

    # 3. Verify typo tolerance: "show me total cost of each deparement year on year"
    detected = QueryAgent._detect_relevant_tables("show me total cost of each deparement year on year", ws, {})
    assert "departments" in detected
    assert "faculty" in detected
    assert "students" not in detected

    jp = OKFStore().get_join_plan(ws, detected)
    sql = QueryAgent._deterministic_sql_compiler(
        "show me total cost of each deparement year on year", ws, jp, []
    )
    assert '"departments"."department_name"' in sql
    assert 'SUM("faculty"."salary")' in sql
    assert "year_of_birth" not in sql
    assert "current_semester_num" not in sql

    # 4. Verify 2D time-series chart recommendation is 'line' (not 'stacked_bar') with hire_year on X
    from src.viz.chart_recommender import ChartRecommender

    df_dept_yoy = pd.DataFrame({
        "department_name": [f"Dept {i}" for i in range(15) for _ in range(3)],
        "hire_year": ["2021", "2022", "2023"] * 15,
        "total_salary": [100000.0 + i * 5000 for i in range(45)],
        "hire_year_count": [2] * 45,
    })
    rec_yoy = ChartRecommender.recommend_chart_type(df_dept_yoy, "show me total cost of each deparement year on year")
    assert rec_yoy == "line"
    fig_yoy = ChartRecommender.build_figure(df_dept_yoy, rec_yoy, "show me total cost of each deparement year on year")
    assert fig_yoy is not None
    assert len(fig_yoy.data) == 10  # Capped at Top 10 departments for legibility

    # 5. Verify multi-fact comparison: "show me a comparison between the expenditure and earning of each college"
    q_comp = "show me a comparison between the expenditure and earning of each college"
    det_comp = QueryAgent._detect_relevant_tables(q_comp, ws, {})
    assert "colleges" in det_comp
    assert "departments" in det_comp
    assert "fee_invoices" in det_comp
    jp_comp = OKFStore().get_join_plan(ws, det_comp)
    sql_comp = QueryAgent._deterministic_sql_compiler(q_comp, ws, jp_comp, [])
    assert "WITH dept_budget_cte AS" in sql_comp
    assert "fee_earnings_cte AS" in sql_comp
    assert "AS college_name" in sql_comp
    assert "AS total_earnings" in sql_comp
    assert "AS total_expenditure" in sql_comp
    assert jp_comp.has_fan_trap is True

    df_comp = pd.DataFrame({
        "college_name": ["College of Engineering", "College of Business"],
        "total_earnings": [15000000.0, 12000000.0],
        "total_expenditure": [9500000.0, 8200000.0],
        "department_budget": [5000000.0, 4200000.0],
        "faculty_payroll": [4500000.0, 4000000.0],
        "net_surplus_deficit": [5500000.0, 3800000.0],
    })
    rec_comp = ChartRecommender.recommend_chart_type(df_comp, q_comp)
    assert rec_comp == "bar"
    fig_comp = ChartRecommender.build_figure(df_comp, rec_comp, q_comp)
    assert fig_comp is not None
    assert len(fig_comp.data) == 2  # Plots both total_earnings and total_expenditure side-by-side
    sum_comp = QueryAgent._build_executive_summary(q_comp, df_comp, [], [], None, ws)
    assert "Highest Earnings" in sum_comp
    assert "Highest Expenditure" in sum_comp
    assert "Overall Net Position" in sum_comp

    # 6. Verify compiler_mode and compiler_detail indicator ("heuristic" vs "llm")
    sql_h, dax_h, mode_h, detail_h = QueryAgent._generate_query(
        q_comp, ws, jp_comp, [], {}, False, None, None
    )
    assert mode_h == "heuristic"
    assert "Heuristic" in detail_h


def test_partial_entity_matching_chasm_drilldown_prompt_improver_and_chart_fixes(test_client: TestClient):
    """Verify partial category matching ('philosophy' -> 'History & Philosophy'), partial column matching,
    chasm-free College->Department drill-down, multi-fact drill-down, prompt improver, and chart recommendation fixes."""
    from src.knowledge.okf_models import OKFWorkspace, TableRole, TableSchema, ColumnProfile, RelationshipEdge
    from src.knowledge.okf_store import OKFStore
    from src.knowledge.join_inference import JoinInferenceEngine
    from src.viz.chart_recommender import ChartRecommender

    ws = OKFWorkspace(
        workspace_id="test_uni_drill_partial",
        workspace_name="University Drill & Partial Test",
        tables=[
            TableSchema(
                name="colleges",
                row_count=8,
                table_role=TableRole.DIMENSION,
                columns=[
                    ColumnProfile(name="college_id", data_type="VARCHAR", is_primary_key=True, distinct_count=8),
                    ColumnProfile(
                        name="college_name",
                        data_type="VARCHAR",
                        distinct_count=8,
                        sample_values=[
                            "College of Engineering & Computing",
                            "College of Arts & Humanities",
                            "College of Business & Management",
                        ],
                    ),
                ],
            ),
            TableSchema(
                name="departments",
                row_count=32,
                table_role=TableRole.DIMENSION,
                columns=[
                    ColumnProfile(name="department_id", data_type="VARCHAR", is_primary_key=True, distinct_count=32),
                    ColumnProfile(name="college_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=8),
                    ColumnProfile(name="department_code", data_type="VARCHAR", distinct_count=32),
                    ColumnProfile(
                        name="department_name",
                        data_type="VARCHAR",
                        distinct_count=32,
                        sample_values=[
                            "History & Philosophy",
                            "Electrical & Computer Engineering",
                            "Computer Science & Engineering",
                            "Mechanical & Aerospace Engineering",
                            "English & Comparative Literature",
                        ],
                    ),
                    ColumnProfile(name="budget_allocated", data_type="BIGINT", distinct_count=32),
                ],
            ),
            TableSchema(
                name="degree_programs",
                row_count=64,
                table_role=TableRole.DIMENSION,
                columns=[
                    ColumnProfile(name="program_id", data_type="VARCHAR", is_primary_key=True, distinct_count=64),
                    ColumnProfile(name="department_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=32),
                    ColumnProfile(
                        name="program_name",
                        data_type="VARCHAR",
                        distinct_count=64,
                        sample_values=[
                            "B.Tech in Artificial Intelligence & Data Science",
                            "B.A. in History & Philosophy",
                        ],
                    ),
                    ColumnProfile(name="degree_level", data_type="VARCHAR", distinct_count=4),
                ],
            ),
            TableSchema(
                name="faculty",
                row_count=421,
                table_role=TableRole.DIMENSION,
                columns=[
                    ColumnProfile(name="faculty_id", data_type="VARCHAR", is_primary_key=True, distinct_count=421),
                    ColumnProfile(name="department_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=32),
                    ColumnProfile(name="academic_rank", data_type="VARCHAR", distinct_count=5),
                    ColumnProfile(name="salary", data_type="BIGINT", distinct_count=400),
                ],
            ),
            TableSchema(
                name="students",
                row_count=12800,
                table_role=TableRole.DIMENSION,
                columns=[
                    ColumnProfile(name="student_id", data_type="VARCHAR", is_primary_key=True, distinct_count=12800),
                    ColumnProfile(name="college_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=8),
                    ColumnProfile(name="program_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=64),
                    ColumnProfile(name="gender", data_type="VARCHAR", distinct_count=3),
                ],
            ),
            TableSchema(
                name="enrollments_marks",
                row_count=344054,
                table_role=TableRole.FACT,
                columns=[
                    ColumnProfile(name="enrollment_id", data_type="VARCHAR", is_primary_key=True, distinct_count=344054),
                    ColumnProfile(name="student_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=12800),
                    ColumnProfile(name="internal_assessment_score", data_type="DOUBLE", distinct_count=100),
                    ColumnProfile(name="midterm_score", data_type="DOUBLE", distinct_count=100),
                    ColumnProfile(name="total_marks", data_type="DOUBLE", distinct_count=100),
                ],
            ),
            TableSchema(
                name="fee_invoices",
                row_count=105260,
                table_role=TableRole.FACT,
                columns=[
                    ColumnProfile(name="invoice_id", data_type="VARCHAR", is_primary_key=True, distinct_count=105260),
                    ColumnProfile(name="student_id", data_type="VARCHAR", is_foreign_key=True, distinct_count=12800),
                    ColumnProfile(name="technology_library_fee", data_type="BIGINT", distinct_count=50),
                    ColumnProfile(name="hostel_mess_fee", data_type="BIGINT", distinct_count=50),
                    ColumnProfile(name="balance_outstanding", data_type="BIGINT", distinct_count=500),
                    ColumnProfile(name="net_amount_due", data_type="BIGINT", distinct_count=500),
                ],
            ),
        ],
    )
    ws.relationships = JoinInferenceEngine.infer_relationships(ws.tables, connector=None)
    OKFStore.classify_table_roles(ws)
    okf = OKFStore()

    # 1. Partial category value matching: "philosophy" -> departments.department_name = 'History & Philosophy'
    matches_phil = QueryAgent._detect_categorical_value_matches("show faculty salary in philosophy", ws)
    assert any(
        m["table"] == "departments"
        and m["column"] == "department_name"
        and m["value"] == "History & Philosophy"
        for m in matches_phil
    )
    det_phil = QueryAgent._detect_relevant_tables("show faculty salary in philosophy", ws, {})
    assert "departments" in det_phil
    assert "faculty" in det_phil
    jp_phil = okf.get_join_plan(ws, det_phil)
    sql_phil = QueryAgent._deterministic_sql_compiler("show faculty salary in philosophy", ws, jp_phil, [])
    assert "History & Philosophy" in sql_phil

    # 2. Partial column name matching: "internal score", "outstanding", "library fee"
    det_int = QueryAgent._detect_relevant_tables("show internal score by college", ws, {})
    assert "enrollments_marks" in det_int
    assert "colleges" in det_int
    jp_int = okf.get_join_plan(ws, det_int)
    sql_int = QueryAgent._deterministic_sql_compiler("show internal score by college", ws, jp_int, [])
    assert "internal_assessment_score" in sql_int

    det_out = QueryAgent._detect_relevant_tables("show outstanding fees by college", ws, {})
    assert "fee_invoices" in det_out
    jp_out = okf.get_join_plan(ws, det_out)
    sql_out = QueryAgent._deterministic_sql_compiler("show outstanding fees by college", ws, jp_out, [])
    assert "balance_outstanding" in sql_out

    # 3. Chasm-free Steiner tree & SQL for College -> Department drill-down on student marks
    det_drill = ["enrollments_marks", "students", "colleges", "departments"]
    jp_drill = okf.get_join_plan(ws, det_drill)
    # Must route through degree_programs (N:1 hierarchy), NOT direct colleges <- departments chasm fan-out
    assert "degree_programs" in jp_drill.required_tables
    q_drill = "Show average marks for each college for College of Engineering & Computing broken down by Department"
    sql_drill = QueryAgent._deterministic_sql_compiler(q_drill, ws, jp_drill, [])
    assert '"departments"."department_name"' in sql_drill
    assert "College of Engineering & Computing" in sql_drill
    assert '"degree_programs"' in sql_drill

    # 4. Multi-fact Expenditure vs. Earnings drill-down into a specific College by Department
    q_b2_drill = (
        "show me a comparison between the expenditure and earning of each college "
        "for College of Engineering & Computing broken down by Department"
    )
    sql_b2_drill = QueryAgent._deterministic_sql_compiler(q_b2_drill, ws, jp_drill, [])
    assert "AS department_name" in sql_b2_drill
    assert "College of Engineering & Computing" in sql_b2_drill
    assert "degree_programs" in sql_b2_drill

    # 5. Hierarchy-aware drill-down options: viewing department_name must NOT offer College (coarse parent)
    df_dept = pd.DataFrame({
        "department_name": ["History & Philosophy", "Computer Science & Engineering"],
        "avg_salary": [92000.0, 118000.0],
    })
    drill_opts = QueryAgent._generate_drill_down_options(df_dept, ws, ["departments", "faculty"])
    opt_dims = [d["dimension"] for d in drill_opts]
    assert "college" not in opt_dims
    assert "degree_program" in opt_dims or "academic_rank" in opt_dims

    # 6. Prompt Improver (QueryAgent.improve_user_prompt & POST /api/chat/improve-prompt)
    imp = QueryAgent.improve_user_prompt("average marks in philosophy", ws)
    assert "History & Philosophy" in imp["improved_question"]
    assert len(imp["resolutions"]) >= 1

    test_client.post("/api/connect/demo")
    imp_res = test_client.post("/api/chat/improve-prompt", json={"question": "rev by cat"})
    assert imp_res.status_code == 200
    imp_json = imp_res.json()
    assert "improved_question" in imp_json

    # 7. Chart Recommender: 2D percentage share (gender_ratio_pct) uses stacked_bar_100, and 32-row horizontal_bar scales height
    df_gender_pct = pd.DataFrame({
        "college_name": ["College A", "College A", "College B", "College B"],
        "gender": ["Female", "Male", "Female", "Male"],
        "gender_ratio_pct": [52.0, 48.0, 49.0, 51.0],
        "student_count": [520, 480, 490, 510],
    })
    rec_pct = ChartRecommender.recommend_chart_type(df_gender_pct, "show gender ratio in each college")
    assert rec_pct in ("stacked_bar_100", "stacked_bar")

    df_32_depts = pd.DataFrame({
        "department_name": [f"Department {i}" for i in range(32)],
        "avg_salary": [85000.0 + i * 1000 for i in range(32)],
        "row_count": [12 + (i % 5) for i in range(32)],
    })
    rec_32 = ChartRecommender.recommend_chart_type(df_32_depts, "show average salary by department")
    assert rec_32 == "horizontal_bar"
    fig_32 = ChartRecommender.build_figure(df_32_depts, rec_32, "show average salary by department")
    assert fig_32.layout.height >= 700
    assert len(fig_32.data) == 1  # Does NOT stack row_count on top of avg_salary













