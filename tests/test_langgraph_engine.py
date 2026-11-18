"""Comprehensive Phase 7 Backend Verification Suite for the LangGraph State Engine."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
import sqlglot
from fastapi.testclient import TestClient
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import ValidationError

from src.agents.compilers.deterministic_compiler import compile_deterministic_sql
from src.agents.compilers.semantic_helpers import (
    build_adaptive_schema_context,
    expand_schema_from_error,
    select_adaptive_schema_tiers,
)
from src.agents.compilers.text_matching import detect_relevant_tables, match_measures
from src.agents.graph import (
    BlockingAmbiguityPayload,
    ErrorDiagnostic,
    JoinPlanSnapshot,
    NodeTelemetryRecord,
    QueryGraphContext,
    QueryGraphState,
    SupervisorState,
    build_default_checkpointer,
    build_query_graph,
    build_supervisor_graph,
    get_query_graph_topology,
    get_supervisor_graph_topology,
    resume_query_graph,
    run_query_graph,
    run_supervisor_graph,
    stream_query_graph,
    trace_graph_node,
)
from src.agents.graph.nodes import (
    build_out_of_domain,
    check_verified_memory,
    clarification_interrupt,
    classify_error_diagnostic,
    compile_initial_query,
    critic_repair_query,
    dataset_overview,
    execute_sandbox,
    handle_exhausted_failure,
    ingest_and_resolve_followup,
    post_process_and_summarize,
    repair_query_with_guard,
    retrieve_okf_and_rag,
    route_after_followup,
    route_after_retrieval,
)
from src.agents.llm_router import LLMRouter
from src.agents.query_agent import QueryAgent, QueryResponse
from src.api.runtime import AppRuntime, SessionState
from src.connectors.duckdb_connector import DuckDBConnector
from src.knowledge.okf_models import (
    BusinessRule,
    ColumnProfile,
    InterviewQuestion,
    JoinCardinality,
    JoinStatus,
    OKFWorkspace,
    RelationshipEdge,
    SemanticMeasure,
    TableRole,
    TableSchema,
    VerifiedQuery,
)
from src.knowledge.okf_store import JoinPlan, OKFStore
from src.knowledge.rag_store import RAGStore
from src.security.privacy_firewall import PrivacyFirewall


def _make_retail_env(tmp_path: Path) -> tuple[OKFWorkspace, OKFStore, RAGStore, DuckDBConnector]:
    okf_store = OKFStore(base_dir=tmp_path / "workspaces")
    ws = OKFWorkspace(workspace_id="retail_test", workspace_name="Retail Test")
    conn = DuckDBConnector(tmp_path / "workspaces" / "retail_test")
    conn.ingest_dataframe(
        "customers",
        pd.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4"],
            "region_code": ["NA", "EMEA", "APAC", "LATAM"],
            "cust_tier": ["ENT", "MID", "SMB", "ENT"],
        }),
    )
    conn.ingest_dataframe(
        "products",
        pd.DataFrame({
            "product_id": ["P1", "P2", "P3", "P4"],
            "category": ["Cloud", "Security", "Analytics", "Hardware"],
            "sku_name": ["Cloud Pro", "Shield X", "DataLens", "Edge Box"],
        }),
    )
    conn.ingest_dataframe(
        "orders",
        pd.DataFrame({
            "order_id": ["O1", "O2", "O3", "O4", "O5", "O6"],
            "customer_id": ["C1", "C2", "C3", "C4", "C1", "C2"],
            "product_id": ["P1", "P2", "P3", "P4", "P2", "P1"],
            "order_date": pd.to_datetime([
                "2025-01-15", "2025-02-10", "2025-03-05", "2025-04-20", "2025-05-12", "2025-06-08"
            ]),
            "ord_stat": ["C", "C", "P", "X", "C", "C"],
            "sales_channel": ["Direct", "Partner", "Direct", "Direct", "Partner", "Direct"],
            "rev_net_usd": [12000.0, 9500.0, 6200.0, 1500.0, 8800.0, 14100.0],
            "cogs_usd": [4000.0, 3100.0, 2200.0, 700.0, 2900.0, 4500.0],
        }),
    )
    conn.ingest_dataframe(
        "returns",
        pd.DataFrame({
            "return_id": ["R1", "R2"],
            "order_id": ["O4", "O2"],
            "product_id": ["P4", "P2"],
            "customer_id": ["C4", "C2"],
            "refund_amt_usd": [1500.0, 800.0],
            "ret_reason_cd": ["DEF", "LATE"],
        }),
    )

    ws.tables = [
        TableSchema(
            name="orders",
            table_role=TableRole.FACT,
            row_count=10,
            columns=[
                ColumnProfile(name="order_id", data_type="VARCHAR", is_primary_key=True),
                ColumnProfile(name="customer_id", data_type="VARCHAR", is_foreign_key=True),
                ColumnProfile(name="product_id", data_type="VARCHAR", is_foreign_key=True),
                ColumnProfile(name="order_date", data_type="DATE"),
                ColumnProfile(
                    name="ord_stat",
                    data_type="VARCHAR",
                    distinct_count=4,
                    sample_values=["C", "P", "X", "H"],
                ),
                ColumnProfile(
                    name="sales_channel",
                    data_type="VARCHAR",
                    distinct_count=2,
                    sample_values=["Direct", "Partner"],
                ),
                ColumnProfile(name="rev_net_usd", data_type="DOUBLE"),
                ColumnProfile(name="cogs_usd", data_type="DOUBLE"),
            ],
        ),
        TableSchema(
            name="customers",
            table_role=TableRole.DIMENSION,
            row_count=5,
            columns=[
                ColumnProfile(name="customer_id", data_type="VARCHAR", is_primary_key=True),
                ColumnProfile(
                    name="region_code",
                    data_type="VARCHAR",
                    distinct_count=4,
                    sample_values=["NA", "EMEA", "APAC", "LATAM"],
                ),
                ColumnProfile(
                    name="cust_tier",
                    data_type="VARCHAR",
                    distinct_count=3,
                    sample_values=["ENT", "MID", "SMB"],
                ),
            ],
        ),
        TableSchema(
            name="products",
            table_role=TableRole.DIMENSION,
            row_count=5,
            columns=[
                ColumnProfile(name="product_id", data_type="VARCHAR", is_primary_key=True),
                ColumnProfile(
                    name="category",
                    data_type="VARCHAR",
                    distinct_count=4,
                    sample_values=["Cloud", "Security", "Analytics", "Hardware"],
                ),
                ColumnProfile(name="sku_name", data_type="VARCHAR", distinct_count=5),
            ],
        ),
        TableSchema(
            name="returns",
            table_role=TableRole.FACT,
            row_count=3,
            columns=[
                ColumnProfile(name="return_id", data_type="VARCHAR", is_primary_key=True),
                ColumnProfile(name="order_id", data_type="VARCHAR", is_foreign_key=True),
                ColumnProfile(name="product_id", data_type="VARCHAR", is_foreign_key=True),
                ColumnProfile(name="customer_id", data_type="VARCHAR", is_foreign_key=True),
                ColumnProfile(name="refund_amt_usd", data_type="DOUBLE"),
                ColumnProfile(name="ret_reason_cd", data_type="VARCHAR"),
            ],
        ),
    ]
    ws.relationships = [
        RelationshipEdge(
            id="r_ord_cust",
            source_table="orders",
            source_column="customer_id",
            target_table="customers",
            target_column="customer_id",
            status=JoinStatus.APPROVED,
        ),
        RelationshipEdge(
            id="r_ord_prod",
            source_table="orders",
            source_column="product_id",
            target_table="products",
            target_column="product_id",
            status=JoinStatus.APPROVED,
        ),
        RelationshipEdge(
            id="r_ret_prod",
            source_table="returns",
            source_column="product_id",
            target_table="products",
            target_column="product_id",
            status=JoinStatus.APPROVED,
        ),
    ]
    ws.measures = [
        SemanticMeasure(
            id="m_net_rev",
            name="Net Recognized Revenue",
            table_name="orders",
            dax_expression="SUM('orders'[rev_net_usd])",
            sql_expression="SUM(orders.rev_net_usd)",
            involved_tables=["orders"],
            synonyms=["net revenue", "recognized revenue"],
        )
    ]
    okf_store.save(ws)
    rag = RAGStore(tmp_path / "workspaces" / "retail_test")
    return ws, okf_store, rag, conn


def _make_academic_env(tmp_path: Path) -> tuple[OKFWorkspace, OKFStore, RAGStore, DuckDBConnector]:
    okf_store = OKFStore(base_dir=tmp_path / "workspaces")
    ws = OKFWorkspace(workspace_id="acad_test", workspace_name="University Test")
    conn = DuckDBConnector(tmp_path / "workspaces" / "acad_test")
    conn.ingest_dataframe(
        "colleges",
        pd.DataFrame({"college_id": [1, 2], "college_name": ["College of Engineering", "College of Arts"]}),
    )
    conn.ingest_dataframe(
        "departments",
        pd.DataFrame({
            "department_id": [10, 20],
            "college_id": [1, 2],
            "department_name": ["Computer Science", "History & Philosophy"],
            "budget_allocated": [500000.0, 250000.0],
        }),
    )
    conn.ingest_dataframe(
        "degree_programs",
        pd.DataFrame({
            "program_id": [100, 200],
            "department_id": [10, 20],
            "program_name": ["B.S. CS", "B.A. Philosophy"],
        }),
    )
    conn.ingest_dataframe(
        "students",
        pd.DataFrame({
            "student_id": [1, 2, 3],
            "college_id": [1, 1, 2],
            "program_id": [100, 100, 200],
            "gender": ["F", "M", "F"],
        }),
    )
    conn.ingest_dataframe(
        "faculty",
        pd.DataFrame({
            "faculty_id": [1, 2],
            "department_id": [10, 20],
            "academic_rank": ["Professor", "Associate"],
            "salary": [120000.0, 90000.0],
        }),
    )
    conn.ingest_dataframe(
        "fee_invoices",
        pd.DataFrame({
            "invoice_id": [1, 2, 3],
            "student_id": [1, 2, 3],
            "amount_paid": [15000.0, 15000.0, 12000.0],
            "net_amount_due": [15000.0, 15000.0, 14000.0],
        }),
    )
    conn.ingest_dataframe(
        "enrollments_marks",
        pd.DataFrame({
            "mark_id": [1, 2, 3],
            "student_id": [1, 2, 3],
            "marks_obtained": [92.5, 85.0, 88.0],
        }),
    )
    ws.tables = [
        TableSchema(
            name="colleges",
            table_role=TableRole.DIMENSION,
            row_count=2,
            columns=[
                ColumnProfile(name="college_id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(name="college_name", data_type="VARCHAR", distinct_count=2, sample_values=["College of Engineering", "College of Arts"]),
            ],
        ),
        TableSchema(
            name="departments",
            table_role=TableRole.DIMENSION,
            row_count=2,
            columns=[
                ColumnProfile(name="department_id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(name="college_id", data_type="INTEGER", is_foreign_key=True),
                ColumnProfile(name="department_name", data_type="VARCHAR", distinct_count=2, sample_values=["Computer Science", "History & Philosophy"]),
                ColumnProfile(name="budget_allocated", data_type="DOUBLE"),
            ],
        ),
        TableSchema(
            name="degree_programs",
            table_role=TableRole.DIMENSION,
            row_count=2,
            columns=[
                ColumnProfile(name="program_id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(name="department_id", data_type="INTEGER", is_foreign_key=True),
                ColumnProfile(name="program_name", data_type="VARCHAR", distinct_count=2, sample_values=["B.S. CS", "B.A. Philosophy"]),
            ],
        ),
        TableSchema(
            name="students",
            table_role=TableRole.DIMENSION,
            row_count=3,
            columns=[
                ColumnProfile(name="student_id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(name="college_id", data_type="INTEGER", is_foreign_key=True),
                ColumnProfile(name="program_id", data_type="INTEGER", is_foreign_key=True),
                ColumnProfile(name="gender", data_type="VARCHAR", distinct_count=2, sample_values=["F", "M"]),
            ],
        ),
        TableSchema(
            name="faculty",
            table_role=TableRole.FACT,
            row_count=2,
            columns=[
                ColumnProfile(name="faculty_id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(name="department_id", data_type="INTEGER", is_foreign_key=True),
                ColumnProfile(name="academic_rank", data_type="VARCHAR", distinct_count=2, sample_values=["Professor", "Associate"]),
                ColumnProfile(name="salary", data_type="DOUBLE"),
            ],
        ),
        TableSchema(
            name="fee_invoices",
            table_role=TableRole.FACT,
            row_count=3,
            columns=[
                ColumnProfile(name="invoice_id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(name="student_id", data_type="INTEGER", is_foreign_key=True),
                ColumnProfile(name="amount_paid", data_type="DOUBLE"),
                ColumnProfile(name="net_amount_due", data_type="DOUBLE"),
            ],
        ),
        TableSchema(
            name="enrollments_marks",
            table_role=TableRole.FACT,
            row_count=3,
            columns=[
                ColumnProfile(name="mark_id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(name="student_id", data_type="INTEGER", is_foreign_key=True),
                ColumnProfile(name="marks_obtained", data_type="DOUBLE"),
            ],
        ),
    ]
    ws.relationships = [
        RelationshipEdge(id="d_c", source_table="departments", source_column="college_id", target_table="colleges", target_column="college_id", status=JoinStatus.APPROVED),
        RelationshipEdge(id="dp_d", source_table="degree_programs", source_column="department_id", target_table="departments", target_column="department_id", status=JoinStatus.APPROVED),
        RelationshipEdge(id="s_c", source_table="students", source_column="college_id", target_table="colleges", target_column="college_id", status=JoinStatus.APPROVED),
        RelationshipEdge(id="s_dp", source_table="students", source_column="program_id", target_table="degree_programs", target_column="program_id", status=JoinStatus.APPROVED),
        RelationshipEdge(id="f_d", source_table="faculty", source_column="department_id", target_table="departments", target_column="department_id", status=JoinStatus.APPROVED),
        RelationshipEdge(id="fi_s", source_table="fee_invoices", source_column="student_id", target_table="students", target_column="student_id", status=JoinStatus.APPROVED),
        RelationshipEdge(id="em_s", source_table="enrollments_marks", source_column="student_id", target_table="students", target_column="student_id", status=JoinStatus.APPROVED),
    ]
    okf_store.save(ws)
    rag = RAGStore(tmp_path / "workspaces" / "acad_test")
    return ws, okf_store, rag, conn


class _StubLLMRouter:
    def __init__(self, responses: list[dict[str, Any] | Exception], model: str = "stub-model") -> None:
        self._responses = list(responses)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def is_configured(self) -> bool:
        return True

    def complete_json(self, purpose: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append({"purpose": purpose, "system_prompt": system_prompt, "user_prompt": user_prompt})
        item = self._responses.pop(0) if self._responses else {}
        if isinstance(item, Exception):
            raise item
        return item

    def complete_text(self, purpose: str, system_prompt: str, user_prompt: str) -> str:
        return ""

    def complete(self, purpose: str, system_prompt: str, user_prompt: str) -> str:
        return ""


# ==============================================================================
# 1. Strict Pydantic v2 State Validation, Round-Trip & Forensic Fixes #1-#6
# ==============================================================================


def test_strict_pydantic_rejects_arbitrary_objects_and_roundtrips():
    """Arbitrary non-serializable objects (DataFrames, DB connections) must raise ValidationError."""
    with pytest.raises(ValidationError):
        QueryGraphState(question="test", response_dict={"df": pd.DataFrame({"x": [1]})})

    with pytest.raises(ValidationError):
        SupervisorState(question="test", response_dict={"conn": duckdb.connect(":memory:")})

    edge = RelationshipEdge(
        id="e1",
        source_table="orders",
        source_column="customer_id",
        target_table="customers",
        target_column="customer_id",
        status=JoinStatus.APPROVED,
    )
    meas = SemanticMeasure(
        id="m1",
        name="Net Revenue",
        table_name="orders",
        dax_expression="SUM('orders'[rev_net_usd])",
        sql_expression="SUM(orders.rev_net_usd)",
        involved_tables=["orders"],
    )
    snap = JoinPlanSnapshot(required_tables=["orders", "customers"], join_edges=[edge], has_fan_trap=True)
    state = QueryGraphState(
        question="Show revenue by region",
        join_plan=snap,
        matched_measures=[meas],
        node_telemetry=[NodeTelemetryRecord(node="compile_initial_query", duration_ms=1.25, attempt=1)],
        error_diagnostic=ErrorDiagnostic(
            error_code="DB_CATALOG_ERROR",
            error_category="database",
            failed_node="execute_sandbox",
            user_message="Column missing",
        ),
    )
    restored = QueryGraphState.model_validate_json(state.model_dump_json())
    assert isinstance(restored.join_plan.join_edges[0], RelationshipEdge)
    assert restored.join_plan.has_fan_trap is True
    assert isinstance(restored.matched_measures[0], SemanticMeasure)
    assert restored.matched_measures[0].sql_expression == "SUM(orders.rev_net_usd)"
    assert restored.node_telemetry[0].duration_ms == pytest.approx(1.25)

    # Forensic Fix #1: to_join_plan preserves typed RelationshipEdge objects
    jp = restored.join_plan.to_join_plan()
    assert isinstance(jp.join_edges[0], RelationshipEdge)
    assert jp.join_edges[0].target_table == "customers"


def test_trace_graph_node_handles_command_and_propagates_interrupt():
    """trace_graph_node must wrap both dict and Command cleanly and never swallow GraphInterrupt."""
    @trace_graph_node("cmd_node")
    def cmd_node(state: QueryGraphState, config: Any) -> Command:
        return Command(update={"q_clean": "cleaned"}, goto="next_step")

    out_cmd = cmd_node(QueryGraphState(question="test"), {})
    assert isinstance(out_cmd, Command)
    assert out_cmd.goto == "next_step"
    assert out_cmd.update["node_trace"] == ["cmd_node"]
    assert out_cmd.update["node_telemetry"][0].updated_keys == ["q_clean"]

    @trace_graph_node("int_node")
    def int_node(state: QueryGraphState, config: Any) -> dict[str, Any]:
        raise GraphInterrupt("blocking")

    with pytest.raises(GraphInterrupt):
        int_node(QueryGraphState(question="test"), {})


# ==============================================================================
# 2. Individual Node Unit Tests + Case B2 Fan-Trap Writeback + Fix #2/#3/#4
# ==============================================================================


def test_all_11_nodes_and_routers_unit(tmp_path: Path):
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    ctx = QueryGraphContext(workspace=ws, okf_store=okf_store, rag_store=rag, connector=conn)
    cfg = {"configurable": {"thread_id": "unit_1", "dt_context": ctx}}

    # Node 1 + Forensic Fix #3 (zero What-If stripping + prompt extraction)
    s0 = QueryGraphState(
        question="What if revenue increases by 15% by category?",
        what_if_overrides={"Revenue Multiplier %": 0.0, "Cost Multiplier %": 0.0},
    )
    u1 = ingest_and_resolve_followup(s0, cfg)
    s1 = s0.model_copy(update=u1)
    assert s1.what_if_overrides == {"Revenue Multiplier %": pytest.approx(0.15)}
    assert route_after_followup(s1) == "check_verified_memory"

    # Node 2: dataset_overview
    s_ov = s1.model_copy(update={"is_overview": True, "resolved_question": "What is this dataset about?"})
    assert route_after_followup(s_ov) == "dataset_overview"
    u2 = dataset_overview(s_ov, cfg)
    assert u2["response_dict"]["success"] is True
    assert not ctx.dataframe_registry["result"].empty

    # Node 3: Verified Memory miss -> retrieve_okf_and_rag
    cmd3_miss = check_verified_memory(s1, cfg)
    assert cmd3_miss.goto == "retrieve_okf_and_rag"

    # Node 3: Verified Memory hit -> infers involved_tables (Forensic Fix #4)
    ws.verified_queries.append(
        VerifiedQuery(
            id="vq_cat",
            question_pattern="What if revenue increases by 15% by category?",
            executed_query="SELECT products.category, SUM(orders.rev_net_usd) AS net_revenue_usd FROM orders LEFT JOIN products ON orders.product_id = products.product_id GROUP BY products.category",
            query_dialect="duckdb",
        )
    )
    cmd3_hit = check_verified_memory(s1, cfg)
    assert cmd3_hit.goto == "__end__"
    assert "orders" in cmd3_hit.update["response_dict"]["involved_tables"]
    assert "products" in cmd3_hit.update["response_dict"]["involved_tables"]
    ws.verified_queries.clear()

    # Node 4: retrieve_okf_and_rag (non_blocking_inline clarification hint)
    u4 = retrieve_okf_and_rag(s1, cfg)
    s4 = s1.model_copy(update=u4)
    assert s4.clarification_severity == "non_blocking_inline"
    assert "ord_stat" in (s4.clarification_suggestion or "")
    assert route_after_retrieval(s4, cfg) == "compile_initial_query"

    # Node 5: build_out_of_domain
    s_ood = s4.model_copy(update={"out_of_domain": True, "q_clean": "unicorn quantum flux"})
    assert route_after_retrieval(s_ood, cfg) == "build_out_of_domain"
    u5 = build_out_of_domain(s_ood, cfg)
    assert u5["error_diagnostic"].error_code == "OUT_OF_DOMAIN"

    # Node 7 & Node 8 & Node 10: compile -> execute -> post_process
    u7 = compile_initial_query(s4, cfg)
    s7 = s4.model_copy(update=u7)
    cmd8 = execute_sandbox(s7, cfg)
    assert cmd8.goto == "post_process_and_summarize"
    s8 = s7.model_copy(update=cmd8.update)
    u10 = post_process_and_summarize(s8, cfg)
    assert u10["response_dict"]["success"] is True
    assert "What-If (Revenue Multiplier %: +15.0%)" in u10["response_dict"]["measures_applied"]


def test_case_b2_fan_trap_writeback_preserved(tmp_path: Path):
    """Forensic Fix #1: Academic expenditure vs earnings sets has_fan_trap=True inside JoinPlanSnapshot."""
    ws, okf_store, rag, conn = _make_academic_env(tmp_path)
    res = QueryAgent.answer_question(
        question="Compare department expenditure vs fee earnings across colleges",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
    )
    resp, df = res
    assert resp.success is True
    assert resp.fan_trap_mitigated is True
    assert "WITH dept_budget_cte" in resp.executed_query
    assert not df.empty


# ==============================================================================
# 3. 3x Self-Healing Critic Loop, Pre-Flight AST Guard & Measure Preservation
# ==============================================================================


def test_critic_loop_ast_guard_and_measure_preservation(tmp_path: Path):
    """AST-violating LLM repair must be blocked inside critic_repair_query and fall back with measures intact."""
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    llm_stub = _StubLLMRouter(
        responses=[
            {"sql_query": "SELECT missing_bogus_col FROM orders", "dax_query": "EVALUATE orders"},
            {"repaired_query": "DROP TABLE orders; SELECT 1"},  # AST firewall must reject inside critic node!
        ]
    )
    resp, df = QueryAgent.answer_question(
        question="Net Recognized Revenue by category with 10% revenue shock",
        what_if_overrides={"Revenue Multiplier %": 0.10},
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
        llm_router=llm_stub,  # type: ignore[arg-type]
    )
    assert resp.success is True
    assert resp.attempts_used == 2
    assert any("Pre-flight AST Firewall Guard intercepted unsafe repair" in line for line in resp.healing_trace)
    assert "Net Recognized Revenue" in resp.measures_applied
    assert "What-If (Revenue Multiplier %: +10.0%)" in resp.measures_applied
    assert "SUM(orders.rev_net_usd)" in resp.executed_query or "* 1.1000" in resp.executed_query
    assert not df.empty


def test_exhausted_retries_produce_structured_error_diagnostic(tmp_path: Path):
    """After MAX_RETRIES=3 failed executions, handle_exhausted_failure emits redacted ErrorDiagnostic."""
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    # Stub connector failing deterministically to test Node 8 -> Node 9 loop cap -> Node 11
    class _BrokenConnector:
        dialect = "duckdb"

        def execute_query(self, sql: str, **kwargs: Any) -> pd.DataFrame:
            raise RuntimeError("Catalog Error: Table orders leaked user@example.com")

    resp, df = QueryAgent.answer_question(
        question="Show total revenue by category",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=_BrokenConnector(),  # type: ignore[arg-type]
    )
    assert resp.success is False
    assert df.empty
    assert resp.attempts_used == 3
    assert resp.error_diagnostic is not None
    assert resp.error_diagnostic.error_code == "DB_CATALOG_ERROR"
    assert resp.error_diagnostic.error_category == "database"
    assert "user@example.com" not in resp.error_message
    assert "<REDACTED_EMAIL_PII>" in resp.error_message
    assert resp.node_trace.count("execute_sandbox") == 3
    assert resp.node_trace.count("critic_repair_query") == 2
    assert "handle_exhausted_failure" in resp.node_trace


def test_error_diagnostic_classification_matrix():
    for msg, expected_code, expected_cat in [
        ("SecurityViolationError: Multi-statement DDL/DML blocked", "AST_SECURITY_VIOLATION", "security"),
        ("Catalog Error: Column foo does not exist", "DB_CATALOG_ERROR", "database"),
        ("Parser Error: syntax error near SELECT", "SQL_SYNTAX_ERROR", "sql_compilation"),
        ("LiteLLM API Key HTTP 401 unauthorized", "LLM_PROVIDER_ERROR", "llm_provider"),
        ("No matching tables found in workspace", "OUT_OF_DOMAIN", "domain"),
        ("Recursion limit of 25 reached", "GRAPH_RECURSION_LIMIT", "internal"),
    ]:
        diag = classify_error_diagnostic(msg, "execute_sandbox", attempt=2)
        assert diag.error_code == expected_code
        assert diag.error_category == expected_cat


# ==============================================================================
# 4. Dual Clarification Severity (Blocking Interrupt vs Non-Blocking Inline) & Thread Isolation
# ==============================================================================


def test_blocking_modal_interrupt_and_resume_concurrent_isolation(tmp_path: Path):
    """Cross-source pending join pauses via interrupt() when allow_interrupts=True without disturbing concurrent threads."""
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    ws.relationships[0].is_cross_source = True
    ws.relationships[0].status = JoinStatus.PENDING
    okf_store.save(ws)

    cp = build_default_checkpointer()

    # Thread 1 pauses cleanly at clarification_interrupt
    res1 = run_query_graph(
        question="Show revenue by region",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
        allow_interrupts=True,
        thread_id="sess_A:retail_test:chat:r1",
        checkpointer=cp,
    )
    assert res1["interrupted"] is True
    assert res1["interrupt_payload"]["severity"] == "blocking_modal"
    assert res1["interrupt_payload"]["type"] == "cross_source_join"

    # Concurrent Thread 2 (single-table product query, non-blocking) runs cleanly on the same checkpointer
    res2 = run_query_graph(
        question="Show revenue by category",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
        allow_interrupts=True,
        thread_id="sess_B:retail_test:chat:r2",
        checkpointer=cp,
    )
    assert res2["interrupted"] is False
    assert res2["response_dict"]["success"] is True

    # Resume Thread 1 approving cross-source join -> persists BusinessRule & finishes cleanly!
    resumed = resume_query_graph(
        thread_id="sess_A:retail_test:chat:r1",
        resume_payload={"answer": "Yes, approve cross-source join", "save_rule": True},
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
        checkpointer=cp,
    )
    assert resumed["interrupted"] is False
    assert resumed["response_dict"]["success"] is True
    assert ws.relationships[0].status == JoinStatus.APPROVED
    assert len(ws.business_rules) >= 1


# ==============================================================================
# 5. Telemetry Bounds & SupervisorGraph Routing
# ==============================================================================


def test_telemetry_and_supervisor_routing(tmp_path: Path):
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    cp = build_default_checkpointer()

    # 1. Descriptive SQL via Supervisor -> enters QueryAgentSubgraph
    res_sql = run_supervisor_graph(
        question="Show net revenue by category",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
        checkpointer=cp,
        thread_id="sup_sql",
    )
    assert res_sql["response_dict"]["success"] is True
    assert "classify_bi_intent" in res_sql["node_trace"]
    assert "compile_initial_query" in res_sql["node_trace"]
    step_sum = sum(t["duration_ms"] for t in res_sql["node_telemetry"])
    assert step_sum <= res_sql["total_latency_ms"] + 1e-6

    # 2. Diagnostic variance -> routes to RootCauseAgent
    res_rc = run_supervisor_graph(
        question="Why did revenue change? Run root cause variance analysis",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
        checkpointer=cp,
        thread_id="sup_rc",
    )
    assert "run_root_cause" in res_rc["node_trace"]
    assert res_rc["response_dict"]["compiler_mode"] == "supervisor_root_cause"

    # 3. Forecast / Simulation -> routes to SimulationEngine
    res_sim = run_supervisor_graph(
        question="Forecast next 4 periods revenue",
        workspace=ws,
        okf_store=okf_store,
        rag_store=rag,
        connector=conn,
        checkpointer=cp,
        thread_id="sup_sim",
    )
    assert "run_simulation" in res_sim["node_trace"]
    assert res_sim["response_dict"]["compiler_mode"] == "supervisor_simulation"


# ==============================================================================
# 6. All 9 HTTP Endpoints, SSE Stream, Resume & Simulation Unit Fix (#3)
# ==============================================================================


def test_all_9_langgraph_http_endpoints_and_what_if_normalization(test_client: TestClient, test_runtime: AppRuntime):
    # Load demo dataset
    assert test_client.post("/api/connect/demo").status_code == 200

    # 1. GET /api/state exposes LangGraph metadata
    st_res = test_client.get("/api/state")
    assert st_res.status_code == 200
    st_json = st_res.json()
    assert st_json["langgraph_enabled"] is True
    assert st_json["checkpointer_mode"] == "in_memory"

    # 2. POST /api/chat/query (normal offline query + zero-shock prompt What-If Fix #3!)
    q_res = test_client.post(
        "/api/chat/query",
        json={"question": "What if revenue increases by 15% by category?", "revenue_shock_pct": 0.0},
    )
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["entry"]["response"]["success"] is True
    assert "What-If (Revenue Multiplier %: +15.0%)" in q_data["entry"]["response"]["measures_applied"]
    assert len(q_data["entry"]["response"]["node_trace"]) >= 4
    assert len(q_data["entry"]["response"]["node_telemetry"]) >= 4

    # 3. POST /api/chat/query with use_supervisor=True
    sup_q = test_client.post(
        "/api/chat/query",
        json={"question": "Forecast next 4 periods revenue", "use_supervisor": True},
    )
    assert sup_q.status_code == 200
    assert sup_q.json()["entry"]["response"]["compiler_mode"] == "supervisor_simulation"

    # 4. POST /api/chat/query/stream SSE frames
    with test_client.stream(
        "POST",
        "/api/chat/query/stream",
        json={"question": "Show net revenue by region"},
    ) as s_resp:
        assert s_resp.status_code == 200
        body = "".join(chunk for chunk in s_resp.iter_text())
    assert "event: stage" in body
    assert "event: node_update" in body
    assert "event: clarification_hint" in body
    assert "event: complete" in body

    # 5. Blocking Interrupt + POST /api/chat/query/resume (and 404 on invalid thread_id)
    ws = test_runtime.get_workspace("enterprise_demo")
    for rel in ws.relationships:
        if "customers" in (rel.source_table, rel.target_table):
            rel.is_cross_source = True
            rel.status = JoinStatus.PENDING
    test_runtime.okf_store.save(ws)

    paused_res = test_client.post(
        "/api/chat/query",
        json={"question": "Show revenue by region", "allow_interrupts": True, "force_recompile": True},
    )
    assert paused_res.status_code == 200
    paused_entry = paused_res.json()["entry"]
    assert paused_entry["interrupted"] is True
    tid = paused_entry["thread_id"]
    assert paused_res.json()["state"]["pending_interrupt"]["thread_id"] == tid

    resumed_res = test_client.post(
        "/api/chat/query/resume",
        json={"thread_id": tid, "answer": "Yes, approve join", "save_rule": True},
    )
    assert resumed_res.status_code == 200
    assert resumed_res.json()["entry"]["response"]["success"] is True
    assert resumed_res.json()["entry"]["resumed_from_thread_id"] == tid
    assert resumed_res.json()["state"]["pending_interrupt"] is None

    assert (
        test_client.post(
            "/api/chat/query/resume",
            json={"thread_id": "nonexistent_thread_999", "answer": "ok"},
        ).status_code
        == 404
    )

    # 6. GET /api/chat/graph-topology (query & supervisor)
    topo_q = test_client.get("/api/chat/graph-topology?graph_type=query").json()
    assert topo_q["graph_type"] == "query"
    assert "execute_sandbox" in topo_q["mermaid"]
    topo_s = test_client.get("/api/chat/graph-topology?graph_type=supervisor").json()
    assert topo_s["graph_type"] == "supervisor"
    assert "classify_bi_intent" in topo_s["mermaid"]

    # 7. Simulation stores normalized decimal ratio (0.15, not 15.0) & Chat Thread inherits telemetry
    sim_res = test_client.post("/api/simulation/run", json={"revenue_shock_pct": 15.0})
    assert sim_res.status_code == 200
    thr_res = test_client.post(
        "/api/chat/thread",
        json={"entry_idx": 0, "question": "Break down by region"},
    )
    assert thr_res.status_code == 200
    assert thr_res.json()["turn"]["source_mode"] == "sql_query"
    assert len(thr_res.json()["turn"]["node_trace"]) >= 3

    # 8. improve-prompt, teach, clear
    assert test_client.post("/api/chat/improve-prompt", json={"question": "rev by region"}).status_code == 200
    assert test_client.post("/api/chat/teach", json={"question": "rev", "rule_text": "Exclude X"}).status_code == 200
    clr = test_client.post("/api/chat/clear")
    assert clr.status_code == 200
    assert clr.json()["chat_history"] == []


# ==============================================================================
# 7. Canonical sqlglot AST Equivalence, Fuzzing, Large Schema Pruning & Latency
# ==============================================================================


@pytest.mark.parametrize(
    "question",
    [
        "Show revenue and profit by category",
        "Monthly revenue trend over time",
        "Compare revenue and returns by category",
        "Top 3 customers tier by net revenue",
    ],
)
def test_sqlglot_canonical_ast_equivalence_retail(tmp_path: Path, question: str):
    """Compiled LangGraph SQL must parse cleanly and match deterministic compiler canonical sqlglot AST."""
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    resp, df = QueryAgent.answer_question(question, ws, okf_store, rag, conn)
    assert resp.success is True
    assert resp.compiler_mode == "heuristic"
    assert not df.empty

    tables = detect_relevant_tables(question, ws, {})
    plan = okf_store.get_join_plan(ws, tables)
    measures = match_measures(question, ws)
    expected_sql = compile_deterministic_sql(question, ws, plan, measures)

    ast_graph = sqlglot.parse_one(resp.executed_query, read="duckdb").sql(pretty=False)
    ast_direct = sqlglot.parse_one(expected_sql, read="duckdb").sql(pretty=False)
    assert ast_graph == ast_direct


@pytest.mark.parametrize(
    "noisy_question",
    [
        "Show revenue\u2011net by \u2018category\u2019",
        "salery by each every department",
        "top 2 colleges with average marks",
        "what if revenue increases by 12.5% by region",
        "'; DROP TABLE orders; -- show revenue by category",
        "   ",
    ],
)
def test_property_based_input_fuzzing_safe(tmp_path: Path, noisy_question: str):
    """Noisy/adversarial inputs never crash the graph and any executed SQL remains AST read-only safe."""
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    resp, _ = QueryAgent.answer_question(noisy_question, ws, okf_store, rag, conn)
    assert isinstance(resp, QueryResponse)
    if resp.success and not resp.executed_query.startswith("/*"):
        PrivacyFirewall.validate_sql_read_only(resp.executed_query, dialect="duckdb")


def test_large_schema_adaptive_pruning_and_expansion(tmp_path: Path):
    """Workspaces > 20 tables split into Tier-1 full + Tier-2 compact and self-heal on missed tables."""
    okf_store = OKFStore(base_dir=tmp_path / "ws")
    ws = OKFWorkspace(workspace_id="enterprise_25", workspace_name="Enterprise 25 Tables")
    ws.tables = [
        TableSchema(
            name=f"tbl_{i:02d}",
            table_role=TableRole.DIMENSION if i > 2 else TableRole.FACT,
            columns=[
                ColumnProfile(name="id", data_type="INTEGER", is_primary_key=True),
                ColumnProfile(
                    name=f"metric_{i:02d}",
                    data_type="DOUBLE",
                    sample_values=["val_a", "val_b", "val_c"],
                ),
            ],
        )
        for i in range(1, 26)
    ]
    ws.relationships = [
        RelationshipEdge(
            id="r1",
            source_table="tbl_01",
            source_column="id",
            target_table="tbl_05",
            target_column="id",
            status=JoinStatus.APPROVED,
        )
    ]
    plan = okf_store.get_join_plan(ws, ["tbl_01"])
    tier1, tier2 = select_adaptive_schema_tiers(ws, plan, {})
    assert len(tier1) + len(tier2) == 25
    assert {t.name for t in tier1} >= {"tbl_01", "tbl_05"}  # 1-hop neighbor included in Tier-1!
    assert len(tier2) >= 20

    schema_text = build_adaptive_schema_context(ws, plan, {})
    assert "Tier-2 Compact Directory" in schema_text
    assert "`tbl_23`" in schema_text

    # Self-healing expansion promotes a Tier-2 table referenced in SQL/error into Tier-1
    expanded = expand_schema_from_error(
        question="Show metric_23",
        failed_query="SELECT metric_23 FROM tbl_23",
        error_msg="Catalog Error: tbl_23 not in join plan",
        workspace=ws,
        current_tables=["tbl_01"],
    )
    assert "tbl_23" in expanded

    # <= 20 tables -> zero pruning
    ws.tables = ws.tables[:15]
    tier1_small, tier2_small = select_adaptive_schema_tiers(ws, plan, {})
    assert len(tier1_small) == 15
    assert tier2_small == []


def test_latency_benchmarks_offline_graph(tmp_path: Path):
    """Offline heuristic graph (<50ms median) and Verified Memory (<20ms median) stay sub-50ms."""
    ws, okf_store, rag, conn = _make_retail_env(tmp_path)
    cp = build_default_checkpointer()

    # Warm-up compile & SQLite cache
    run_query_graph("Show net revenue by category", ws, okf_store, rag, conn, checkpointer=cp)

    heuristic_times: list[float] = []
    for i in range(20):
        t0 = time.perf_counter()
        res = run_query_graph(
            "Show net revenue by category",
            ws,
            okf_store,
            rag,
            conn,
            force_recompile=True,
            thread_id=f"bench_h_{i}",
            checkpointer=cp,
        )
        heuristic_times.append((time.perf_counter() - t0) * 1000.0)
        assert res["response_dict"]["success"] is True

    ws.verified_queries.append(
        VerifiedQuery(
            id="vq_bench",
            question_pattern="Show net revenue by category",
            executed_query="SELECT products.category, SUM(orders.rev_net_usd) AS net_revenue_usd FROM orders LEFT JOIN products ON orders.product_id = products.product_id GROUP BY products.category",
            query_dialect="duckdb",
        )
    )
    # Warm-up initial VQ hit
    run_query_graph("Show net revenue by category", ws, okf_store, rag, conn, thread_id="bench_vq_warm", checkpointer=cp)
    vq_times: list[float] = []
    for i in range(20):
        t0 = time.perf_counter()
        res_vq = run_query_graph(
            "Show net revenue by category",
            ws,
            okf_store,
            rag,
            conn,
            thread_id=f"bench_vq_{i}",
            checkpointer=cp,
        )
        vq_times.append((time.perf_counter() - t0) * 1000.0)
        assert res_vq["response_dict"]["from_verified_memory"] is True

    assert statistics.median(heuristic_times) < 50.0, f"Heuristic median too high: {statistics.median(heuristic_times):.2f}ms"
    assert statistics.median(vq_times) < 20.0, f"Verified Memory median too high: {statistics.median(vq_times):.2f}ms"
